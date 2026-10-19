"""Frozen online action supervision, kept outside the deployable student."""
from collections import deque
import math

import torch


def masked_action_mse(prediction, teacher_target, mask):
    if prediction.shape != teacher_target.shape or prediction.ndim != 3:
        raise ValueError("Student and teacher actions must have matching [B,T,A] shapes")
    if tuple(mask.shape) != tuple(prediction.shape[:2]):
        raise ValueError("Action mask must have shape [B,T]")
    weights = mask.unsqueeze(-1).float()
    squared = (prediction.float() - teacher_target.detach().float()).square()
    denominator = (weights.sum() * prediction.shape[-1]).clamp_min(1.0)
    return (squared * weights).sum() / denominator


class OnlineActionDistillation:
    """Pair each training forward with teacher targets; never register teacher weights.

    The hook runs under no_grad, using the same normalized images/states and
    instructions as the student. A target can be consumed by exactly one loss.
    Teacher construction must happen outside the student's random-number stream.
    """

    def __init__(self, teacher, supervised_loss, weight=0.5, micro_batch=1,
                 teacher_dtype=torch.bfloat16, log_every=20):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Distillation weight must be finite and nonnegative")
        if micro_batch < 1 or log_every < 1:
            raise ValueError("Micro-batch and log interval must be positive")
        self.teacher = teacher
        self.supervised_loss = supervised_loss
        self.weight = float(weight)
        self.micro_batch = int(micro_batch)
        self.teacher_dtype = teacher_dtype
        self.log_every = int(log_every)
        self.target = None
        self.batches = 0
        self.recent = deque(maxlen=log_every)
        self.handle = None
        self.teacher_device = next(teacher.parameters()).device

    def attach(self, student):
        if self.handle is not None:
            raise RuntimeError("Distillation is already attached")
        if {id(p) for p in student.parameters()} & {id(p) for p in self.teacher.parameters()}:
            raise ValueError("Teacher and student must have independent parameters")
        self.teacher.to(dtype=self.teacher_dtype).eval().requires_grad_(False)
        self.handle = student.register_forward_pre_hook(self._before_forward)

    @torch.no_grad()
    def _before_forward(self, student, inputs):
        if not student.training:
            return
        if self.target is not None:
            raise RuntimeError("Previous teacher target was not consumed")
        if self.weight == 0:
            return
        instructions, samples, states = inputs
        batch = len(instructions)
        if states.shape[0] != batch or any(x.shape[0] != batch for x in samples.values()):
            raise ValueError("Teacher input batch sizes disagree")
        if self.teacher_device != states.device:
            self.teacher.to(device=states.device, dtype=self.teacher_dtype)
            self.teacher_device = states.device
        self.teacher.eval()
        pieces = []
        for start in range(0, batch, self.micro_batch):
            stop = min(batch, start + self.micro_batch)
            chunk = {key: value[start:stop].to(dtype=self.teacher_dtype)
                     if value.is_floating_point() else value[start:stop]
                     for key, value in samples.items()}
            # The student's surrounding AMP context must not change the
            # teacher's native inference precision (notably its action head).
            with torch.autocast(device_type=states.device.type, enabled=False):
                piece = self.teacher(instructions[start:stop], chunk,
                                     states[start:stop].to(dtype=self.teacher_dtype))
            pieces.append(piece.detach().float())
        self.target = torch.cat(pieces, dim=0)
        if not torch.isfinite(self.target).all():
            raise ValueError("Teacher produced non-finite actions")

    def loss(self, prediction, truth, mask):
        supervised = self.supervised_loss(prediction, truth, mask)
        if self.weight == 0:
            kd = supervised.detach().new_zeros(())
        else:
            if self.target is None:
                raise RuntimeError("Missing teacher target for this student forward")
            target, self.target = self.target, None
            kd = masked_action_mse(prediction, target, mask)
        total = supervised + self.weight * kd
        self.batches += 1
        self.recent.append((float(supervised.detach()), float(kd.detach()),
                            float(total.detach())))
        if self.batches % self.log_every == 0:
            print("[action_distillation] " + str(self.diagnostics()), flush=True)
        return total

    def diagnostics(self):
        names = ("supervised_masked_l1", "teacher_masked_mse", "total_loss")
        return {"batches": self.batches, "window_batches": len(self.recent),
                "weight": self.weight,
                **{name: sum(row[i] for row in self.recent) / len(self.recent)
                   if self.recent else None for i, name in enumerate(names)}}

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.target = None
