"""Teacher-only batching that preserves the legacy policy's batch-one shift."""
from types import MethodType
import torch
import torch.nn.functional as F


def _per_sample_manual_attention(self, v, l, attention_mask_v=None, attention_mask_l=None):
    batch, target_len, _ = v.size()
    source_len = l.size(1)
    shape = (batch * self.num_heads, -1, self.head_dim)
    queries = self._shape(self.v_proj(v) * self.scale, target_len, batch).view(*shape)
    keys = self._shape(self.l_proj(l), source_len, batch).view(*shape)
    values_v = self._shape(self.values_v_proj(v), target_len, batch).view(*shape)
    values_l = self._shape(self.values_l_proj(l), source_len, batch).view(*shape)
    scores = torch.bmm(queries, keys.transpose(1, 2))
    # The upstream batch-one policy subtracts a maximum across all heads and
    # tokens. Preserve that exact scope independently for each observation.
    maximum = scores.view(batch, self.num_heads, target_len, source_len).amax(dim=(1, 2, 3))
    scores = scores - maximum.repeat_interleave(self.num_heads).view(-1, 1, 1)
    if self.clamp_min_for_underflow:
        scores = torch.clamp(scores, min=-50000)
    if self.clamp_max_for_overflow:
        scores = torch.clamp(scores, max=50000)
    language_scores = scores.transpose(1, 2)
    language_scores = language_scores - language_scores.max(dim=-1, keepdim=True)[0]
    if self.clamp_min_for_underflow:
        language_scores = torch.clamp(language_scores, min=-50000)
    if self.clamp_max_for_overflow:
        language_scores = torch.clamp(language_scores, max=50000)
    if attention_mask_v is not None:
        mask = attention_mask_v[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
        language_scores.masked_fill_(mask, float('-inf'))
    if attention_mask_l is not None:
        mask = attention_mask_l[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
        scores.masked_fill_(mask, float('-inf'))
    probabilities_v = F.dropout(scores.softmax(dim=-1), p=self.dropout, training=self.training)
    probabilities_l = F.dropout(language_scores.softmax(dim=-1), p=self.dropout, training=self.training)
    output_v = torch.bmm(probabilities_v, values_l).view(batch, self.num_heads, target_len, self.head_dim)
    output_l = torch.bmm(probabilities_l, values_v).view(batch, self.num_heads, source_len, self.head_dim)
    output_v = output_v.transpose(1, 2).reshape(batch, target_len, self.embed_dim)
    output_l = output_l.transpose(1, 2).reshape(batch, source_len, self.embed_dim)
    return self.out_v_proj(output_v), self.out_l_proj(output_l)


def stabilize_teacher_batching(teacher):
    """Patch only legacy manual attention instances; no parameters are added."""
    from turbovla.models.components.fusion import BiMultiHeadAttention
    selected = []
    for name, module in teacher.named_modules():
        if isinstance(module, BiMultiHeadAttention) and module.attention_backend == 'manual':
            if not module.stable_softmax_2d:
                raise ValueError('Expected the release manual-attention stability shift')
            if hasattr(module, '_teacher_native_forward'):
                raise RuntimeError('Teacher attention is already patched')
            module._teacher_native_forward = module.forward
            module.forward = MethodType(_per_sample_manual_attention, module)
            selected.append(name)
    return selected


def restore_native_teacher_attention(teacher):
    for module in teacher.modules():
        if hasattr(module, '_teacher_native_forward'):
            module.forward = module._teacher_native_forward
            del module._teacher_native_forward
