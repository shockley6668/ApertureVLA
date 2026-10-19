"""Save shared-parameter EMA aliases consistently and publish atomically."""
import os
from pathlib import Path


def ema_state_with_aliases(model, state, ema_parameters):
    result = dict(state)
    converted = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if parameter not in ema_parameters or name not in state:
            continue
        if parameter not in converted:
            converted[parameter] = ema_parameters[parameter].detach().to(
                device="cpu", dtype=state[name].dtype
            )
        result[name] = converted[parameter]
    return result


def install_checkpoint_writer(trainer, model, compression=None, training_recipe=None,
                              training_diagnostics=None):
    from turbovla.training import pi05

    def save(payload, destination, *args, **kwargs):
        optimizer = pi05._ACTIVE_OPTIMIZER
        if isinstance(payload, dict) and "model_state_dict" in payload:
            payload = dict(payload)
            if optimizer is not None:
                payload["ema_model_state_dict"] = ema_state_with_aliases(
                    model, payload["model_state_dict"], optimizer._ema_params
                )
                payload["ema_decay"] = pi05.EMA_DECAY
            payload["training_protocol"] = "mixed_encoded_episode_interleave_v2"
            if compression is not None:
                payload["compression"] = compression
            if training_recipe is not None:
                payload["training_recipe"] = training_recipe
            if training_diagnostics is not None:
                payload["training_diagnostics"] = training_diagnostics()
        if isinstance(destination, (str, os.PathLike)):
            target = Path(destination)
            temporary = target.with_name(target.name + ".partial")
            pi05._ORIGINAL_TORCH_SAVE(payload, temporary, *args, **kwargs)
            os.replace(temporary, target)
        else:
            pi05._ORIGINAL_TORCH_SAVE(payload, destination, *args, **kwargs)

    trainer.torch.save = save
