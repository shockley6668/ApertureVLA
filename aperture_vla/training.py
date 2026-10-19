"""Training fixes kept outside the pinned upstream checkout."""
from .shared_basis import SharedBasisLinear
import re
import math


def _keep_bert_eval(encoder, inputs):
    # Keep the release's frozen-BERT dropout behavior while allowing gradients.
    encoder.bert.eval()


def train_text_compression(model):
    encoder = model.text_encoder
    encoder.bert.requires_grad_(False)
    for module in encoder.bert.modules():
        if isinstance(module, SharedBasisLinear):
            module.requires_grad_(True)
    # Upstream uses this flag to enter no_grad(), independently of requires_grad.
    encoder.config.frozen = False
    encoder.bert.eval()
    encoder.register_forward_pre_hook(_keep_bert_eval)


def train_dense_text_ffns(model):
    """Match the compressed control's trainable BERT module locations."""
    encoder = model.text_encoder
    encoder.bert.requires_grad_(False)
    pattern = re.compile(r"encoder\.layer\.\d+\.(?:intermediate|output)\.dense")
    selected = 0
    for name, module in encoder.bert.named_modules():
        if pattern.fullmatch(name):
            module.requires_grad_(True)
            selected += 1
    if selected != 24:
        raise ValueError(f"Expected 24 dense BERT FFNs, found {selected}")
    encoder.config.frozen = False
    encoder.bert.eval()
    encoder.register_forward_pre_hook(_keep_bert_eval)


def train_full_text(model):
    """Allow recovery of retained BERT weights without changing dropout behavior."""
    encoder = model.text_encoder
    encoder.bert.requires_grad_(True)
    encoder.config.frozen = False
    encoder.bert.eval()
    encoder.register_forward_pre_hook(_keep_bert_eval)


def split_retained_text_groups(model, groups, retained_text_lr):
    """Give retained BERT parameters a separate LR; keep compressed FFNs at head LR."""
    if not math.isfinite(retained_text_lr) or retained_text_lr <= 0:
        raise ValueError("retained_text_lr must be finite and positive")
    retained = {id(p) for p in model.text_encoder.bert.parameters()}
    ffn_pattern = re.compile(r"encoder\.layer\.\d+\.(?:intermediate|output)\.dense")
    for name,module in model.text_encoder.bert.named_modules():
        if isinstance(module, SharedBasisLinear) or ffn_pattern.fullmatch(name):
            retained.difference_update(id(p) for p in module.parameters())
    result = []
    for group in groups:
        other = [p for p in group["params"] if id(p) not in retained]
        text = [p for p in group["params"] if id(p) in retained]
        if other:
            result.append({**group, "params": other})
        if text:
            result.append({**group, "params": text, "lr": retained_text_lr,
                           "name": "retained_text_" + group["name"]})
    return result


def install_retained_text_lr(pi05, retained_text_lr):
    """Keep upstream AdamW/EMA construction and split groups before scheduling."""
    original = pi05._ORIGINAL_BUILD_OPTIMIZER

    def build(model, args):
        optimizer, _ = original(model, args)
        optimizer.param_groups = split_retained_text_groups(
            model, optimizer.param_groups, retained_text_lr
        )
        args.full_text_finetuning = True
        args.retained_text_lr = retained_text_lr
        summary = [{"name": g["name"], "lr": g["lr"],
                    "weight_decay": g["weight_decay"],
                    "params": sum(p.numel() for p in g["params"])}
                   for g in optimizer.param_groups]
        return optimizer, summary

    pi05._ORIGINAL_BUILD_OPTIMIZER = build


def use_safe_mixed_dataset(train_mixed):
    from .mixed_data import SafeMixedRLDSDataset

    train_mixed.MixedSuiteStatsLiberoRLDSDataset._iter_mixed_episodes = (
        SafeMixedRLDSDataset._iter_mixed_episodes
    )
    train_mixed.MixedSuiteStatsLiberoRLDSDataset._process_image_pair = (
        SafeMixedRLDSDataset._process_image_pair
    )
