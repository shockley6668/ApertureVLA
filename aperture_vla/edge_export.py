"""Tensor-only VLA interface preserving the release's text padding layout."""
from collections import OrderedDict

import torch
from torch import nn


INPUT_NAMES = ("pixel_values", "normalized_state", "input_ids", "token_type_ids", "position_ids",
               "text_token_mask", "text_self_attention_mask", "bert_hidden_mask")


def prepare_text_inputs(encoder, instructions, device="cpu"):
    output_length = encoder.config.padding_length
    if not instructions or output_length is None or output_length < 1:
        raise ValueError("Export requires instructions and a fixed positive text padding length")
    count = len(instructions)
    input_ids = torch.full((count, output_length), encoder.tokenizer.pad_token_id, dtype=torch.long, device=device)
    token_type_ids = torch.zeros_like(input_ids)
    position_ids = torch.zeros_like(input_ids)
    token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    attention_mask = torch.eye(output_length, dtype=torch.bool, device=device).expand(count, -1, -1).clone()
    hidden_mask = torch.zeros(count, output_length, 1, dtype=torch.float32, device=device)
    for index, instruction in enumerate(instructions):
        length = int(encoder.config.padding_length_by_instruction.get(instruction, output_length))
        if not 1 <= length <= output_length:
            raise ValueError(f"Instruction padding length {length} exceeds export shape {output_length}")
        tokenized, self_attention, positions = encoder._tokenize_group([instruction], torch.device(device), length)
        input_ids[index, :length] = tokenized["input_ids"][0]
        token_type_ids[index, :length] = tokenized.get("token_type_ids", torch.zeros_like(tokenized["input_ids"]))[0]
        position_ids[index, :length] = positions[0]
        token_mask[index, :length] = tokenized["attention_mask"][0].bool()
        attention_mask[index, :length, :length] = self_attention[0]
        hidden_mask[index, :length] = 1.0
    return OrderedDict(zip(INPUT_NAMES[2:], (input_ids, token_type_ids, position_ids,
                                           token_mask, attention_mask, hidden_mask)))


class TensorTextTurboVLA(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, normalized_state, input_ids, token_type_ids, position_ids,
                text_token_mask, text_self_attention_mask, bert_hidden_mask):
        encoder = self.model.text_encoder
        bert_inputs = {"input_ids": input_ids, "token_type_ids": token_type_ids,
                       "attention_mask": text_self_attention_mask if encoder.config.sub_sentence_present else text_token_mask}
        if encoder.config.sub_sentence_present:
            bert_inputs["position_ids"] = position_ids
        hidden = encoder.bert(**bert_inputs).last_hidden_state
        hidden = hidden * bert_hidden_mask.to(dtype=hidden.dtype)
        text_tokens = encoder.text_projection(hidden.to(dtype=encoder.text_projection.weight.dtype))
        padding_mask = ~text_token_mask
        if encoder.config.zero_padded_tokens:
            text_tokens = text_tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        visual_tokens = self.model.encode_vision(pixel_values)
        visual_tokens, text_tokens = self.model.vision_language_interaction(
            visual_tokens=visual_tokens, text_tokens=text_tokens,
            text_key_padding_mask=padding_mask, text_self_attention_masks=text_self_attention_mask,
        )
        condition = torch.cat((visual_tokens, text_tokens), dim=1)
        dtype = self.model.action_head.decoder.action_queries.weight.dtype
        return self.model.action_head(condition.to(dtype=dtype), normalized_state.to(dtype=dtype))
