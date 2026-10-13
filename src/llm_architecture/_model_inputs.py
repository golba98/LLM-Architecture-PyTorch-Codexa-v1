"""Validate model inputs and derive positions for cached or padded decoding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .model import KeyValueCache, ModelConfig


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def prepare_model_inputs(
    input_ids: torch.Tensor,
    labels: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    past_key_values: KeyValueCache | None,
    use_cache: bool,
    config: ModelConfig,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Return the full key mask and positions for the current sequence."""
    if input_ids.ndim != 2:
        raise ValueError(
            "input_ids must have shape [batch, sequence]; "
            f"got shape {tuple(input_ids.shape)}."
        )
    if input_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(
            "input_ids must use an integer dtype; "
            f"got {input_ids.dtype}."
        )

    batch_size, sequence_length = input_ids.shape
    past_length = 0
    if past_key_values is not None:
        if len(past_key_values) != config.num_layers:
            raise ValueError("past_key_values must contain one entry per layer.")
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
        if any(cache[0].shape[2] != past_length for cache in past_key_values):
            raise ValueError("All cached layers must have the same sequence length.")
    if past_length + sequence_length > config.context_length:
        raise ValueError(
            "input sequence length exceeds context_length; "
            f"got {past_length + sequence_length} and maximum "
            f"{config.context_length}."
        )
    if past_key_values is not None and not use_cache:
        raise ValueError("past_key_values requires use_cache=True.")

    if labels is not None:
        if labels.shape != input_ids.shape:
            raise ValueError(
                "labels must have the same shape as input_ids; "
                f"got labels {tuple(labels.shape)} and "
                f"input_ids {tuple(input_ids.shape)}."
            )
        if labels.dtype not in _INTEGER_DTYPES:
            raise TypeError(f"labels must use an integer dtype; got {labels.dtype}.")

    if attention_mask is not None:
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.to(dtype=torch.bool)
        if attention_mask.shape == (batch_size, past_length + sequence_length):
            pass
        elif attention_mask.shape == input_ids.shape:
            attention_mask = torch.cat(
                (
                    torch.ones(
                        (batch_size, past_length),
                        dtype=torch.bool,
                        device=attention_mask.device,
                    ),
                    attention_mask,
                ),
                dim=1,
            )
        else:
            raise ValueError(
                "attention_mask must have shape [batch, past_length + sequence] "
                "or [batch, sequence]; got "
                f"{tuple(attention_mask.shape)} for input_ids "
                f"{tuple(input_ids.shape)} and past_length {past_length}."
            )

    if position_ids is None:
        if attention_mask is None:
            positions = torch.arange(
                past_length,
                past_length + sequence_length,
                device=input_ids.device,
                dtype=torch.long,
            )
            if config.position_embedding_type == "rotary":
                positions = positions.unsqueeze(0).expand(batch_size, -1)
        else:
            positions = attention_mask.long().cumsum(dim=-1) - 1
            positions = positions[:, past_length:].clamp_(min=0)
    else:
        if position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must match input_ids shape.")
        positions = position_ids.to(device=input_ids.device, dtype=torch.long)
        if int(positions.min().item()) < 0:
            raise ValueError("position_ids must be non-negative.")
    if int(positions.max().item()) >= config.context_length:
        raise ValueError("position_ids exceed context_length.")
    return attention_mask, positions
