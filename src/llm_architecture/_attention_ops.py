"""Tensor operations used by causal self-attention.

The public module owns parameters and the forward order. These helpers own the
RoPE, cache and masking details, so those contracts can be read in one place.
"""

import torch
import torch.nn.functional as F


def split_heads(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch_size, sequence_length, hidden_size = tensor.shape
    return tensor.reshape(
        batch_size, sequence_length, num_heads, hidden_size // num_heads
    ).transpose(1, 2)


def apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the first and second halves of every attention head."""
    half = query.shape[-1] // 2
    frequencies = torch.arange(0, half, device=query.device, dtype=torch.float32)
    frequencies = theta ** (-frequencies / half)
    angles = position_ids.to(dtype=torch.float32)[..., None] * frequencies
    cosines = angles.cos()[:, None, :, :].to(dtype=query.dtype)
    sines = angles.sin()[:, None, :, :].to(dtype=query.dtype)

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        first, second = tensor[..., :half], tensor[..., half : 2 * half]
        rotated = torch.cat(
            (first * cosines - second * sines, first * sines + second * cosines),
            dim=-1,
        )
        if tensor.shape[-1] > 2 * half:
            rotated = torch.cat((rotated, tensor[..., 2 * half :]), dim=-1)
        return rotated

    return rotate(query), rotate(key)


def append_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Validate a layer cache, then extend it with the current tokens."""
    if past_key_value is None:
        return key, value, 0

    past_key, past_value = past_key_value
    if past_key.ndim != 4 or past_value.ndim != 4:
        raise ValueError("Cached keys and values must be four-dimensional.")
    if past_key.shape[:2] != key.shape[:2] or past_value.shape[:2] != key.shape[:2]:
        raise ValueError("Cached keys and values have incompatible batch or head dimensions.")
    if past_key.shape[-1] != key.shape[-1] or past_value.shape[-1] != key.shape[-1]:
        raise ValueError("Cached keys and values have an incompatible head dimension.")
    if past_key.shape[2] != past_value.shape[2]:
        raise ValueError("Cached keys and values must have the same sequence length.")

    past_length = past_key.shape[2]
    key = torch.cat((past_key.to(dtype=key.dtype), key), dim=2)
    value = torch.cat((past_value.to(dtype=value.dtype), value), dim=2)
    return key, value, past_length


def causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_length: int,
    dropout_p: float,
) -> torch.Tensor:
    """Attend to current and cached tokens, honoring a full or current mask."""
    batch_size, _, sequence_length, _ = query.shape
    if attention_mask is None and past_length == 0:
        return F.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout_p, is_causal=True
        )

    causal = torch.ones(
        (sequence_length, past_length + sequence_length),
        dtype=torch.bool,
        device=query.device,
    )
    allowed = torch.tril(causal, diagonal=past_length)[None, None, :, :]
    if attention_mask is not None:
        # A current-token mask defines the entire cached prefix as valid.
        if attention_mask.shape == (batch_size, past_length + sequence_length):
            key_mask = attention_mask
        elif attention_mask.shape == (batch_size, sequence_length):
            past_mask = torch.ones(
                (batch_size, past_length), dtype=torch.bool, device=query.device
            )
            key_mask = torch.cat((past_mask, attention_mask), dim=1)
        else:
            raise ValueError(
                "attention_mask must match [batch, past_length + sequence] "
                "or [batch, sequence]."
            )
        allowed = allowed & key_mask[:, None, None, :]
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=allowed, dropout_p=dropout_p, is_causal=False
    )
