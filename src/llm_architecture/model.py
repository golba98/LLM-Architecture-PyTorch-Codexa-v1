"""Decoder-only Transformer components for the initial smoke model."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ._attention_ops import append_cache, apply_rotary, causal_attention, split_heads
from ._model_inputs import prepare_model_inputs

LayerKeyValue = tuple[torch.Tensor, torch.Tensor]
KeyValueCache = tuple[LayerKeyValue, ...]


@dataclass
class ModelConfig:
    """Architecture settings for the decoder-only language model."""

    vocab_size: int = 8192
    context_length: int = 256
    num_layers: int = 8
    hidden_size: int = 384
    num_heads: int = 6
    intermediate_size: int = 1024
    dropout: float = 0.0
    tie_embeddings: bool = True
    position_embedding_type: str = "learned"
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        integer_dimensions = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
        }
        for name, value in integer_dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(
                    f"{name} must be a positive integer; got {value!r}."
                )

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads; "
                f"got hidden_size={self.hidden_size} and num_heads={self.num_heads}."
            )

        if (
            not isinstance(self.dropout, (int, float))
            or isinstance(self.dropout, bool)
            or not 0.0 <= self.dropout <= 1.0
        ):
            raise ValueError(
                f"dropout must be between 0.0 and 1.0; got {self.dropout!r}."
            )

        if self.position_embedding_type not in {"learned", "rotary"}:
            raise ValueError(
                "position_embedding_type must be 'learned' or 'rotary'; "
                f"got {self.position_embedding_type!r}."
            )
        if self.position_embedding_type == "rotary" and (
            self.hidden_size // self.num_heads
        ) % 2:
            raise ValueError("Rotary attention requires an even head dimension.")
        if not isinstance(self.rope_theta, (int, float)) or self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive; got {self.rope_theta!r}.")


class RMSNorm(nn.Module):
    """Root mean square normalization with stable float32 accumulation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive for RMSNorm; got {hidden_size}."
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be positive; got {eps}.")

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_dtype = inputs.dtype
        inputs_float = inputs.float()
        variance = inputs_float.pow(2).mean(dim=-1, keepdim=True)
        normalized = inputs_float * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(dtype=input_dtype)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention backed by PyTorch SDPA."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dimension = config.hidden_size // config.num_heads
        self.dropout = float(config.dropout)
        self.position_embedding_type = config.position_embedding_type
        self.rope_theta = float(config.rope_theta)

        self.qkv_projection = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=False,
        )
        self.output_projection = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: LayerKeyValue | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, LayerKeyValue]:
        if inputs.ndim != 3:
            raise ValueError(
                "attention input must have shape [batch, sequence, hidden_size]; "
                f"got shape {tuple(inputs.shape)}."
            )
        if inputs.shape[-1] != self.hidden_size:
            raise ValueError(
                "attention input hidden dimension must match hidden_size; "
                f"got {inputs.shape[-1]} and expected {self.hidden_size}."
            )

        batch_size, sequence_length, _ = inputs.shape
        query, key, value = self.qkv_projection(inputs).chunk(3, dim=-1)
        query = split_heads(query, self.num_heads)
        key = split_heads(key, self.num_heads)
        value = split_heads(value, self.num_heads)

        if self.position_embedding_type == "rotary":
            if position_ids is None or position_ids.shape != (batch_size, sequence_length):
                raise ValueError("Rotary attention requires position_ids matching inputs.")
            query, key = apply_rotary(query, key, position_ids, self.rope_theta)

        key, value, past_length = append_cache(key, value, past_key_value)
        attention_output = causal_attention(
            query,
            key,
            value,
            attention_mask,
            past_length,
            self.dropout if self.training else 0.0,
        )
        attention_output = attention_output.transpose(1, 2).contiguous().reshape(
            batch_size,
            sequence_length,
            self.hidden_size,
        )
        output = self.output_projection(attention_output)
        if use_cache:
            return output, (key, value)
        return output


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_projection = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_projection = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_projection(inputs)) * self.up_projection(inputs)
        return self.down_projection(gated)


class TransformerBlock(nn.Module):
    """Pre-normalized Transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.feed_forward = FeedForward(config)

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: LayerKeyValue | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, LayerKeyValue]:
        attention_result = self.attention(
            self.attention_norm(inputs),
            attention_mask,
            position_ids,
            past_key_value,
            use_cache,
        )
        if use_cache:
            attention_output, present_key_value = attention_result
        else:
            attention_output = attention_result
        hidden_states = inputs + attention_output
        output = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        if use_cache:
            return output, present_key_value
        return output


class LanguageModel(nn.Module):
    """Decoder-only causal language model."""

    supports_kv_cache = True

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.position_embeddings = (
            nn.Embedding(config.context_length, config.hidden_size)
            if config.position_embedding_type == "learned"
            else None
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        self.apply(self._initialize_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Enable activation recomputation for Transformer blocks."""

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean.")
        self.gradient_checkpointing = enabled

    def resize_token_embeddings(
        self,
        new_vocab_size: int,
        *,
        seed: int = 42,
    ) -> None:
        """Grow token embeddings while preserving rows and weight tying."""

        old_vocab_size = self.config.vocab_size
        if new_vocab_size < old_vocab_size:
            raise ValueError("Token embeddings cannot be shrunk.")
        if new_vocab_size == old_vocab_size:
            return
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        old_embeddings = self.token_embeddings
        old_lm_head_weight = self.lm_head.weight.detach().clone()
        device = old_embeddings.weight.device
        dtype = old_embeddings.weight.dtype
        replacement = nn.Embedding(
            new_vocab_size,
            self.config.hidden_size,
            device=device,
            dtype=dtype,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        with torch.no_grad():
            replacement.weight[:old_vocab_size].copy_(old_embeddings.weight)
            replacement.weight[old_vocab_size:].normal_(
                mean=0.0,
                std=0.02,
                generator=generator,
            )
        self.token_embeddings = replacement
        self.lm_head = nn.Linear(
            self.config.hidden_size,
            new_vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        if self.config.tie_embeddings:
            self.lm_head.weight = self.token_embeddings.weight
        else:
            with torch.no_grad():
                self.lm_head.weight[:old_vocab_size].copy_(
                    old_lm_head_weight
                )
                self.lm_head.weight[old_vocab_size:].normal_(
                    mean=0.0,
                    std=0.02,
                    generator=generator,
                )
        self.config.vocab_size = new_vocab_size

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: KeyValueCache | None = None,
        use_cache: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor | None, KeyValueCache]
    ):
        """Run the decoder, optionally continuing from cached keys and values.

        ``attention_mask`` marks real (True) and padding (False) tokens. Its shape
        is either ``[batch, past_length + sequence]``, covering cached and current
        tokens, or ``[batch, sequence]``, covering current tokens only, in which
        case the cached prefix is treated as fully valid. Padding inside a cache
        is honored only through the full form, which the caller keeps across
        steps; the cache itself stores no mask.

        Without ``position_ids``, positions count real tokens across the cached
        and current tokens (``cumsum(mask) - 1``), or are
        ``past_length .. past_length + sequence - 1`` when no mask is given.
        """

        attention_mask, positions = prepare_model_inputs(
            input_ids,
            labels,
            attention_mask,
            position_ids,
            past_key_values,
            use_cache,
            self.config,
        )
        hidden_states = self.token_embeddings(input_ids.to(dtype=torch.long))
        if self.position_embeddings is not None:
            hidden_states = hidden_states + self.position_embeddings(positions)

        present_key_values: list[LayerKeyValue] = []
        for layer_index, block in enumerate(self.blocks):
            past_key_value = (
                None if past_key_values is None else past_key_values[layer_index]
            )
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    raise ValueError(
                        "use_cache=True is incompatible with gradient checkpointing."
                    )
                hidden_states = checkpoint(
                    block,
                    hidden_states,
                    attention_mask,
                    positions,
                    use_reentrant=False,
                )
            else:
                block_result = block(
                    hidden_states,
                    attention_mask,
                    positions,
                    past_key_value,
                    use_cache,
                )
                if use_cache:
                    hidden_states, present_key_value = block_result
                    present_key_values.append(present_key_value)
                else:
                    hidden_states = block_result

        logits = self.lm_head(self.final_norm(hidden_states))
        loss: torch.Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.to(dtype=torch.long).reshape(-1),
                ignore_index=-100,
            )

        if use_cache:
            return logits, loss, tuple(present_key_values)
        return logits, loss


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count unique model parameters, including safely tied parameters once."""

    seen_parameters: set[int] = set()
    total = 0
    for _, parameter in model.named_parameters(remove_duplicate=False):
        identity = id(parameter)
        if identity in seen_parameters:
            continue
        seen_parameters.add(identity)
        if trainable_only and not parameter.requires_grad:
            continue
        total += parameter.numel()
    return total
