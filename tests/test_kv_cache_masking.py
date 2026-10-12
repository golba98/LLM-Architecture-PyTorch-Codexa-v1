"""KV-cache, attention-mask and position-ID contract of LanguageModel.forward.

Contract under test:

* ``attention_mask`` is either ``[batch, past_length + sequence]`` (full mask,
  covering cached and current tokens) or ``[batch, sequence]`` (current tokens
  only; the cached prefix is then defined to be fully valid).
* Default positions are ``cumsum(full_mask) - 1`` restricted to the current
  tokens, i.e. they continue from the cached real-token count.
* Unmasked calls keep ``arange(past_length, past_length + sequence)``.
"""

import pytest
import torch

from llm_architecture.model import CausalSelfAttention, LanguageModel, ModelConfig

POSITIONS = ("learned", "rotary")
DEVICES = (
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
    ),
)
# FP32 tolerance shared with the existing cache-equivalence regressions.
FP32 = dict(atol=1e-5, rtol=1e-4)
CONTEXT = 16


def build(positions: str, device: str = "cpu", dtype: torch.dtype = torch.float32) -> LanguageModel:
    torch.manual_seed(1234)
    config = ModelConfig(
        vocab_size=64,
        context_length=CONTEXT,
        num_layers=2,
        hidden_size=32,
        num_heads=4,
        intermediate_size=64,
        position_embedding_type=positions,
    )
    model = LanguageModel(config)
    # At the 0.02 initialization attention is nearly uniform, so a wrong rotary
    # position moves logits by only ~1e-3. Sharpening attention makes position
    # errors observable for both position types (rotary reset error ~0.08).
    with torch.no_grad():
        for block in model.blocks:
            block.attention.qkv_projection.weight.mul_(8.0)
    return model.to(device=device, dtype=dtype).eval()


def tokens(*shape: int, device: str = "cpu") -> torch.Tensor:
    generator = torch.Generator().manual_seed(99)
    return torch.randint(3, 64, shape, generator=generator).to(device)


def pad_rows(rows: list[list[int]], side: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(row) for row in rows)
    ids, mask = [], []
    for row in rows:
        padding = [0] * (width - len(row))
        ids.append(padding + row if side == "left" else row + padding)
        valid = [False] * len(padding)
        mask.append(valid + [True] * len(row) if side == "left" else [True] * len(row) + valid)
    return torch.tensor(ids, device=device), torch.tensor(mask, device=device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("positions", POSITIONS)
@pytest.mark.parametrize("chunk", (1, 3))
@pytest.mark.parametrize("mask_mode", ("none", "current", "full"))
def test_incremental_decoding_matches_full_sequence(device, positions, chunk, mask_mode):
    model = build(positions, device)
    ids = tokens(2, 13, device=device)
    with torch.inference_mode():
        full = model(ids)[0]
        logits, _, cache = model(ids[:, :4], use_cache=True)
        pieces = [logits]
        for start in range(4, ids.shape[1], chunk):
            current = ids[:, start : start + chunk]
            past = cache[0][0].shape[2]
            if mask_mode == "none":
                mask = None
            elif mask_mode == "current":
                mask = torch.ones_like(current, dtype=torch.bool)
            else:
                mask = torch.ones((2, past + current.shape[1]), dtype=torch.bool, device=device)
            logits, _, cache = model(current, attention_mask=mask, past_key_values=cache, use_cache=True)
            pieces.append(logits)
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, **FP32)


@pytest.mark.parametrize("positions", POSITIONS)
def test_masked_cached_default_positions_continue_from_past(positions):
    """The confirmed defect: masked cached calls reset positions to zero."""

    model = build(positions)
    ids = tokens(1, 9)
    with torch.inference_mode():
        full = model(ids)[0][:, -1:]
        cache = model(ids[:, :8], use_cache=True)[2]
        mask = torch.ones((1, 1), dtype=torch.bool)
        default = model(ids[:, 8:], attention_mask=mask, past_key_values=cache, use_cache=True)[0]
        reset = model(
            ids[:, 8:],
            attention_mask=mask,
            position_ids=torch.zeros((1, 1), dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
        )[0]
    torch.testing.assert_close(default, full, **FP32)
    # The pre-repair reset positions are far outside the FP32 tolerance.
    assert (reset - full).abs().max() > 100 * FP32["atol"]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("positions", POSITIONS)
@pytest.mark.parametrize("side", ("left", "right"))
def test_padded_mixed_length_batch_cached_decoding(device, positions, side):
    model = build(positions, device)
    sequences = tokens(3, 9, device=device).tolist()
    lengths = [3, 6, 5]
    prompts = [sequence[:length] for sequence, length in zip(sequences, lengths)]
    ids, mask = pad_rows(prompts, side, device)
    with torch.inference_mode():
        references = [model(torch.tensor([sequence], device=device))[0][0] for sequence in sequences]
        prefill, _, cache = model(ids, attention_mask=mask, use_cache=True)
        assert torch.isfinite(prefill).all()
        for row, length in enumerate(lengths):
            torch.testing.assert_close(prefill[row][mask[row]], references[row][:length], **FP32)
        full_mask = mask
        for step in range(3):
            current = torch.tensor(
                [[sequence[length + step]] for sequence, length in zip(sequences, lengths)],
                device=device,
            )
            full_mask = torch.cat((full_mask, torch.ones((3, 1), dtype=torch.bool, device=device)), dim=1)
            logits, _, cache = model(current, attention_mask=full_mask, past_key_values=cache, use_cache=True)
            assert torch.isfinite(logits).all()
            for row, length in enumerate(lengths):
                torch.testing.assert_close(logits[row, -1], references[row][length + step], **FP32)


@pytest.mark.parametrize("positions", POSITIONS)
def test_current_token_mask_defines_cached_prefix_as_valid(positions):
    """Legacy [batch, sequence] masks equal a full mask whose past is all ones."""

    model = build(positions)
    ids, mask = pad_rows([[5, 6, 7], [8, 9, 10, 11, 12]], "left", "cpu")
    current = torch.tensor([[13, 0], [14, 15]])
    current_mask = torch.tensor([[True, False], [True, True]])
    with torch.inference_mode():
        cache = model(ids, use_cache=True)[2]
        legacy = model(current, attention_mask=current_mask, past_key_values=cache, use_cache=True)[0]
        explicit = model(
            current,
            attention_mask=torch.cat((torch.ones_like(mask), current_mask), dim=1),
            past_key_values=cache,
            use_cache=True,
        )[0]
    assert torch.equal(legacy, explicit)


@pytest.mark.parametrize("positions", POSITIONS)
def test_explicit_and_inferred_position_ids_agree(positions):
    model = build(positions)
    ids, mask = pad_rows([[5, 6, 7], [8, 9, 10, 11, 12]], "left", "cpu")
    current = torch.tensor([[13, 14], [15, 16]])
    with torch.inference_mode():
        cache = model(ids, attention_mask=mask, use_cache=True)[2]
        past = ids.shape[1]
        cases = (
            (None, torch.arange(past, past + 2).expand(2, -1)),
            (torch.ones((2, 2), dtype=torch.bool), torch.arange(past, past + 2).expand(2, -1)),
            (
                torch.cat((mask, torch.ones((2, 2), dtype=torch.bool)), dim=1),
                torch.tensor([[3, 4], [5, 6]]),
            ),
        )
        for attention_mask, position_ids in cases:
            inferred = model(current, attention_mask=attention_mask, past_key_values=cache, use_cache=True)[0]
            explicit = model(
                current,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )[0]
            assert torch.equal(inferred, explicit)
        # Uncached padded forward keeps cumsum positions.
        inferred = model(ids, attention_mask=mask)[0]
        explicit = model(ids, attention_mask=mask, position_ids=(mask.long().cumsum(-1) - 1).clamp(min=0))[0]
        assert torch.equal(inferred, explicit)


@pytest.mark.parametrize("positions", POSITIONS)
@pytest.mark.parametrize("past", (1, CONTEXT - 1))
@pytest.mark.parametrize("masked", (False, True))
def test_short_and_near_maximum_context(positions, past, masked):
    model = build(positions)
    ids = tokens(1, past + 1)
    with torch.inference_mode():
        full = model(ids)[0][:, -1:]
        cache = model(ids[:, :past], use_cache=True)[2]
        mask = torch.ones((1, past + 1), dtype=torch.bool) if masked else None
        cached = model(ids[:, past:], attention_mask=mask, past_key_values=cache, use_cache=True)[0]
    torch.testing.assert_close(cached, full, **FP32)


@pytest.mark.parametrize("positions", POSITIONS)
def test_context_overflow_is_rejected(positions):
    model = build(positions)
    ids = tokens(1, CONTEXT)
    with torch.inference_mode():
        cache = model(ids[:, : CONTEXT - 1], use_cache=True)[2]
        with pytest.raises(ValueError, match="context_length"):
            model(ids[:, :2], attention_mask=torch.ones((1, CONTEXT + 1), dtype=torch.bool),
                  past_key_values=cache, use_cache=True)
        with pytest.raises(ValueError, match="exceed context_length"):
            model(ids[:, :1], position_ids=torch.tensor([[CONTEXT]]))


@pytest.mark.parametrize("positions", POSITIONS)
def test_unmasked_paths_unchanged(positions):
    model = build(positions)
    ids = tokens(2, 10)
    with torch.inference_mode():
        full = model(ids)[0]
        assert torch.equal(full, model(ids, position_ids=torch.arange(10).expand(2, -1))[0])
        cache = model(ids[:, :6], use_cache=True)[2]
        cached = model(ids[:, 6:], past_key_values=cache, use_cache=True)[0]
        explicit = model(ids[:, 6:], position_ids=torch.arange(6, 10).expand(2, -1),
                         past_key_values=cache, use_cache=True)[0]
    assert torch.equal(cached, explicit)
    torch.testing.assert_close(cached, full[:, 6:], **FP32)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("positions", POSITIONS)
def test_bf16_padded_cached_decoding(device, positions):
    """BF16 has an 8-bit significand (eps 2**-8). Through two blocks a few
    roundings accumulate, so outputs are compared on the logit scale with a
    bound of four BF16 eps (2**-6) of the largest reference logit. The
    position-reset defect must remain clearly larger than this bound."""

    reference_model = build(positions, device)
    model = build(positions, device, torch.bfloat16)
    sequences = tokens(2, 9, device=device).tolist()
    lengths = [4, 7]
    ids, mask = pad_rows([s[:n] for s, n in zip(sequences, lengths)], "left", device)
    current = torch.tensor([[s[n]] for s, n in zip(sequences, lengths)], device=device)
    full_mask = torch.cat((mask, torch.ones((2, 1), dtype=torch.bool, device=device)), dim=1)
    with torch.inference_mode():
        cache = model(ids, attention_mask=mask, use_cache=True)[2]
        cached = model(current, attention_mask=full_mask, past_key_values=cache, use_cache=True)[0][:, -1]
        reset = model(current, attention_mask=full_mask, position_ids=torch.zeros_like(current),
                      past_key_values=cache, use_cache=True)[0][:, -1]
        for row, length in enumerate(lengths):
            compact = torch.tensor([sequences[row][: length + 1]], device=device)
            bf16_full = model(compact)[0][0, -1].float()
            fp32_full = reference_model(compact)[0][0, -1]
            bound = 2.0**-6 * fp32_full.abs().max().item()
            assert cached.dtype == torch.bfloat16 and torch.isfinite(cached).all()
            assert (cached[row].float() - bf16_full).abs().max().item() <= bound
            assert (cached[row].float() - fp32_full).abs().max().item() <= bound
            assert (bf16_full - fp32_full).abs().max().item() <= bound
            assert (reset[row].float() - fp32_full).abs().max().item() > 2 * bound


def test_attention_module_accepts_current_and_full_key_masks():
    torch.manual_seed(0)
    attention = CausalSelfAttention(ModelConfig(hidden_size=16, num_heads=4, context_length=8)).eval()
    past = (torch.randn(2, 4, 3, 4), torch.randn(2, 4, 3, 4))
    inputs = torch.randn(2, 2, 16)
    current = torch.tensor([[True, True], [True, False]])
    full = torch.cat((torch.ones((2, 3), dtype=torch.bool), current), dim=1)
    with torch.inference_mode():
        legacy = attention(inputs, current, None, past, False)
        explicit = attention(inputs, full, None, past, False)
        assert torch.equal(legacy, explicit)
        with pytest.raises(ValueError, match="attention_mask"):
            attention(inputs, torch.ones((2, 4), dtype=torch.bool), None, past, False)


@pytest.mark.parametrize("positions", POSITIONS)
def test_cache_mask_and_position_incompatibilities(positions):
    model = build(positions)
    ids = tokens(2, 6)
    with torch.inference_mode():
        cache = model(ids[:, :4], use_cache=True)[2]
        current = ids[:, 4:]
        invalid_masks = (
            torch.ones((2, 5), dtype=torch.bool),  # neither 2 nor 4 + 2
            torch.ones((3, 2), dtype=torch.bool),  # batch mismatch
            torch.ones((3, 6), dtype=torch.bool),  # batch mismatch, full length
            torch.ones((2,), dtype=torch.bool),  # not two-dimensional
        )
        for mask in invalid_masks:
            with pytest.raises(ValueError, match="attention_mask"):
                model(current, attention_mask=mask, past_key_values=cache, use_cache=True)
        with pytest.raises(ValueError, match="attention_mask"):
            model(ids, attention_mask=torch.ones((2, 7), dtype=torch.bool))
        with pytest.raises(ValueError, match="position_ids must match"):
            model(current, position_ids=torch.tensor([[4, 5]]), past_key_values=cache, use_cache=True)
        with pytest.raises(ValueError, match="non-negative"):
            model(current, position_ids=torch.tensor([[-1, 0], [0, 1]]), past_key_values=cache, use_cache=True)
        with pytest.raises(ValueError, match="same sequence length"):
            uneven = (cache[0], (cache[1][0][:, :, :3], cache[1][1][:, :, :3]))
            model(current, past_key_values=uneven, use_cache=True)
        with pytest.raises(ValueError, match="incompatible batch"):
            model(current[:1], past_key_values=cache, use_cache=True)
        with pytest.raises(ValueError, match="requires use_cache"):
            model(current, past_key_values=cache)
