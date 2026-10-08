"""Lightweight executable tests for the smoke language model."""

import math
from collections.abc import Callable
from pathlib import Path
import sys

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_architecture.model import LanguageModel, ModelConfig, count_parameters


SMALL_CONFIG = ModelConfig(
    vocab_size=128,
    context_length=16,
    num_layers=2,
    hidden_size=32,
    num_heads=4,
    intermediate_size=64,
    dropout=0.0,
    tie_embeddings=True,
)
EXPECTED_DEFAULT_PARAMETER_COUNT = 17_406_336


def assert_raises(
    exception_type: type[BaseException],
    operation: Callable[[], object],
    message_fragment: str,
) -> None:
    """Assert that an operation raises the expected informative exception."""

    try:
        operation()
    except exception_type as error:
        assert message_fragment in str(error), (
            f"Expected {message_fragment!r} in error message, got {error!r}."
        )
    else:
        raise AssertionError(f"Expected {exception_type.__name__} to be raised.")


def test_configuration_validation() -> None:
    """Check invalid dimensions, attention geometry, and dropout values."""

    dimension_names = (
        "vocab_size",
        "context_length",
        "num_layers",
        "hidden_size",
        "num_heads",
        "intermediate_size",
    )
    for name in dimension_names:
        assert_raises(
            ValueError,
            lambda name=name: ModelConfig(**{name: 0}),
            f"{name} must be a positive integer",
        )

    assert_raises(
        ValueError,
        lambda: ModelConfig(hidden_size=30, num_heads=8),
        "hidden_size must be divisible by num_heads",
    )
    assert_raises(
        ValueError,
        lambda: ModelConfig(dropout=-0.1),
        "dropout must be between 0.0 and 1.0",
    )
    assert_raises(
        ValueError,
        lambda: ModelConfig(dropout=1.1),
        "dropout must be between 0.0 and 1.0",
    )


def test_input_validation() -> None:
    """Check public forward-input validation."""

    model = LanguageModel(SMALL_CONFIG)
    valid_input = torch.randint(
        0,
        SMALL_CONFIG.vocab_size,
        (2, 8),
        dtype=torch.long,
    )
    assert_raises(
        ValueError,
        lambda: model(
            torch.randint(
                0,
                SMALL_CONFIG.vocab_size,
                (1, SMALL_CONFIG.context_length + 1),
            )
        ),
        "exceeds context_length",
    )
    assert_raises(
        ValueError,
        lambda: model(valid_input, labels=valid_input[:, :-1]),
        "labels must have the same shape as input_ids",
    )
    assert_raises(
        TypeError,
        lambda: model(valid_input.float()),
        "input_ids must use an integer dtype",
    )
    assert_raises(
        ValueError,
        lambda: model(valid_input.unsqueeze(0)),
        "input_ids must have shape [batch, sequence]",
    )


def test_weight_tying() -> None:
    """Verify tied and untied output-head behavior and unique counting."""

    tied_model = LanguageModel(SMALL_CONFIG)
    assert tied_model.lm_head.weight is tied_model.token_embeddings.weight

    untied_config = ModelConfig(
        vocab_size=SMALL_CONFIG.vocab_size,
        context_length=SMALL_CONFIG.context_length,
        num_layers=SMALL_CONFIG.num_layers,
        hidden_size=SMALL_CONFIG.hidden_size,
        num_heads=SMALL_CONFIG.num_heads,
        intermediate_size=SMALL_CONFIG.intermediate_size,
        dropout=SMALL_CONFIG.dropout,
        tie_embeddings=False,
    )
    untied_model = LanguageModel(untied_config)
    assert untied_model.lm_head.weight is not untied_model.token_embeddings.weight
    assert (
        untied_model.lm_head.weight.data_ptr()
        != untied_model.token_embeddings.weight.data_ptr()
    )
    assert count_parameters(untied_model) - count_parameters(tied_model) == (
        SMALL_CONFIG.vocab_size * SMALL_CONFIG.hidden_size
    )


def test_embedding_resize_and_padding_mask() -> None:
    config = ModelConfig(**vars(SMALL_CONFIG))
    model = LanguageModel(config)
    original = model.token_embeddings.weight.detach().clone()
    original_count = count_parameters(model)
    model.resize_token_embeddings(config.vocab_size + 4, seed=42)
    assert model.config.vocab_size == SMALL_CONFIG.vocab_size + 4
    assert torch.equal(
        model.token_embeddings.weight[: SMALL_CONFIG.vocab_size],
        original,
    )
    assert model.lm_head.weight is model.token_embeddings.weight
    assert count_parameters(model) == original_count + 4 * SMALL_CONFIG.hidden_size

    input_ids = torch.tensor([[1, 7, 0], [1, 0, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    logits, _ = model(input_ids, attention_mask=attention_mask)
    assert logits.shape == (2, 3, SMALL_CONFIG.vocab_size + 4)


def run_cuda_tests() -> tuple[str, str]:
    """Run CUDA and BF16 smoke forwards when the runtime supports them."""

    if not torch.cuda.is_available():
        return "skipped (CUDA unavailable)", "skipped (CUDA unavailable)"

    device = torch.device("cuda")
    input_ids = torch.randint(
        0,
        SMALL_CONFIG.vocab_size,
        (2, 8),
        device=device,
    )
    labels = torch.randint(
        0,
        SMALL_CONFIG.vocab_size,
        (2, 8),
        device=device,
    )

    cuda_model = LanguageModel(SMALL_CONFIG).to(device).eval()
    with torch.inference_mode():
        logits, loss = cuda_model(input_ids, labels)
    assert logits.shape == (2, 8, SMALL_CONFIG.vocab_size)
    assert loss is not None and loss.ndim == 0 and torch.isfinite(loss)
    cuda_result = "passed"

    if not torch.cuda.is_bf16_supported():
        return cuda_result, "skipped (BF16 unsupported)"

    bf16_model = LanguageModel(SMALL_CONFIG).to(
        device=device,
        dtype=torch.bfloat16,
    ).eval()
    with torch.inference_mode():
        bf16_logits, bf16_loss = bf16_model(input_ids, labels)
    assert bf16_logits.dtype == torch.bfloat16
    assert torch.isfinite(bf16_logits).all()
    assert bf16_loss is not None and torch.isfinite(bf16_loss)
    return cuda_result, "passed"


def test_gradient_checkpointing() -> None:
    """Checkpointed blocks preserve outputs and finite gradients."""

    torch.manual_seed(7)
    reference = LanguageModel(SMALL_CONFIG)
    checkpointed = LanguageModel(SMALL_CONFIG)
    checkpointed.load_state_dict(reference.state_dict())
    checkpointed.set_gradient_checkpointing(True)
    assert checkpointed.gradient_checkpointing
    assert_raises(
        TypeError,
        lambda: checkpointed.set_gradient_checkpointing(1),  # type: ignore[arg-type]
        "enabled must be a boolean",
    )
    input_ids = torch.randint(
        0,
        SMALL_CONFIG.vocab_size,
        (2, 8),
    )
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    reference.train()
    checkpointed.train()
    reference_logits, reference_loss = reference(input_ids, labels)
    checkpointed_logits, checkpointed_loss = checkpointed(input_ids, labels)
    assert torch.equal(reference_logits, checkpointed_logits)
    assert reference_loss is not None and checkpointed_loss is not None
    assert torch.equal(reference_loss, checkpointed_loss)
    checkpointed_loss.backward()
    gradients = [
        parameter.grad
        for parameter in checkpointed.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def main() -> None:
    """Run all smoke-model tests and print a concise environment summary."""

    default_model = LanguageModel(ModelConfig())
    default_parameter_count = count_parameters(default_model)
    assert default_parameter_count > 0
    assert default_parameter_count == EXPECTED_DEFAULT_PARAMETER_COUNT
    del default_model

    torch.manual_seed(42)
    test_configuration_validation()
    test_weight_tying()
    test_embedding_resize_and_padding_mask()
    test_gradient_checkpointing()

    model = LanguageModel(SMALL_CONFIG)
    assert count_parameters(model) > 0
    assert count_parameters(model, trainable_only=True) == count_parameters(model)
    test_input_validation()

    batch_size = 2
    sequence_length = 12
    input_ids = torch.randint(
        0,
        SMALL_CONFIG.vocab_size,
        (batch_size, sequence_length),
        dtype=torch.long,
    )
    labels = torch.randint(
        0,
        SMALL_CONFIG.vocab_size,
        (batch_size, sequence_length),
        dtype=torch.long,
    )

    model.eval()
    with torch.inference_mode():
        logits_without_labels, loss_without_labels = model(input_ids)
    expected_shape = (
        batch_size,
        sequence_length,
        SMALL_CONFIG.vocab_size,
    )
    assert logits_without_labels.shape == expected_shape
    assert loss_without_labels is None

    model.train()
    logits, loss = model(input_ids, labels)
    assert logits.shape == expected_shape
    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert abs(loss.item() - math.log(SMALL_CONFIG.vocab_size)) < 1.0

    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients, "Backward propagation did not create any gradients."
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    cuda_result, bf16_result = run_cuda_tests()
    gpu_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "not available"
    )

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {gpu_name}")
    print(f"Smoke-model parameter count: {default_parameter_count:,}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Initial loss: {loss.item():.6f}")
    print(f"CUDA test: {cuda_result}")
    print(f"BF16 CUDA test: {bf16_result}")
    print("All model tests passed.")


if __name__ == "__main__":
    main()
