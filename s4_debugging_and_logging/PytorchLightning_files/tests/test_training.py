import torch
import pytest
from s4_debugging_and_logging.PytorchLightning_files.model import JoelsAwesomeModel


def test_training():
    """Training-step sanity checks.

    - Loss is finite (not NaN/inf) after one step.
    - Output shape matches (batch_size, 10).
    - Backward pass works.
    - Model parameters change after one optimizer step.
    - Accuracy is within [0, 1].
    """
    torch.manual_seed(0)
    model = JoelsAwesomeModel()
    model.train()

    batch_size = 4
    images = torch.randn(batch_size, 1, 28, 28)
    targets = torch.randint(0, 10, (batch_size,))

    outputs = model(images)
    assert outputs.shape == (batch_size, 10)

    loss = model.loss_fn(outputs, targets)
    assert torch.isfinite(loss)

    loss.backward()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    optimizer.step()

    after = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    any_changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
    assert any_changed

    acc = (targets == outputs.argmax(dim=-1)).float().mean().item()
    assert 0.0 <= acc <= 1.0


def test_training_step_runs_and_returns_loss():
    """Directly exercise Lightning's training_step for coverage."""
    torch.manual_seed(0)
    model = JoelsAwesomeModel()
    model.train()

    batch_size = 4
    images = torch.randn(batch_size, 1, 28, 28)
    targets = torch.randint(0, 10, (batch_size,))

    loss_train = model.training_step((images, targets))
    assert torch.isfinite(loss_train)

    loss_val = model.validation_step((images, targets))
    assert torch.isfinite(loss_val)


def test_optimizer():
    model = JoelsAwesomeModel()

    lr = 1e-3
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == lr
