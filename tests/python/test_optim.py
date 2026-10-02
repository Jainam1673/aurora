"""Unit and convergence tests for AURORA optimizers, schedulers, and checkpoints."""

import tempfile

import numpy as np
from aurora.checkpoint import load_checkpoint, save_checkpoint
from aurora.nn.layers import MLP, Linear
from aurora.nn.parameter import Parameter
from aurora.optim.adamw import AdamW
from aurora.optim.optimizer import clip_grad_norm, clip_grad_value
from aurora.optim.scheduler import CosineAnnealingLR, LinearWarmupDecayLR
from aurora.optim.sgd import SGD
from aurora.tensor import tensor


def test_sgd_quadratic_convergence() -> None:
    # Minimize f(w) = (w - 3.0)^2
    w = Parameter([0.0])
    target = tensor([3.0])
    opt = SGD([w], lr=0.05, momentum=0.8)

    init_loss = ((w - target) * (w - target)).item()
    for _ in range(60):
        opt.zero_grad()
        loss = (w - target) * (w - target)
        loss.backward()
        opt.step()

    final_loss = ((w - target) * (w - target)).item()
    assert final_loss < init_loss * 1e-3
    assert np.isclose(w.item(), 3.0, atol=2e-2)


def test_adam_and_adamw_convergence() -> None:
    # Linear regression: y = 2*x + 1
    x = tensor([[1.0], [2.0], [3.0], [4.0]])
    y_true = tensor([[3.0], [5.0], [7.0], [9.0]])

    model = Linear(1, 1)
    opt = AdamW(model.parameters(), lr=0.1, weight_decay=0.0)

    init_loss = ((model(x) - y_true) * (model(x) - y_true)).mean().item()
    for _ in range(250):
        opt.zero_grad()
        pred = model(x)
        diff = pred - y_true
        loss = (diff * diff).mean()
        loss.backward()
        opt.step()

    final_loss = ((model(x) - y_true) * (model(x) - y_true)).mean().item()
    assert final_loss < init_loss * 1e-2
    assert final_loss < 0.02


def test_gradient_clipping() -> None:
    p1 = Parameter([10.0, 20.0])
    p1.grad = tensor([30.0, 40.0])  # norm = 50.0

    total_norm = clip_grad_norm([p1], max_norm=5.0)
    assert np.isclose(total_norm, 50.0)
    # Scaled down by 5 / 50 = 0.1
    assert p1.grad is not None
    assert np.allclose(p1.grad.numpy(), [3.0, 4.0])

    p2 = Parameter([1.0, 2.0])
    p2.grad = tensor([-10.0, 15.0])
    clip_grad_value([p2], clip_value=5.0)
    assert p2.grad is not None
    assert np.allclose(p2.grad.numpy(), [-5.0, 5.0])


def test_schedulers() -> None:
    p = Parameter([1.0])
    opt = SGD([p], lr=0.1)

    sched_warmup = LinearWarmupDecayLR(opt, warmup_steps=10, total_steps=100, min_lr=0.01)
    # Check warmup ramp
    assert np.isclose(opt.defaults["lr"], 0.0)
    for _ in range(10):
        sched_warmup.step()
    assert np.isclose(opt.defaults["lr"], 0.1, atol=1e-5)

    sched_cos = CosineAnnealingLR(opt, total_steps=100, min_lr=0.0)
    for _ in range(100):
        sched_cos.step()
    assert np.isclose(opt.defaults["lr"], 0.0, atol=1e-5)


def test_checkpoint_roundtrip() -> None:
    mlp = MLP(in_features=4, hidden_dims=[8], out_features=2)
    opt = AdamW(mlp.parameters(), lr=1e-3)

    # Perform a step to populate optimizer state
    x = tensor([[1.0, 2.0, 3.0, 4.0]])
    loss = mlp(x).sum()
    loss.backward()
    opt.step()

    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        save_checkpoint(tmp.name, mlp, opt, metadata={"test": "roundtrip"})

        mlp_loaded = MLP(in_features=4, hidden_dims=[8], out_features=2)
        opt_loaded = AdamW(mlp_loaded.parameters(), lr=1e-3)

        payload = load_checkpoint(tmp.name, mlp_loaded, opt_loaded)
        assert payload["metadata"]["test"] == "roundtrip"

        # Check weights are identical
        for (n1, p1), (n2, p2) in zip(
            mlp.named_parameters(), mlp_loaded.named_parameters(), strict=False
        ):
            assert n1 == n2
            assert np.allclose(p1.numpy(), p2.numpy())

        # Check optimizer step count
        assert opt_loaded.step_count == opt.step_count
