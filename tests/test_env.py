import random

import numpy as np
import torch

from lsn import env


def test_set_seed_makes_python_random_deterministic():
    env.set_seed(42)
    a = [random.random() for _ in range(5)]
    env.set_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_makes_numpy_deterministic():
    env.set_seed(42)
    a = np.random.rand(5)
    env.set_seed(42)
    b = np.random.rand(5)
    assert (a == b).all()


def test_set_seed_makes_torch_deterministic():
    env.set_seed(42)
    a = torch.randn(5)
    env.set_seed(42)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_get_device_returns_torch_device():
    d = env.get_device(None)
    assert isinstance(d, torch.device)
    assert d.type in ("cuda", "cpu")


def test_get_device_respects_cpu_override():
    d = env.get_device("cpu")
    assert d.type == "cpu"


def test_configure_cudnn_does_not_crash():
    env.configure_cudnn(benchmark=True)
    env.configure_cudnn(benchmark=False)
