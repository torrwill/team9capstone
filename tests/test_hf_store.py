import os

import pytest

from lsn.training.hf_store import HFStore


def test_hf_store_uses_token_arg():
    """When token is passed explicitly, it's used."""
    s = HFStore(repo="x/y", subfolder="run_v1", token="explicit-tok")
    assert s.token == "explicit-tok"


def test_hf_store_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env-tok")
    s = HFStore(repo="x/y", subfolder="run_v1", token=None)
    assert s.token == "env-tok"


def test_hf_store_no_token_no_env(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    s = HFStore(repo="x/y", subfolder="run_v1", token=None)
    assert s.token is None
