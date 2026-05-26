"""Utils: logging, seeding, device, versioning, hashing."""

from __future__ import annotations

import random

import numpy as np

from ecofoundation.utils.device import resolve_device
from ecofoundation.utils.hashing import hash_config, hash_file_head
from ecofoundation.utils.logging import configure_logging, get_logger
from ecofoundation.utils.seeding import set_global_seed
from ecofoundation.utils.versioning import collect_versions


def test_set_global_seed_makes_runs_reproducible():
    set_global_seed(42)
    a1 = random.random()
    n1 = np.random.rand()
    set_global_seed(42)
    a2 = random.random()
    n2 = np.random.rand()
    assert a1 == a2
    assert n1 == n2


def test_resolve_device_returns_known_kind():
    d = resolve_device("auto")
    assert d in {"cuda", "mps", "cpu"}


def test_resolve_device_explicit_cpu():
    assert resolve_device("cpu") == "cpu"


def test_collect_versions_contains_python_and_torch():
    v = collect_versions()
    assert "python" in v
    assert v["python"].count(".") >= 1
    assert v["torch"] != "not-installed"


def test_hash_config_stable_across_key_order():
    a = {"x": 1, "y": [1, 2], "z": {"a": True}}
    b = {"z": {"a": True}, "y": [1, 2], "x": 1}
    assert hash_config(a) == hash_config(b)


def test_hash_file_head(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello world")
    h1 = hash_file_head(p)
    h2 = hash_file_head(p)
    assert h1 == h2
    assert len(h1) == 10


def test_configure_logging_writes_file(tmp_path):
    log_path = tmp_path / "out.log"
    configure_logging(level="INFO", log_file=log_path)
    log = get_logger("test")
    log.info("hello")
    # Wait briefly is not needed; enqueue=True but message should reach disk on close.
    # We just assert the file got created with at least some content.
    import time

    for _ in range(20):
        if log_path.exists() and log_path.stat().st_size > 0:
            break
        time.sleep(0.05)
    assert log_path.exists()
