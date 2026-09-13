# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the always-on host-tier diagnostics endpoint."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.host_tier.api_router import attach_router

pytestmark = pytest.mark.cpu_test


class _FakeEngine:
    def __init__(self, info=None, error=None) -> None:
        self._info = info
        self._error = error

    async def host_tier_info(self):
        if self._error is not None:
            raise self._error
        return self._info


def _client(engine) -> TestClient:
    app = FastAPI()
    app.state.engine_client = engine
    attach_router(app)
    return TestClient(app)


def test_host_tier_info_ok() -> None:
    info = {
        "config": {"pid": 1, "block_size": 1600},
        "sessions": {"gpu": [], "ram": [], "ssd": []},
    }
    resp = _client(_FakeEngine(info)).get("/host_tier_info")
    assert resp.status_code == 200
    assert resp.json() == info


def test_host_tier_info_no_engine() -> None:
    app = FastAPI()
    app.state.engine_client = None
    attach_router(app)
    assert TestClient(app).get("/host_tier_info").status_code == 503


def test_host_tier_info_unsupported() -> None:
    engine = _FakeEngine(error=NotImplementedError())
    assert _client(engine).get("/host_tier_info").status_code == 503


def test_host_tier_info_engine_failure() -> None:
    engine = _FakeEngine(error=RuntimeError("boom"))
    assert _client(engine).get("/host_tier_info").status_code == 503
