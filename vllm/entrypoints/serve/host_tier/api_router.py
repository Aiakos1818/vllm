# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only host-tier diagnostics: ``GET /host_tier_info``.

Always registered (like ``/metrics``). Returns the host-tier config plus the
parked-chain inventory (GPU keep-alive / RAM / SSD) as JSON, so external
monitors do not have to scrape ``/proc``, startup logs or a snapshot file.
Only operational fields are exposed (no environment dump, no secrets).
"""

from http import HTTPStatus

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


@router.get("/host_tier_info")
async def host_tier_info(raw_request: Request):
    client = getattr(raw_request.app.state, "engine_client", None)
    if client is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="no engine client (render-only server)",
        )
    try:
        info = await client.host_tier_info()
    except NotImplementedError as err:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="engine client does not support host_tier_info",
        ) from err
    except Exception as err:
        logger.exception("host_tier_info failed")
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail=f"host_tier_info failed: {err}",
        ) from err
    return JSONResponse(content=info)


def attach_router(app: FastAPI):
    app.include_router(router)
