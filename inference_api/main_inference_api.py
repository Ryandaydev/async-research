"""Asynchronous Fantasy Acquisition Inference API"""

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4
import json
import sqlite3

import httpx
import numpy as np
import onnxruntime as rt
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response, status
from pydantic import BaseModel, HttpUrl
from starlette.concurrency import run_in_threadpool

from schemas import FantasyAcquisitionFeatures, PredictionOutput

api_description = """
This API lets clients submit fantasy football acquisition prediction jobs,
track their status, and retrieve results when processing is complete.

The endpoints are grouped into the following categories:

## Analytics
Get information about health of the API.

## Inference Jobs
Submit inference jobs, check job status, and retrieve completed results.
"""

DB_PATH = "job_tracking.db"

sess_10 = None
sess_50 = None
sess_90 = None

input_name_10 = None
label_name_10 = None
input_name_50 = None
label_name_50 = None
input_name_90 = None
label_name_90 = None


class InferenceJobRequest(BaseModel):
    features: FantasyAcquisitionFeatures
    webhook_url: Optional[HttpUrl] = None


class InferenceJobAccepted(BaseModel):
    job_id: str
    status: Literal["queued"]


class InferenceJobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    webhook_url: Optional[str] = None
    webhook_sent: bool
    webhook_status_code: Optional[int] = None
    error_text: Optional[str] = None


class InferenceJobResult(BaseModel):
    job_id: str
    status: Literal["succeeded"]
    result: PredictionOutput
    completed_at: str


# FastAPI constructor with additional details added for OpenAPI Specification
app = FastAPI(
    description=api_description,
    title="Asynchronous Fantasy Acquisition Inference API",
    version="0.1",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inference_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                request_json TEXT NOT NULL,
                result_json TEXT,
                error_text TEXT,
                webhook_url TEXT,
                webhook_sent INTEGER NOT NULL DEFAULT 0,
                webhook_status_code INTEGER,
                webhook_error_text TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            )
            """
        )
        conn.commit()


def create_job_record(
    job_id: str,
    request_json: str,
    webhook_url: Optional[str],
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO inference_jobs (
                job_id, status, request_json, webhook_url, created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (job_id, "queued", request_json, webhook_url, utc_now_iso()),
        )
        conn.commit()


def update_job_running(job_id: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE inference_jobs
            SET status = ?, started_at = ?
            WHERE job_id = ?
            """,
            ("running", utc_now_iso(), job_id),
        )
        conn.commit()


def update_job_succeeded(job_id: str, result_json: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE inference_jobs
            SET status = ?, result_json = ?, completed_at = ?, error_text = NULL
            WHERE job_id = ?
            """,
            ("succeeded", result_json, utc_now_iso(), job_id),
        )
        conn.commit()


def update_job_failed(job_id: str, error_text: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE inference_jobs
            SET status = ?, error_text = ?, completed_at = ?
            WHERE job_id = ?
            """,
            ("failed", error_text, utc_now_iso(), job_id),
        )
        conn.commit()


def update_webhook_delivery(
    job_id: str,
    sent: bool,
    status_code: Optional[int] = None,
    error_text: Optional[str] = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE inference_jobs
            SET webhook_sent = ?, webhook_status_code = ?, webhook_error_text = ?
            WHERE job_id = ?
            """,
            (1 if sent else 0, status_code, error_text, job_id),
        )
        conn.commit()


def fetch_job(job_id: str) -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                job_id,
                status,
                request_json,
                result_json,
                error_text,
                webhook_url,
                webhook_sent,
                webhook_status_code,
                webhook_error_text,
                created_at,
                started_at,
                completed_at
            FROM inference_jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        return row


@app.on_event("startup")
def load_models_and_initialize_db():
    global sess_10, sess_50, sess_90
    global input_name_10, label_name_10
    global input_name_50, label_name_50
    global input_name_90, label_name_90

    initialize_database()

    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1

    # Load the ONNX models
    sess_10 = rt.InferenceSession(
        "acquisition_model_10.onnx",
        sess_options,
        providers=["CPUExecutionProvider"],
    )
    sess_50 = rt.InferenceSession(
        "acquisition_model_50.onnx",
        sess_options,
        providers=["CPUExecutionProvider"],
    )
    sess_90 = rt.InferenceSession(
        "acquisition_model_90.onnx",
        sess_options,
        providers=["CPUExecutionProvider"],
    )

    # Get the input and output names of the models
    input_name_10 = sess_10.get_inputs()[0].name
    label_name_10 = sess_10.get_outputs()[0].name
    input_name_50 = sess_50.get_inputs()[0].name
    label_name_50 = sess_50.get_outputs()[0].name
    input_name_90 = sess_90.get_inputs()[0].name
    label_name_90 = sess_90.get_outputs()[0].name


@app.get(
    "/",
    summary="Check to see if the inference API is running",
    description="""Use this endpoint to check if the API is running.""",
    response_description="A JSON record with a message in it.",
    operation_id="v0_health_check",
    tags=["analytics"],
)
async def root():
    return {"message": "API health check successful"}


def run_prediction(features: FantasyAcquisitionFeatures) -> PredictionOutput:
    # Convert Pydantic model to NumPy array
    input_data = np.array(
        [[
            features.waiver_value_tier,
            features.fantasy_regular_season_weeks_remaining,
            features.league_budget_pct_remaining,
        ]],
        dtype=np.int64,
    )

    # Perform ONNX inference
    pred_onx_10 = sess_10.run([label_name_10], {input_name_10: input_data})[0]
    pred_onx_50 = sess_50.run([label_name_50], {input_name_50: input_data})[0]
    pred_onx_90 = sess_90.run([label_name_90], {input_name_90: input_data})[0]

    # Return prediction as a Pydantic response model
    return PredictionOutput(
        winning_bid_10th_percentile=round(
            float(pred_onx_10.reshape(-1)[0]), 2
        ),
        winning_bid_50th_percentile=round(
            float(pred_onx_50.reshape(-1)[0]), 2
        ),
        winning_bid_90th_percentile=round(
            float(pred_onx_90.reshape(-1)[0]), 2
        ),
    )


async def send_webhook(
    job_id: str,
    webhook_url: str,
    result: PredictionOutput,
) -> None:
    payload = {
        "job_id": job_id,
        "status": "succeeded",
        "result": result.model_dump(),
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(webhook_url, json=payload)
        await run_in_threadpool(
            update_webhook_delivery,
            job_id,
            response.is_success,
            response.status_code,
            None if response.is_success else response.text[:1000],
        )
    except Exception as exc:
        await run_in_threadpool(
            update_webhook_delivery,
            job_id,
            False,
            None,
            str(exc),
        )


async def process_job(
    job_id: str,
    features: FantasyAcquisitionFeatures,
    webhook_url: Optional[str],
) -> None:
    await run_in_threadpool(update_job_running, job_id)

    try:
        result = await run_in_threadpool(run_prediction, features)
        await run_in_threadpool(
            update_job_succeeded,
            job_id,
            json.dumps(result.model_dump()),
        )

        if webhook_url is not None:
            await send_webhook(job_id, webhook_url, result)

    except Exception as exc:
        await run_in_threadpool(update_job_failed, job_id, str(exc))


@app.post(
    "/inference-jobs",
    response_model=InferenceJobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an inference job",
    description="""Use this endpoint to submit a fantasy football acquisition prediction job.""",
    response_description="A JSON record containing the job ID and initial queued status.",
    operation_id="v0_submit_inference_job",
    tags=["inference jobs"],
)
async def submit_inference_job(
    request: InferenceJobRequest,
    background_tasks: BackgroundTasks,
):
    job_id = str(uuid4())

    await run_in_threadpool(
        create_job_record,
        job_id,
        request.features.model_dump_json(),
        str(request.webhook_url) if request.webhook_url is not None else None,
    )

    background_tasks.add_task(
        process_job,
        job_id,
        request.features,
        str(request.webhook_url) if request.webhook_url is not None else None,
    )

    return InferenceJobAccepted(job_id=job_id, status="queued")


@app.get(
    "/inference-jobs/{job_id}",
    response_model=InferenceJobStatus,
    summary="Get inference job status",
    description="""Use this endpoint to check the status of a submitted inference job.""",
    response_description="A JSON record containing job status and metadata.",
    operation_id="v0_get_inference_job_status",
    tags=["inference jobs"],
)
async def get_inference_job_status(job_id: str):
    row = await run_in_threadpool(fetch_job, job_id)

    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return InferenceJobStatus(
        job_id=row["job_id"],
        status=row["status"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        webhook_url=row["webhook_url"],
        webhook_sent=bool(row["webhook_sent"]),
        webhook_status_code=row["webhook_status_code"],
        error_text=row["error_text"],
    )


@app.get(
    "/inference-jobs/{job_id}/result",
    response_model=InferenceJobResult,
    summary="Get completed inference job result",
    description="""Use this endpoint to retrieve the result of a completed inference job.""",
    response_description="A JSON record containing the completed prediction result.",
    operation_id="v0_get_inference_job_result",
    tags=["inference jobs"],
)
async def get_inference_job_result(job_id: str):
    row = await run_in_threadpool(fetch_job, job_id)

    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if row["status"] == "failed":
        raise HTTPException(
            status_code=409,
            detail=f"Job failed: {row['error_text']}",
        )

    if row["status"] != "succeeded":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not complete. Current status: {row['status']}",
        )

    result_payload = json.loads(row["result_json"])

    return InferenceJobResult(
        job_id=row["job_id"],
        status="succeeded",
        result=PredictionOutput(**result_payload),
        completed_at=row["completed_at"],
    )


@app.get(
    "/inference-jobs/{job_id}/request",
    summary="Get submitted job input",
    description="""Use this endpoint to retrieve the original input payload for a submitted job.""",
    response_description="The original submitted feature payload.",
    operation_id="v0_get_inference_job_request",
    tags=["inference jobs"],
)
async def get_inference_job_request(job_id: str):
    row = await run_in_threadpool(fetch_job, job_id)

    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": row["job_id"],
        "request": json.loads(row["request_json"]),
    }


@app.delete(
    "/inference-jobs/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an inference job",
    description="""Use this endpoint to delete a stored inference job record.""",
    operation_id="v0_delete_inference_job",
    tags=["inference jobs"],
)
async def delete_inference_job(job_id: str):
    def delete_row() -> int:
        with get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM inference_jobs WHERE job_id = ?",
                (job_id,),
            )
            conn.commit()
            return cursor.rowcount

    deleted = await run_in_threadpool(delete_row)

    if deleted == 0:
        raise HTTPException(status_code=404, detail="Job not found")

    return Response(status_code=status.HTTP_204_NO_CONTENT)