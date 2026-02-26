import time
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from loguru import logger
from pydantic import BaseModel

from src.service.celery_app import app as celery_app
from src.core.utils import get_progress, get_result, REDIS_CLIENT, ROOT_SAVE_DIR
from src.config import settings

app = FastAPI(
    title="OCR Service (Server 2)",
    description="PDF OCR extraction service powered by Celery workers",
    version="1.0.0",
)

SET_IMG_SIZE_CONSTANT = settings.SET_IMG_SIZE_CONSTANT

SHARED_UPLOAD_DIR = Path("/tmp/ocr_uploads")
SHARED_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class ExtractResponse(BaseModel):
    """Response model for the PDF extraction submission endpoint.

    Attributes:
        task_id: The Celery task ID assigned to the OCR job.
        file_id: The client-provided unique identifier for the file.
        message: A human-readable confirmation message.
    """

    task_id: str
    file_id: str
    message: str


class ProgressResponse(BaseModel):
    """Response model for the OCR progress polling endpoint.

    Attributes:
        state: Current pipeline state (e.g., ``"PENDING"``, ``"PROCESSING"``,
            ``"COMPLETED"``, ``"FAILURE"``).
        total_pages: Total number of pages in the PDF.
        completed_pages: Number of pages processed so far.
        percent: Completion percentage (0.0–100.0).
        stage: Current pipeline stage name (e.g., ``"extraction"``,
            ``"merging"``).
        message: Human-readable status message.
        error: Error description if applicable; empty string otherwise.
    """

    state: str
    total_pages: int
    completed_pages: int
    percent: float
    stage: str
    message: str
    error: str


class ResultResponse(BaseModel):
    """Response model for the final OCR result endpoint.

    Attributes:
        status: ``True`` if results were successfully retrieved.
        file_id: The unique identifier for the processed file.
        total_pages: Number of pages in the result set.
        pages: List of per-page OCR result dicts containing extracted content.
    """

    status: bool
    file_id: str
    total_pages: int
    pages: list[dict]


def _reset_file_state(file_id: str) -> list[str]:
    """Clear stale Redis state and temporary files for a given file ID.

    Removes the ``ocr_progress`` and ``ocr_results`` Redis keys associated
    with ``file_id``, and deletes the corresponding temporary upload and
    result directories from disk if they exist.

    Args:
        file_id: Unique identifier for the file whose state should be cleared.

    Returns:
        A list of string identifiers describing what was actually cleared
        (e.g., ``["ocr_progress:<file_id>", "temp_dir"]``).
    """
    cleared = []

    for prefix in ["ocr_progress", "ocr_results"]:
        key = f"{prefix}:{file_id}"
        if REDIS_CLIENT.delete(key):
            cleared.append(key)

    temp_dir = SHARED_UPLOAD_DIR / file_id
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        cleared.append("temp_dir")

    result_dir = Path(ROOT_SAVE_DIR) / file_id
    if result_dir.exists():
        shutil.rmtree(result_dir)
        cleared.append("result_dir")

    return cleared


@app.post(
    "/ocr/extract",
    response_model=ExtractResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a PDF for OCR extraction",
)
async def extract_pdf(
    file: UploadFile = File(...),
    file_id: str = Form(...),
    batch_size: int = Form(default=4),
):
    """Accept a PDF upload and dispatch an asynchronous OCR extraction task.

    The endpoint validates the uploaded file, clears any stale state from a
    previous run with the same ``file_id``, persists the PDF to a shared
    temporary directory, and sends a Celery task to the ``process_file`` queue.

    Args:
        file: The uploaded PDF file (multipart form-data).
        file_id: Client-provided unique identifier used to track progress
            and retrieve results.
        batch_size: Number of PDF pages per OCR batch. Defaults to 4.

    Returns:
        An ``ExtractResponse`` with the Celery task ID, file ID, and a
        confirmation message.

    Raises:
        HTTPException (400): If the file is not a PDF or is empty.
        HTTPException (500): If the file cannot be saved to disk.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

    # ── Clear any stale state from a previous run with same file_id ──
    cleared = _reset_file_state(file_id)
    if cleared:
        logger.info(f"🧹 Cleared stale state for file_id={file_id}: {cleared}")

    file_temp_dir = SHARED_UPLOAD_DIR / file_id
    file_temp_dir.mkdir(parents=True, exist_ok=True)

    file_path = file_temp_dir / file.filename
    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file",
            )

        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"📁 Saved file to {file_path} ({len(content)} bytes)")

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file",
        )
    finally:
        await file.close()

    task = celery_app.send_task(
        'task.process_file',
        kwargs={
            "file_path": str(file_path),
            "filename": file.filename,
            "file_id": file_id,
            "batch_size": batch_size,
            "set_img_size_constant": SET_IMG_SIZE_CONSTANT,
        },
        queue='process_file',
        task_id=f"ocr-{file_id}-{int(time.time())}",
    )

    logger.info(f"🚀 Dispatched Celery task {task.id} for file_id={file_id}")

    return ExtractResponse(
        task_id=task.id,
        file_id=file_id,
        message="OCR task submitted successfully",
    )


@app.get(
    "/ocr/progress/{file_id}",
    response_model=ProgressResponse,
    summary="Get OCR progress for a file",
)
async def ocr_progress(file_id: str):
    """Poll the current OCR processing progress for a given file.

    Reads progress data from Redis and returns it as a structured response.
    If no progress entry exists, the returned state will be ``"PENDING"``.

    Args:
        file_id: Unique identifier for the file to check.

    Returns:
        A ``ProgressResponse`` with current state, page counts, percentage,
        stage, message, and any error information.
    """
    progress = get_progress(file_id)
    return ProgressResponse(**progress)


@app.get(
    "/ocr/result/{file_id}",
    response_model=ResultResponse,
    summary="Get final OCR result for a file",
)
async def ocr_result(file_id: str):
    """Retrieve the final combined OCR result for a processed file.

    Should only be called after the progress endpoint reports
    ``state="SUCCESS"``. If results are not yet available, the endpoint
    inspects the current progress state to return an appropriate error.

    Args:
        file_id: Unique identifier for the processed file.

    Returns:
        A ``ResultResponse`` containing the per-page OCR results.

    Raises:
        HTTPException (202): If the task is still in progress.
        HTTPException (500): If the OCR task failed.
        HTTPException (404): If the result was not found (expired or
            never completed).
    """
    pages = get_result(file_id)

    if pages is None:
        # Result not in Redis yet — check why
        progress = get_progress(file_id)
        if progress["state"] in ("PENDING", "PROCESSING", "COMBINING"):
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail=f"Task still in progress: {progress['message']}",
            )
        elif progress["state"] == "FAILURE":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"OCR failed: {progress['error']}",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Result not found. It may have expired or never completed.",
            )

    return ResultResponse(
        status=True,
        file_id=file_id,
        total_pages=len(pages),
        pages=pages,
    )


@app.post(
    "/ocr/reset/{file_id}",
    summary="Clear stale Redis state and temp files for re-processing",
)
async def reset_file(file_id: str):
    """Clear all stale state so a file can be re-processed from scratch.

    Delegates to ``_reset_file_state`` to remove Redis keys and temporary
    directories associated with the given ``file_id``.

    Args:
        file_id: Unique identifier for the file to reset.

    Returns:
        A dict with the ``file_id`` and a list of cleared resources.
    """
    cleared = _reset_file_state(file_id)
    logger.info(f"🧹 Reset file_id={file_id}: {cleared}")
    return {"file_id": file_id, "cleared": cleared}


@app.delete(
    "/ocr/cleanup/{file_id}",
    summary="Clean up temp files for a file_id",
)
async def cleanup(file_id: str):
    """Remove temporary upload and result files after processing is complete.

    Intended to be called by the upstream server (Server 1) once it has
    consumed the OCR results and no longer needs the on-disk artifacts.

    Args:
        file_id: Unique identifier for the file whose temp files should
            be removed.

    Returns:
        A dict with the ``file_id`` and a list of cleaned resource types
        (e.g., ``["temp_upload", "results"]``).
    """
    cleaned = []

    temp_dir = SHARED_UPLOAD_DIR / file_id
    if temp_dir.exists():
        logger.warning(f"Removing {temp_dir}")
        shutil.rmtree(temp_dir)
        cleaned.append("temp_upload")

    result_dir = Path(ROOT_SAVE_DIR) / file_id
    if result_dir.exists():
        logger.warning(f"Removing {result_dir}")
        shutil.rmtree(result_dir)
        cleaned.append("results")

    return {"file_id": file_id, "cleaned": cleaned}


@app.get("/health")
async def health():
    """Liveness health check endpoint.

    Returns:
        A dict with ``status`` and ``service`` name, confirming the
        OCR server is running.
    """
    return {"status": "ok", "service": "ocr-server-2"}