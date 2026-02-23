import os
import shutil
from pathlib import Path
from tempfile import mkdtemp
from uuid import uuid4
import json

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel


from src.service.celery_app import app as celery_app
from src.core.utils import get_progress, get_result, ROOT_SAVE_DIR
from src.config import settings

app = FastAPI(
    title="OCR Service (Server 2)",
    description="PDF OCR extraction service powered by Celery workers",
    version="1.0.0",
)

SET_IMG_SIZE_CONSTANT = settings.SET_IMG_SIZE_CONSTANT

SHARED_UPLOAD_DIR = Path("/tmp/ocr_uploads")
SHARED_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────
class ExtractResponse(BaseModel):
    task_id: str
    file_id: str
    message: str


class ProgressResponse(BaseModel):
    state: str
    total_pages: int
    completed_pages: int
    percent: float
    stage: str
    message: str
    error: str


class ResultResponse(BaseModel):
    status: bool
    file_id: str
    total_pages: int
    pages: list[dict]


# ──────────────────────────────────────────────
# POST /ocr/extract
# ──────────────────────────────────────────────
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
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

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
            "set_img_size_constant": SET_IMG_SIZE_CONSTANT
        },
        queue='process_file',
        task_id=f"ocr-{file_id}"
    )

    logger.info(f"🚀 Dispatched Celery task {task.id} for file_id={file_id}")

    return ExtractResponse(
        task_id=task.id,
        file_id=file_id,
        message="OCR task submitted successfully",
    )


# ──────────────────────────────────────────────
# GET /ocr/progress/{file_id}
# ──────────────────────────────────────────────
@app.get(
    "/ocr/progress/{file_id}",
    response_model=ProgressResponse,
    summary="Get OCR progress for a file",
)
async def ocr_progress(file_id: str):
    progress = get_progress(file_id)
    return ProgressResponse(**progress)


# ──────────────────────────────────────────────
# GET /ocr/result/{file_id}
# ──────────────────────────────────────────────
@app.get(
    "/ocr/result/{file_id}",
    response_model=ResultResponse,
    summary="Get final OCR result for a file",
)
async def ocr_result(file_id: str):
    """
    Returns the final combined OCR result from Redis.
    Only call this after progress shows state=SUCCESS.
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


# ──────────────────────────────────────────────
# DELETE /ocr/cleanup/{file_id}
# ──────────────────────────────────────────────
@app.delete(
    "/ocr/cleanup/{file_id}",
    summary="Clean up temp files for a file_id",
)
async def cleanup(file_id: str):
    """Remove temp upload and result files after Server 1 is done."""
    cleaned = []

    temp_dir = SHARED_UPLOAD_DIR / file_id
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        cleaned.append("temp_upload")

    result_dir = Path(ROOT_SAVE_DIR) / file_id
    if result_dir.exists():
        shutil.rmtree(result_dir)
        cleaned.append("results")

    return {"file_id": file_id, "cleaned": cleaned}


# ──────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "ocr-server-2"}