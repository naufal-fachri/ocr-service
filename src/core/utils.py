from typing import Union, Any
from pathlib import Path
from loguru import logger
from PIL import Image
import pymupdf
import redis
import os
import json

from src.config import settings

ROOT_SAVE_DIR = "/app/ocr_results"
os.makedirs(ROOT_SAVE_DIR, exist_ok=True)

REDIS_CLIENT = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD,
    db=0,
    decode_responses=True,
)

PROGRESS_TTL = 3600  # 1 hour
MAX_BATCH_SIZE = 16
DPI = 200
TARGET_HEIGHT = 2048
TARGET_WIDTH = 1536


def resize_image(
    img: Image.Image,
    target_width: int | None = None,
    target_height: int | None = None,
) -> Image.Image:
    """Resize a PIL Image to the specified target dimensions.

    Supports three modes of operation:
        - Both target_width and target_height provided: resizes to exact dimensions
          (ignores aspect ratio).
        - Only target_height provided: scales width proportionally to preserve
          the original aspect ratio.
        - Only target_width provided: scales height proportionally to preserve
          the original aspect ratio.

    Args:
        img: The source PIL Image to resize.
        target_width: Desired width in pixels. If None, width is calculated
            from target_height to preserve aspect ratio.
        target_height: Desired height in pixels. If None, height is calculated
            from target_width to preserve aspect ratio.

    Returns:
        A new PIL Image resized to the computed dimensions.

    Raises:
        AssertionError: If neither target_width nor target_height is provided.
    """
    assert target_width or target_height, "Target width or height must be filled"

    if target_width and target_height:
        # Exact resize — no aspect ratio preservation
        return img.resize((target_width, target_height))

    w, h = img.size

    if target_height:
        # Scale width proportionally based on target_height
        new_target_width = int(w * (target_height / h))
        return img.resize((new_target_width, target_height))

    else:
        # Scale height proportionally based on target_width
        new_target_height = int(h * (target_width / w))
        return img.resize((target_width, new_target_height))


def create_batches(
    file_path: str,
    temp_dir: Union[str, Path],
    batch_size: int = MAX_BATCH_SIZE,
    set_img_size_constant: bool = False,
) -> list[tuple[list[str], list[int]]]:
    """Convert a PDF file into batches of page images saved to disk.

    Opens the PDF at ``file_path``, renders each page as a PNG image at the
    configured DPI, optionally resizes it to a constant size, and groups the
    resulting file paths into fixed-size batches for downstream OCR processing.

    Args:
        file_path: Absolute path to the source PDF file.
        temp_dir: Directory where rendered page images will be saved.
            Created automatically if it does not exist.
        batch_size: Maximum number of pages per batch. Defaults to
            ``MAX_BATCH_SIZE`` (16).
        set_img_size_constant: If True, every page image is resized to
            ``TARGET_WIDTH`` x ``TARGET_HEIGHT`` before saving.

    Returns:
        A list of tuples, each containing:
            - A list of file path strings for the page images in that batch.
            - A list of corresponding zero-based page indices.
        Returns an empty list if a fatal error occurs during processing.

    Raises:
        AssertionError: If ``file_path`` or ``temp_dir`` is None.
    """
    assert file_path is not None, "file_path must be provided"
    assert temp_dir is not None, "temp_dir must be provided"

    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        with pymupdf.open(filename=file_path, filetype="pdf") as doc:
            total_pages = doc.page_count
            logger.info(f"📄 Opened PDF: {file_path} | total_pages={total_pages}")

            batch_files = []

            for start_page in range(0, total_pages, batch_size):
                end_page = min(start_page + batch_size, total_pages)
                current_batch_file_path = []
                current_batch_file_indices = []

                for page_num in range(start_page, end_page):
                    try:
                        # ── Render page to pixmap ──
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap(dpi=DPI)

                        if pix.width == 0 or pix.height == 0:
                            logger.warning(f"⚠️  Page {page_num}: empty pixmap (w={pix.width} h={pix.height}), skipping")
                            continue

                        # ── Convert to PIL Image ──
                        pil_img = pix.pil_image()
                        if pil_img is None:
                            logger.warning(f"⚠️  Page {page_num}: pil_image() returned None, skipping")
                            continue

                        # ── Optional resize ──
                        if set_img_size_constant:
                            pil_img = resize_image(
                                img=pil_img,
                                target_height=TARGET_HEIGHT,
                                target_width=TARGET_WIDTH,
                            )

                        # ── Save to disk ──
                        target_path = temp_dir / f"page_{page_num}.png"
                        logger.info(
                            f"💾 Saving page {page_num} → {target_path.name} "
                            f"({pil_img.width}x{pil_img.height}px)"
                        )
                        pil_img.save(target_path)

                        # ── Verify save succeeded ──
                        if not target_path.exists() or target_path.stat().st_size == 0:
                            logger.error(f"❌ Page {page_num}: save silently failed — file missing or empty: {target_path}")
                            continue

                        current_batch_file_path.append(str(target_path))
                        current_batch_file_indices.append(page_num)

                    except Exception as page_err:
                        logger.exception(f"❌ Page {page_num}: failed to render/save — {page_err}")
                        continue

                if current_batch_file_path:
                    batch_files.append((current_batch_file_path, current_batch_file_indices))
                else:
                    logger.warning(f"⚠️  Batch [{start_page}–{end_page-1}]: all pages failed, skipping batch")

        logger.info(f"✅ create_batches done | {len(batch_files)} batch(es) | pages saved: {sum(len(i) for _, i in batch_files)}/{total_pages}")
        return batch_files

    except Exception as e:
        logger.exception(f"❌ Fatal error creating batches from {file_path}: {e}")
        return []


def _init_progress(file_id: str, total_pages: int):
    """Initialize OCR progress tracking for a file in Redis.

    Creates a Redis hash with initial state ``PROCESSING`` and sets a TTL
    of ``PROGRESS_TTL`` seconds. Should be called once at the start of an
    OCR pipeline run before any batch processing begins.

    Args:
        file_id: Unique identifier for the file being processed.
        total_pages: Total number of pages to be OCR-processed.
    """
    key = f"ocr_progress:{file_id}"
    REDIS_CLIENT.hset(
        key,
        mapping={
            "state": "PROCESSING",
            "total_pages": str(total_pages),
            "completed_pages": "0",
            "stage": "extraction",
            "message": f"Starting OCR on {total_pages} pages...",
            "error": "",
        },
    )
    REDIS_CLIENT.expire(key, PROGRESS_TTL)


def _increment_pages(file_id: str, pages_done: int) -> dict:
    """Atomically increment the completed page count and return current progress.

    Uses Redis ``HINCRBY`` for atomic updates, making it safe to call from
    concurrent batch-processing subtasks.

    Args:
        file_id: Unique identifier for the file being processed.
        pages_done: Number of pages completed in this batch.

    Returns:
        A dict with ``completed`` (new total of completed pages) and
        ``total`` (total pages in the job).
    """
    key = f"ocr_progress:{file_id}"
    new_completed = REDIS_CLIENT.hincrby(key, "completed_pages", pages_done)
    total = int(REDIS_CLIENT.hget(key, "total_pages") or 1)

    message = f"OCR processed {new_completed}/{total} pages..."
    REDIS_CLIENT.hset(key, mapping={
        "message": message,
    })

    return {"completed": new_completed, "total": total}


def _set_stage(file_id: str, state: str, stage: str, message: str, error: str = ""):
    """Update the current stage and state of the OCR pipeline in Redis.

    Used to signal stage transitions (e.g., extraction → merging → done)
    or to record error states.

    Args:
        file_id: Unique identifier for the file being processed.
        state: Pipeline state (e.g., ``"PROCESSING"``, ``"COMPLETED"``,
            ``"FAILED"``).
        stage: Current pipeline stage name (e.g., ``"extraction"``,
            ``"merging"``).
        message: Human-readable status message.
        error: Error description if applicable; empty string otherwise.
    """
    key = f"ocr_progress:{file_id}"
    REDIS_CLIENT.hset(key, mapping={
        "state": state,
        "stage": stage,
        "message": message,
        "error": error,
    })


def _save_result(
    file_id: str,
    results: list[dict[str, Any]],
):
    """Persist OCR results to Redis with a short TTL.

    Serializes the result list as JSON and stores it under a key derived
    from ``file_id``. The key expires after 300 seconds (5 minutes),
    giving the caller enough time to retrieve the results.

    Args:
        file_id: Unique identifier for the processed file.
        results: List of per-page OCR result dicts to store.
    """
    key = f"ocr_results:{file_id}"
    REDIS_CLIENT.set(
        key,
        json.dumps(results),
        ex=300,
    )


def get_progress(file_id: str) -> dict[str, Any]:
    """Read OCR progress from Redis for a given file.

    Called by the FastAPI server to poll the current status of an OCR job.
    If no progress data exists yet, returns a default ``PENDING`` response.

    Args:
        file_id: Unique identifier for the file being processed.

    Returns:
        A dict containing: ``state``, ``total_pages``, ``completed_pages``,
        ``percent`` (0.0–100.0), ``stage``, ``message``, and ``error``.
    """
    key = f"ocr_progress:{file_id}"
    data = REDIS_CLIENT.hgetall(key)
    if not data:
        return {
            "state": "PENDING",
            "total_pages": 0,
            "completed_pages": 0,
            "percent": 0.0,
            "stage": "queued",
            "message": "Waiting in queue...",
            "error": "",
        }

    total = int(data.get("total_pages", 1))
    completed = int(data.get("completed_pages", 0))
    percent = round((completed / total) * 100, 1) if total > 0 else 0.0

    return {
        "state": data.get("state", "PENDING"),
        "total_pages": total,
        "completed_pages": completed,
        "percent": percent,
        "stage": data.get("stage", ""),
        "message": data.get("message", ""),
        "error": data.get("error", ""),
    }


def get_result(file_id: str) -> list[dict[str, Any]] | None:
    """Retrieve stored OCR results from Redis.

    Looks up the JSON-serialized result list saved by ``_save_result``.
    Returns ``None`` if the key has expired or was never set.

    Args:
        file_id: Unique identifier for the processed file.

    Returns:
        The deserialized list of per-page OCR result dicts, or None if
        no results are available.
    """
    key = f"ocr_results:{file_id}"
    data = REDIS_CLIENT.get(key)
    if not data:
        return None
    return json.loads(data)