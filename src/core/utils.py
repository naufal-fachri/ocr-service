from typing import Union, Any
from pathlib import Path
from loguru import logger
from PIL import Image
import pymupdf
import redis
import os

from  src.config import settings

# ──────────────────────────────────────────────
# Redis client for progress tracking
# ──────────────────────────────────────────────

ROOT_SAVE_DIR = "/home/naufal/ocr_service/ocr_results"
os.makedirs(ROOT_SAVE_DIR, exist_ok=True)

REDIS_CLIENT = redis.Redis(
    host="localhost",
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
    target_width: int | None,
    target_height: int | None,
) -> Image.Image :

    assert target_width or target_height, "Target width or heigh must be filled"

    if target_width and target_height:
        w, h = img.size
        rasio = w/h

        if target_width > target_height:

            if rasio >= 1:
                logger.info("Image is kept horizontaly")
                return img.resize((target_width, target_height))

            else:
                logger.info("Image is changed to horizontal")
                return img.resize((target_width, target_height))

        else:
            if rasio <= 1:
                logger.info("Imag is kept vertically")
                return img.resize((target_width, target_height))
            
            else:
                return img.resize((target_width, target_height))

    elif target_height:
        w, h = img.size

        aspect_ratio = (h/w) * target_height
        new_target_width = target_width * aspect_ratio

        return img.resize((new_target_width, target_height))

    else:
        w, h = img.size

        aspect_ratio = (w/h) * target_width
        new_target_height = target_height * aspect_ratio
        
        return img.resize((target_width, new_target_height))


def create_batches(
    file_path: str,
    temp_dir: Union[str, Path],
    batch_size: int = MAX_BATCH_SIZE,
    set_img_size_constant: bool = False
) -> list[tuple[list[str], list[int]]]:


    assert file_path is not None, "file_path must be provided"
    assert temp_dir is not None, "temp_dir must be provided"

    temp_dir = Path(temp_dir)

    try:
        with pymupdf.open(filename=file_path, filetype="pdf") as doc:
            total_pages = doc.page_count
            batch_files = []
            for start_page in range(0, total_pages, batch_size):
                end_page = min(start_page + batch_size, total_pages)
                current_batch_file_path = []
                current_batch_file_indices = []

                for page_num in range(start_page, end_page):
                    pil_img = doc.load_page(page_num).get_pixmap(dpi=DPI).pil_image()

                    if set_img_size_constant:
                        pil_img = resize_image(
                            img=pil_img,
                            target_height=TARGET_HEIGHT,
                            target_width=TARGET_WIDTH
                        )

                    logger.info(f"Saving page {page_num} of {file_path} to temporary directory")

                    target_path = temp_dir / f"page_{page_num}.png"
                    pil_img.save(target_path)
                    current_batch_file_path.append(str(target_path))
                    current_batch_file_indices.append(page_num)

                batch_files.append((current_batch_file_path, current_batch_file_indices))
        
        return batch_files

    except Exception as e:
        logger.error(f"Error while creating batches from {file_path}: {e}")
        return []

def _init_progress(file_id: str, total_pages: int):
    """Initialize progress tracking for a file."""
    key = f"ocr_progress:{file_id}"
    REDIS_CLIENT.hset(key, mapping={
        "state": "PROCESSING",
        "total_pages": str(total_pages),
        "completed_pages": "0",
        "stage": "extraction",
        "message": f"Starting OCR on {total_pages} pages...",
        "error": "",
    })
    REDIS_CLIENT.expire(key, PROGRESS_TTL)


def _increment_pages(file_id: str, pages_done: int) -> dict:
    """
    Atomically increment completed page count and return current progress.
    Called by each ocr_file subtask after finishing its batch.
    """
    key = f"ocr_progress:{file_id}"
    # HINCRBY is atomic — safe for concurrent subtasks
    new_completed = REDIS_CLIENT.hincrby(key, "completed_pages", pages_done)
    total = int(REDIS_CLIENT.hget(key, "total_pages") or 1)
    
    message = f"OCR processed {new_completed}/{total} pages..."
    REDIS_CLIENT.hset(key, mapping={
        "message": message,
    })
    
    return {"completed": new_completed, "total": total}


def _set_stage(file_id: str, state: str, stage: str, message: str, error: str = ""):
    """Update the current stage/state of the pipeline."""
    key = f"ocr_progress:{file_id}"
    REDIS_CLIENT.hset(key, mapping={
        "state": state,
        "stage": stage,
        "message": message,
        "error": error,
    })


def get_progress(file_id: str) -> dict[str, Any]:
    """
    Read progress from Redis. Called by Server 1 (FastAPI) to poll status.
    
    Returns:
        {
            "state": "PROCESSING",
            "total_pages": 20,
            "completed_pages": 12,
            "percent": 60.0,
            "stage": "extraction",
            "message": "OCR processed 12/20 pages...",
            "error": ""
        }
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
