from celery import Task, group, chord
from loguru import logger
from typing import Any, Union, Optional
from uuid import uuid5, NAMESPACE_DNS
from pathlib import Path
import os
import shutil
import json
import time
import redis

from src.config import settings
from src.service.celery_app import app
from src.core.ocr_engine import OCREngine
from src.core.utils import (
    create_batches,
    _init_progress,
    _increment_pages,
    _set_stage,
    _save_result,
    ROOT_SAVE_DIR
)

# ──────────────────────────────────────────────
# Singletons (initialised once per worker process)
# ──────────────────────────────────────────────
logger.info("Initializing OCR...")
OCR_ENGINE = OCREngine(
    device=settings.OCR_DEVICE,
    precision=settings.OCR_PRECISION,
    text_detection_model_name=settings.TEXT_DETECTION_MODEL_NAME,
    text_recognition_model_name=settings.TEXT_RECOGNITION_MODEL_NAME,
    use_doc_orientation_classify=settings.USER_DOC_ORIENTATION_CLASSIFY,
    use_doc_unwarping=settings.USER_DOC_UNWARPING,
    use_textline_orientation=settings.USER_TEXTLINE_ORIENTATION,
    post_processing_config=eval(settings.POST_PROCESSING_CONFIG),
)
logger.info("✅ OCR Engine initialized.")

# ──────────────────────────────────────────────
# 1. Entry point: process_file
# ──────────────────────────────────────────────
@app.task(bind=True, name="task.process_file")
def process_file(
    self: Task,
    file_path: Union[str, Path],
    filename: str,
    file_id: str,
    batch_size: int,
    set_img_size_constant: bool = False,
) -> dict[str, Any]:
    """
    Entry point for OCR pipeline.
    Creates batches → dispatches parallel OCR via chord → combines results.
    
    The file_id is used as the progress tracking key so Server 1
    can poll a single Redis key regardless of how many subtasks run.
    """
    logger.info(f"📁 Starting extraction for file: {filename} (file_id={file_id})")

    parent_path = str(Path(file_path).parent)

    try:
        # Create batches of page image paths
        batches = create_batches(
            file_path=str(file_path),
            temp_dir=parent_path,
            batch_size=batch_size,
            set_img_size_constant=set_img_size_constant
        )

        # Calculate total pages across all batches
        total_pages = sum(len(indices) for _, indices in batches)
        
        logger.info(f"🗂️ Created {len(batches)} batches, {total_pages} total pages")

        # Initialize progress in Redis
        _init_progress(file_id, total_pages)

        # Build chord: parallel OCR → combine
        group_tasks = group(
            ocr_file.s(
                images=images,
                page_indices=indices,
                file_id=file_id,  # Pass file_id so each subtask can report progress
            )
            for images, indices in batches
        )

        aggregate_task = chord(
            group_tasks,
            combine_results.s(
                file_id=file_id,
                save_dir=os.path.join(ROOT_SAVE_DIR, file_id),
                temp_dir=parent_path
            )
        )

    except Exception as e:
        logger.error(f"❌ Failed to start extraction for {filename}: {e}")
        _set_stage(file_id, "FAILURE", "extraction", f"Failed to start: {str(e)}", str(e))
        raise

    return self.replace(aggregate_task)


# ──────────────────────────────────────────────
# 2. OCR a single batch (runs in parallel)
# ──────────────────────────────────────────────
@app.task(bind=True, name="task.ocr_file")
def ocr_file(
    self: Task,
    images: list[str],
    page_indices: list[int],
    file_id: str,
) -> list[dict[str, Any]]:
    """
    OCR a batch of page images.
    After completing, atomically increments the shared progress counter in Redis.
    """
    try:
        ocr_result = OCR_ENGINE.run_ocr(
            images=images,
            page_indices=page_indices,
        )

        # Report batch completion to Redis (atomic increment)
        progress = _increment_pages(file_id, len(page_indices))
        logger.info(
            f"✅ Batch done: pages {page_indices} | "
            f"Progress: {progress['completed']}/{progress['total']}"
        )

        return ocr_result

    except Exception as e:
        logger.error(f"❌ OCR failed for pages {page_indices}: {e}")
        _set_stage(file_id, "FAILURE", "extraction", f"OCR failed on pages {page_indices}: {str(e)}", str(e))
        raise


# ──────────────────────────────────────────────
# 3. Combine results (chord callback)
# ──────────────────────────────────────────────
@app.task(bind=True, name="task.combine_results")
def combine_results(
    self: Task,
    ocr_results: list[list[dict[str, Any]]],
    file_id: str,
    temp_dir: str,
    save_dir: Optional[str] = None,
) -> dict[str, Any]:
    """
    Combine OCR results from all batches, sort by page index, optionally save.
    Updates progress to reflect the combining/saving stage.
    """
    _set_stage(file_id, "PROCESSING", "combining", "Combining OCR results...")

    try:
        combined_texts = []
        for result in ocr_results:
            combined_texts.extend(result)

        sorted_combined_texts = sorted(combined_texts, key=lambda x: x["page_index"])

        if save_dir:
            _set_stage(file_id, "PROCESSING", "saving", "Saving results to disk...")
            
            os.makedirs(save_dir, exist_ok=True)
            save_path = Path(save_dir) / "combined_ocr_result.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(sorted_combined_texts, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Combined OCR result saved to {save_path}")

        # Mark as complete
        _set_stage(file_id, "SUCCESS", "done", "OCR extraction completed!")

        # Save results
        logger.info("Saving to redis for temporary")
        _save_result(file_id, sorted_combined_texts)

        return {
            "status": True,
            "filename": "",  # process_file passes this via the chain
            "pages": sorted_combined_texts,
            "total_pages": len(sorted_combined_texts),
            "file_id": file_id,
        }

    except Exception as e:
        logger.error(f"❌ Failed to combine results: {e}")
        _set_stage(file_id, "FAILURE", "combining", f"Failed to combine: {str(e)}", str(e))
        raise

    finally:
        if temp_dir and os.path.exists(temp_dir):
            logger.info(f"Removing temporary dir: {temp_dir}")
            shutil.rmtree(Path(temp_dir).parent, ignore_errors=True)