from celery import Task, group, chord
from loguru import logger
from typing import Any, Union, Optional
from pathlib import Path
import os
import json
import time

from src.config import settings
from src.service.celery_app import app
from src.core.utils import (
    create_batches,
    _init_progress,
    _increment_pages,
    _set_stage,
    _save_result,
    ROOT_SAVE_DIR
)

_OCR_ENGINE = None


def _get_ocr_engine():
    """Lazily initialize OCR engine on first use (only on GPU workers)."""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from src.core.ocr_engine import OCREngine

        logger.info("Initializing OCR Engine...")
        _OCR_ENGINE = OCREngine(
            device=settings.OCR_DEVICE,
            precision=settings.OCR_PRECISION,
            text_detection_model_name=settings.TEXT_DETECTION_MODEL_NAME,
            text_recognition_model_name=settings.TEXT_RECOGNITION_MODEL_NAME,
            use_doc_orientation_classify=settings.USER_DOC_ORIENTATION_CLASSIFY,
            use_doc_unwarping=settings.USER_DOC_UNWARPING,
            use_textline_orientation=settings.USER_TEXTLINE_ORIENTATION,
            post_processing_config=settings.POST_PROCESSING_CONFIG,
        )
        logger.info("✅ OCR Engine initialized.")
    return _OCR_ENGINE

def _ensure_dir(path: Union[str, Path], label: str = "") -> Path:
    """Ensure directory exists in this container's view of the shared volume."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    logger.debug(f"📂 Dir ensured{f' ({label})' if label else ''}: {p}")
    return p

def _wait_for_file(file_path: Union[str, Path], retries: int = 5, delay: float = 0.2) -> bool:
    """
    Poll until file exists on the shared volume.
    Needed because API container writes the file and dispatches the Celery task
    almost simultaneously — the worker may start before the volume flush completes.
    """
    p = Path(file_path)
    for attempt in range(1, retries + 1):
        if p.exists() and p.stat().st_size > 0:
            logger.debug(f"✅ File ready after {attempt} attempt(s): {p.name} ({p.stat().st_size:,} bytes)")
            return True
        logger.warning(f"⏳ File not ready (attempt {attempt}/{retries}): {p} — retrying in {delay}s")
        time.sleep(delay)
    logger.error(f"❌ File never appeared after {retries} attempts: {p}")
    return False


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
    t_start = time.perf_counter()
    file_path = Path(file_path)
    parent_path = file_path.parent

    logger.info(f"{'='*60}")
    logger.info(f"📁 process_file START | file={filename} | file_id={file_id}")
    logger.info(f"   file_path : {file_path}")
    logger.info(f"   parent_dir: {parent_path}")

    # ── Ensure dir exists from worker's perspective (volume sync safety) ──
    _ensure_dir(parent_path, label="temp upload dir")

    # ── Wait for file to be fully flushed to shared volume ──
    if not _wait_for_file(file_path):
        err = f"File not found on shared volume after retries: {file_path}"
        logger.error(f"❌ {err}")
        _set_stage(file_id, "FAILURE", "extraction", err, err)
        raise FileNotFoundError(err)

    logger.info(f"   file_size : {file_path.stat().st_size:,} bytes")

    try:
        # ── Create page image batches ──
        logger.info(f"🔪 Creating batches (batch_size={batch_size}, set_img_size_constant={set_img_size_constant})")
        batches = create_batches(
            file_path=str(file_path),
            temp_dir=str(parent_path),
            batch_size=batch_size,
            set_img_size_constant=set_img_size_constant,
        )

        total_pages = sum(len(indices) for _, indices in batches)
        logger.info(f"🗂️  Created {len(batches)} batch(es), {total_pages} total page(s) | elapsed={time.perf_counter()-t_start:.2f}s")

        if total_pages == 0:
            err = "PDF produced 0 pages — file may be corrupt, encrypted, or unsupported."
            logger.error(f"❌ {err}")
            _set_stage(file_id, "FAILURE", "extraction", err, err)
            # Return empty chord so combine_results still runs and saves empty result
            aggregate_task = chord(
                group(),
                combine_results.s(
                    file_id=file_id,
                    save_dir=os.path.join(ROOT_SAVE_DIR, file_id),
                    temp_dir=str(parent_path),
                )
            )
            return self.replace(aggregate_task)

        # ── Initialize Redis progress ──
        _init_progress(file_id, total_pages)
        logger.info(f"📊 Progress initialized in Redis for file_id={file_id}")

        # ── Build chord: parallel OCR → combine ──
        group_tasks = group(
            ocr_file.s(
                images=images,
                page_indices=indices,
                file_id=file_id,
            )
            for images, indices in batches
        )

        aggregate_task = chord(
            group_tasks,
            combine_results.s(
                file_id=file_id,
                save_dir=os.path.join(ROOT_SAVE_DIR, file_id),
                temp_dir=str(parent_path),
            )
        )

        logger.info(f"🚀 Dispatching chord with {len(batches)} OCR task(s)")

    except Exception as e:
        logger.exception(f"❌ Failed to start extraction for {filename}: {e}")
        _set_stage(file_id, "FAILURE", "extraction", f"Failed to start: {str(e)}", str(e))
        raise

    return self.replace(aggregate_task)

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
    t_start = time.perf_counter()
    logger.info(f"🔍 ocr_file START | file_id={file_id} | pages={page_indices} | n_images={len(images)}")

    # Verify all images exist before handing to OCR engine
    missing = [img for img in images if not Path(img).exists()]
    if missing:
        err = f"Missing image files: {missing}"
        logger.error(f"❌ {err}")
        _set_stage(file_id, "FAILURE", "extraction", err, err)
        raise FileNotFoundError(err)

    try:
        ocr_engine = _get_ocr_engine()
        ocr_result = ocr_engine.run_ocr(
            images=images,
            page_indices=page_indices,
        )

        progress = _increment_pages(file_id, len(page_indices))
        elapsed = time.perf_counter() - t_start
        logger.info(
            f"✅ ocr_file DONE | pages={page_indices} | "
            f"progress={progress['completed']}/{progress['total']} | "
            f"elapsed={elapsed:.2f}s"
        )
        return ocr_result

    except Exception as e:
        logger.exception(f"❌ OCR failed for pages {page_indices}: {e}")
        _set_stage(file_id, "FAILURE", "extraction", f"OCR failed on pages {page_indices}: {str(e)}", str(e))
        raise

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
    """
    t_start = time.perf_counter()
    logger.info(f"{'='*60}")
    logger.info(f"🔗 combine_results START | file_id={file_id} | n_batches={len(ocr_results)}")
    _set_stage(file_id, "PROCESSING", "combining", "Combining OCR results...")

    try:
        # ── Flatten and sort ──
        combined_texts: list[dict] = []
        for batch_result in ocr_results:
            combined_texts.extend(batch_result)
        sorted_texts = sorted(combined_texts, key=lambda x: x["page_index"])
        logger.info(f"📄 Combined {len(sorted_texts)} page(s) from {len(ocr_results)} batch(es)")

        # ── Persist to disk ──
        if save_dir:
            _set_stage(file_id, "PROCESSING", "saving", "Saving results to disk...")
            os.makedirs(save_dir, exist_ok=True)
            save_path = Path(save_dir) / "combined_ocr_result.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(sorted_texts, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Saved to disk: {save_path} ({save_path.stat().st_size:,} bytes)")

        # ── Mark complete & push to Redis ──
        _set_stage(file_id, "SUCCESS", "done", "OCR extraction completed!")
        _save_result(file_id, sorted_texts)
        logger.info(f"📦 Result saved to Redis | file_id={file_id}")

        elapsed = time.perf_counter() - t_start
        logger.info(f"✅ combine_results DONE | {len(sorted_texts)} page(s) | elapsed={elapsed:.2f}s")
        logger.info(f"{'='*60}")

        return {
            "status": True,
            "filename": "",
            "pages": sorted_texts,
            "total_pages": len(sorted_texts),
            "file_id": file_id,
        }

    except Exception as e:
        logger.exception(f"❌ Failed to combine results for file_id={file_id}: {e}")
        _set_stage(file_id, "FAILURE", "combining", f"Failed to combine: {str(e)}", str(e))
        raise

    finally:
        logger.info(f"🧹 Temp dir cleanup skipped — handled by file-parser service: {temp_dir}")