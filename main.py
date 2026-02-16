import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid5, NAMESPACE_DNS
from typing import AsyncGenerator
from io import BytesIO
from loguru import logger
from fastapi import APIRouter, UploadFile, HTTPException, Depends, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.models.exceptions import FileValidationError
from src.core.validator import FileValidator
from src.services.extract import FileExtractionService
from src.services.docs import upload_file_to_minio
from src.core.config import settings


router = APIRouter()

class ExtractionRequest(BaseModel):
    user_id: str
    chat_id: str

async def async_upload_to_minio(file_content: bytes, filename: str, content_type: str, user_id: str, chat_id: str) -> str:
    """Async wrapper for MinIO upload."""
    loop = asyncio.get_event_loop()
    
    # Create a simple object that mimics UploadFile for sync function
    class TempUploadFileForMinio:
        def __init__(self, content, filename, content_type, size):
            self.file = BytesIO(content)
            self.filename = filename
            self.content_type = content_type
            self.size = size
    
    temp_upload_file = TempUploadFileForMinio(
        content=file_content,
        filename=filename,
        content_type=content_type,
        size=len(file_content)
    )
    logger.info("Uploading to Minio")
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(
            executor, 
            upload_file_to_minio,
            temp_upload_file, 
            user_id, 
            chat_id
        )

def get_extraction_service() -> FileExtractionService:
    """Dependency to get the extraction service."""
    if not _extraction_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return _extraction_service


def set_extraction_service(service: FileExtractionService) -> None:
    """Set the extraction service instance."""
    global _extraction_service
    _extraction_service = service


@router.post(
    "/docs/extract",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-Sent Events (SSE) stream with real-time progress updates",
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "description": "Stream of SSE events in JSON format",
                        "example": 'data: {"status": "started", "message": "Reading file..."}\\n\\n',
                    }
                }
            },
            "headers": {
                "Cache-Control": {
                    "description": "Disables caching for SSE stream",
                    "schema": {"type": "string", "example": "no-cache"},
                },
                "Connection": {
                    "description": "Keeps connection alive for streaming",
                    "schema": {"type": "string", "example": "keep-alive"},
                },
                "X-Accel-Buffering": {
                    "description": "Disables nginx buffering",
                    "schema": {"type": "string", "example": "no"},
                },
            },
        },
        503: {
            "description": "Service Unavailable - Extraction service not initialized",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "detail": {
                                "type": "string",
                                "example": "Service not initialized",
                            }
                        },
                    },
                    "example": {"detail": "Service not initialized"},
                }
            },
        },
    },
)
async def extract_file(
    file: UploadFile,
    user_id: str = Form(...),
    chat_id: str = Form(...),
    extraction_service: FileExtractionService = Depends(get_extraction_service),
):
    """Extract with progress updates via SSE."""

    async def progress_generator() -> AsyncGenerator[str, None]:
        content = None
        file_content_for_minio = None
        filename = file.filename or "unknown"
        file_id = str(uuid5(NAMESPACE_DNS, f"{filename}_{user_id}_{chat_id}"))
        extraction_task = None

        try:
            # Send initial status
            yield f"data: {json.dumps({'status': 'started', 'message': 'Reading file...'})}\n\n"

            # Read file content with timeout
            try:
                content = await asyncio.wait_for(file.read(), timeout=60.0)
                file_content_for_minio = content

            except asyncio.TimeoutError:
                raise FileValidationError("File upload timeout")

            # Validate file (checks extension, size, empty)
            FileValidator.validate_file(file, content)

            # Check if file is an image
            is_image = FileValidator.is_image(file.filename)

            if is_image:
                # IMAGE PROCESSING: Direct upload to MinIO without extraction
                yield f"data: {json.dumps({'status': 'processing', 'message': 'Uploading image to storage...'})}\n\n"
                
                try:
                    file_url = await async_upload_to_minio(
                        file_content=file_content_for_minio,
                        filename=file.filename,
                        content_type=file.content_type or "image/jpeg",
                        user_id=user_id,
                        chat_id=chat_id
                    )
                    
                    logger.info(f"Image uploaded to MinIO: {file_url}")
                    
                    # Send completion for image
                    final_result = {
                        "status": "completed",
                        "message": "Image uploaded successfully!",
                        "file_metadata": {
                            "filename": filename,
                            "file_id": file_id,
                            "file_url": file_url,
                            "file_type": "image"
                        },
                        "success": True,
                        "error": None,
                    }
                    
                    yield f"data: {json.dumps(final_result)}\n\n"
                    
                except Exception as minio_error:
                    logger.error(f"Failed to upload image to MinIO: {minio_error}")
                    error_result = {
                        "status": "failed",
                        "message": "Failed to upload image",
                        "file_metadata": {
                            "filename": filename,
                            "file_id": file_id,
                            "file_type": "image"
                        },
                        "success": False,
                        "error": str(minio_error),
                    }
                    yield f"data: {json.dumps(error_result)}\n\n"
                
            else:
                # DOCUMENT PROCESSING: Extract, chunk, upsert, then upload
                yield f"data: {json.dumps({'status': 'processing', 'message': 'Extracting content...'})}\n\n"

                # Add overall timeout and make task cancellable
                try:
                    async with asyncio.timeout(600):  # 10 minute timeout
                        async with extraction_service.semaphore:
                            if file.filename.lower().endswith(".pdf"): <- request to server 2 using aiohttp
                                extraction_result = await asyncio.to_thread(
                                    extraction_service.extract_pdf,
                                    content,
                                    file.filename,
                                    20,
                                    True
                                )
                            else:  # .docx
                                extraction_result = await asyncio.to_thread(
                                    extraction_service.extract_word,
                                    content,
                                    file.filename
                                )
                                
                except asyncio.TimeoutError:
                    raise FileValidationError(
                        "Extraction timeout - file too large or complex"
                    )

                # Clear content from memory ASAP
                del content
                content = None

                if extraction_result.get("status", False):
                    logger.info(f"Extraction successful for file: {file.filename}")

                    yield f"data: {json.dumps({'status': 'processing', 'message': 'Chunking document...'})}\n\n"

                    chunked_documents, ids = await extraction_service.chunk_file(
                        parsed_file_result=extraction_result,
                        user_id=user_id,
                        chat_id=chat_id,
                        chunker=extraction_service.chunker,
                    )

                    # Clear extraction result from memory
                    del extraction_result

                    yield f"data: {json.dumps({'status': 'processing', 'message': f'Upserting {len(chunked_documents)} chunks...'})}\n\n"

                    upsert_status = await extraction_service.upsert_chunks_to_vector_store(
                        documents=chunked_documents,
                        ids=ids,
                        batch_size=settings.VECTOR_STORE_BATCH_SIZE,
                        vector_store=extraction_service.vector_store,
                    )

                    # Clear chunks from memory
                    del chunked_documents, ids

                    # Upload file to MinIO after successful processing
                    file_url = None
                    if upsert_status:
                        try:
                            yield f"data: {json.dumps({'status': 'processing', 'message': 'Uploading file to storage...'})}\n\n"
                            
                            file_url = await async_upload_to_minio(
                                file_content=file_content_for_minio,
                                filename=file.filename,
                                content_type=file.content_type,
                                user_id=user_id,
                                chat_id=chat_id
                            )
                            
                            logger.info(f"Document uploaded to MinIO: {file_url}")
                            
                        except Exception as minio_error:
                            logger.error(f"Failed to upload document to MinIO: {minio_error}")
                            # Don't fail the entire process if MinIO upload fails
                            
                        finally:
                            # Clear MinIO content from memory
                            if file_content_for_minio:
                                del file_content_for_minio
                                file_content_for_minio = None

                    # Send completion
                    final_result = {
                        "status": "completed" if upsert_status else "failed",
                        "message": "Processing completed!" if upsert_status else "Processing failed",
                        "file_metadata": {
                            "filename": filename,
                            "file_id": file_id,
                            "file_url": file_url if upsert_status else None,
                            "file_type": "document"
                        },
                        "success": upsert_status,
                        "error": None,
                    }

                    yield f"data: {json.dumps(final_result)}\n\n"

                else:
                    error_result = {
                        "status": "failed",
                        "message": "Extraction failed or no content found",
                        "file_metadata": {
                            "filename": filename,
                            "file_id": file_id,
                            "file_type": "document"
                        },
                        "success": False,
                        "error": extraction_result.get(
                            "error", "Unknown error during extraction"
                        ),
                    }
                    yield f"data: {json.dumps(error_result)}\n\n"

        except asyncio.CancelledError:
            # Handle client disconnect
            logger.warning(f"Request cancelled for file: {filename}")
            if extraction_task and not extraction_task.done():
                extraction_task.cancel()
            raise

        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}", exc_info=True)
            error_result = {
                "status": "error",
                "message": str(e),
                "file_metadata": {
                    "filename": filename,
                    "file_id": file_id
                },
                "success": False,
                "error": str(e),
            }
            yield f"data: {json.dumps(error_result)}\n\n"

        finally:
            # Always cleanup resources
            if content:
                del content
            if file_content_for_minio:
                del file_content_for_minio
            try:
                await file.close()
            except Exception as e:
                logger.warning(f"Error closing file: {e}")

    return StreamingResponse(
        progress_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )