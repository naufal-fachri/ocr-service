# 📄 OCR Service

A high-performance PDF OCR extraction service built with FastAPI, Celery, and PaddleOCR. Designed for asynchronous, GPU-accelerated document processing with a clean REST API.

---

## 🏗️ Architecture

```
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│   FastAPI (API) │──────▶│  Celery Worker  │──────▶│  PaddleOCR/GPU  │
│   :8001         │        │  (Thread Pool)  │        │                 │
└────────┬────────┘        └────────┬────────┘        └─────────────────┘
         │                          │
         ▼                          ▼
   ┌──────────┐              ┌──────────────┐
   │  Redis   │              │  ocr_results/│
   │(Progress)│              │  (File Store)│
   └──────────┘              └──────────────┘
```

The API server receives PDF uploads and dispatches tasks to the Celery workers via Redis. The OCR worker processes pages in parallel batches using PaddleOCR (GPU-accelerated), while a separate coordinator worker handles file ingestion and result combining. Results are stored on disk and progress is tracked in Redis. Temp files are shared between containers via a shared volume.

---

## 🚀 Getting Started

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU + CUDA drivers (for GPU-accelerated OCR)
- NVIDIA Container Toolkit

---

## 🔧 Required Infrastructure Setup

Before running the OCR service, make sure **RabbitMQ** and **Redis** are up and running.

### RabbitMQ

RabbitMQ is used as the Celery message broker.

```bash
docker run --detach --name rabbitmq \
  --env RABBITMQ_DEFAULT_USER=user \
  --env RABBITMQ_DEFAULT_PASS=password \
  --env RABBITMQ_DEFAULT_VHOST=your_vhost_name \
  --publish 15672:15672 \
  --publish 5672:5672 \
  rabbitmq:4.2-management-alpine
```

| Port | Description |
|------|-------------|
| `5672` | AMQP protocol (used by Celery) |
| `15672` | Management UI — `http://localhost:15672` |

> 📌 Replace `your_vhost_name` with your desired virtual host name, then update `RABBITMQ_DEFAULT_VHOST` in your `.env` accordingly.
>
> For more details, see the official docs: https://hub.docker.com/_/rabbitmq

---

### Redis

Redis is used as the Celery result backend and for storing OCR progress.

```bash
docker run -d --name redis -p 6379:6379 redis:latest
```

**Optional: Using a custom `redis.conf`**

If you need custom Redis configuration (e.g. setting maxmemory, persistence, auth), mount a local config file:

```bash
docker run -d --name redis \
  -p 6379:6379 \
  -v /your/local/redis/conf:/usr/local/etc/redis \
  redis redis-server /usr/local/etc/redis/redis.conf
```

Alternatively, build a custom image:

```dockerfile
FROM redis
COPY redis.conf /usr/local/etc/redis/redis.conf
CMD ["redis-server", "/usr/local/etc/redis/redis.conf"]
```

> ⚠️ The mounted config directory must be **writable**, as Redis may need to create or rewrite config files depending on its mode of operation.

---

### Verify Both Services Are Running

```bash
docker ps | grep -E "rabbitmq|redis"
```

You should see both containers with status `Up` before proceeding.

---

### 1. Clone & Configure

```bash
git clone <repo-url>
cd ocr_service

cp .env.example .env
# Edit .env with your configuration
```

### 2. Create Required Directories

```bash
mkdir -p ocr_results
```

### 3. Start Services

```bash
docker compose up --build -d
```

The API will be available at `http://localhost:8001`.

---

## ⚙️ Configuration

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

### `.env.example`

```env
# --- Redis Configuration ---
REDIS_HOST=host.docker.internal
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# --- RabbitMQ Configuration ---
RABBITMQ_HOST=host.docker.internal
RABBITMQ_USERNAME=your_rabbitmq_username
RABBITMQ_PASSWORD=your_rabbitmq_password
RABBITMQ_VHOST=your_vhost_name

# --- OCR Configuration ---
USER_DOC_ORIENTATION_CLASSIFY=True
USER_DOC_UNWARPING=True
USER_TEXTLINE_ORIENTATION=True
OCR_DEVICE=gpu                              # gpu or cpu
OCR_PRECISION=fp32                          # fp32 or fp16
TEXT_DETECTION_MODEL_NAME=PP-OCRv5_mobile_det
TEXT_RECOGNITION_MODEL_NAME=PP-OCRv5_mobile_rec
POST_PROCESSING_CONFIG='{"y_threshold": 10, "column_threshold": 0.3}'
SET_IMG_SIZE_CONSTANT=False
```

### Variable Reference

| Variable | Description |
|----------|-------------|
| `REDIS_HOST` | Redis host (`host.docker.internal` to reach host from container) |
| `REDIS_PORT` | Redis port |
| `REDIS_PASSWORD` | Redis auth password (leave empty if none) |
| `RABBITMQ_HOST` | RabbitMQ host |
| `RABBITMQ_USERNAME` | RabbitMQ username |
| `RABBITMQ_PASSWORD` | RabbitMQ password |
| `RABBITMQ_VHOST` | RabbitMQ virtual host name |
| `USER_DOC_ORIENTATION_CLASSIFY` | Enable document orientation classification |
| `USER_DOC_UNWARPING` | Enable document unwarping correction |
| `USER_TEXTLINE_ORIENTATION` | Enable text line orientation detection |
| `OCR_DEVICE` | Inference device — `gpu` or `cpu` |
| `OCR_PRECISION` | Model precision — `fp32` or `fp16` |
| `TEXT_DETECTION_MODEL_NAME` | PaddleOCR text detection model |
| `TEXT_RECOGNITION_MODEL_NAME` | PaddleOCR text recognition model |
| `POST_PROCESSING_CONFIG` | JSON config for post-processing (y_threshold, column_threshold) |
| `SET_IMG_SIZE_CONSTANT` | Fix image resize to a constant size (`False` to disable) |

---

## 📡 API Reference

### `POST /ocr/extract`
Submit a PDF for OCR extraction.

**Form Data:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | ✅ | PDF file to process |
| `file_id` | string | ✅ | Unique identifier for this file |
| `batch_size` | int | ❌ | Pages per batch (default: `4`) |

**Response `202 Accepted`:**
```json
{
  "task_id": "ocr-abc123",
  "file_id": "abc123",
  "message": "OCR task submitted successfully"
}
```

---

### `GET /ocr/progress/{file_id}`
Get real-time OCR progress for a submitted file.

**Response `200 OK`:**
```json
{
  "state": "PROCESSING",
  "total_pages": 20,
  "completed_pages": 8,
  "percent": 40.0,
  "stage": "ocr",
  "message": "Processing page 8 of 20",
  "error": ""
}
```

**Possible states:** `PENDING` → `PROCESSING` → `COMBINING` → `SUCCESS` / `FAILURE`

---

### `GET /ocr/result/{file_id}`
Get the final OCR result. Only call this after progress shows `state: SUCCESS`.

**Response `200 OK`:**
```json
{
  "status": true,
  "file_id": "abc123",
  "total_pages": 20,
  "pages": [
    { "page": 1, "text": "Extracted text content..." },
    ...
  ]
}
```

**Error responses:**
- `202` — Task still in progress
- `404` — Result not found or expired
- `500` — OCR processing failed

---

### `DELETE /ocr/cleanup/{file_id}`
Remove temp upload files and OCR result files after you're done consuming the results.

**Response `200 OK`:**
```json
{
  "file_id": "abc123",
  "cleaned": ["temp_upload", "results"]
}
```

---

### `GET /health`
Service health check.

```json
{ "status": "ok", "service": "ocr-server-2" }
```

---

## 🔄 Typical Workflow

```
1. POST /ocr/extract      → get task_id & file_id
2. GET  /ocr/progress/:id → poll until state == "SUCCESS"
3. GET  /ocr/result/:id   → retrieve extracted pages
4. DEL  /ocr/cleanup/:id  → clean up temp files
```

---

## 🐳 Docker Compose Services

| Service | Container | Port | Description |
|---------|-----------|------|-------------|
| `api` | `ocr_service` | `8001:8000` | FastAPI HTTP server |
| `worker` | `ocr_worker` | — | Celery OCR worker (GPU-enabled, concurrency: 2) |
| `worker_coordinator` | `ocr_coordinator` | — | Celery coordinator worker (CPU, concurrency: 4) |

The worker architecture is split into two specialized workers:

- **`worker`** — Handles the `ocr_file` queue. Runs with GPU access and `--concurrency=2` for parallel OCR batch processing via PaddleOCR.
- **`worker_coordinator`** — Handles the `process_file` and `combine_results` queues. Runs CPU-only with `--concurrency=4` for file ingestion, page splitting, and result merging.

This separation ensures GPU resources are dedicated to OCR inference while coordination tasks run independently without competing for GPU memory.

**Shared Volumes:**
- `${PWD}/ocr_results` — Persistent OCR output files (bind mount)
- `shared_tmp` — Temporary PDF uploads shared between API and workers

**Celery Queues:**
- `process_file` — Initial file ingestion and splitting (handled by `worker_coordinator`)
- `ocr_file` — Per-batch OCR processing (handled by `worker`)
- `combine_results` — Merging batch results (handled by `worker_coordinator`)

---

## 📁 Project Structure

```
ocr_service/
├── src/
│   ├── app/
│   │   └── api.py              # FastAPI application & endpoints
│   ├── core/
│   │   ├── ocr_engine.py       # PaddleOCR processing logic
│   │   └── utils.py            # Progress/result helpers, ROOT_SAVE_DIR
│   └── service/
│       ├── celery_app.py       # Celery app configuration
│       └── celery_task.py      # Task definitions
├── config.py                   # App settings (Pydantic)
├── ocr_results/                # OCR output directory (bind mount)
├── docker-compose.yaml
├── Dockerfile
├── pyproject.toml
└── .env
```

---

## 🛠️ Development

### Running Locally (without Docker)

```bash
# Install dependencies
uv sync

# Start the API
uv run uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --reload

# Start the OCR worker (in another terminal)
uv run celery -A src.service.celery_app worker \
  --loglevel=info \
  --hostname=worker_ocr@%h \
  --concurrency=2 \
  --pool=threads \
  --queues=ocr_file

# Start the coordinator worker (in another terminal)
uv run celery -A src.service.celery_app worker \
  --loglevel=info \
  --hostname=worker_coord@%h \
  --concurrency=4 \
  --pool=threads \
  --queues=process_file,combine_results
```

### Viewing Logs

```bash
# API logs
docker logs ocr_service -f

# OCR worker logs
docker logs ocr_worker -f

# Coordinator worker logs
docker logs ocr_coordinator -f
```

---

## 📝 Notes

- Results are stored in Redis with a TTL — retrieve them promptly after completion.
- The OCR worker uses `--concurrency=2` with thread pool to balance GPU utilization and memory usage.
- The coordinator worker uses `--concurrency=4` for lightweight file processing and result combining tasks.
- Batch size (`batch_size`) can be tuned based on GPU VRAM — larger batches are faster but use more memory.
- Always call `DELETE /ocr/cleanup/{file_id}` after consuming results to free disk space.