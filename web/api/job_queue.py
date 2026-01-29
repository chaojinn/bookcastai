'''
create a in memory job queue, length limited to 1000 jobs;
each job have 
{
  id: uuid
  command: string
  params: array of key value pair
  status: queued/running/cancelled/success/fail
  started_at: timestamp
  finished_at: timestamp
  task: asyncio.Task
  progress: int 0-100
  progress_msg: string
}
there should be a loop to run through the queue and run each job one after another
at any given time, there should be only one job running
when the queue is full, remove 500 oldest jobs

implement api
GET /api/job/{id} query job info, return job json data + a queue_pos field in json showing the position of the job in the queue. return 404 if job not found
POST /apo/job with job json data as follow
{
  command: string
  params: array of key value pair
}
it should create a new job and return the new created job id

DELETE /api/job/{id}
cancels the job, set status to cancelled and finish time to now and call cancel on the task

'''

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from supertokens_python.recipe.session import SessionContainer
from supertokens_python.recipe.session.framework.fastapi import verify_session

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_JOBS = 1000
PRUNE_COUNT = 500

JobHandler = Callable[["Job"], Awaitable[None] | None]


class JobPayload(BaseModel):
    command: str = Field(..., min_length=1)
    params: List[Dict[str, Any]] = Field(default_factory=list)


@dataclass
class Job:
    id: str
    command: str
    params: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "queued"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    task: Optional[asyncio.Task] = None
    progress: int = 0
    progress_msg: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "command": self.command,
            "params": self.params,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": self.progress,
            "progress_msg": self.progress_msg,
        }


_jobs: Dict[str, Job] = {}
_job_queue: Deque[str] = deque()
_job_order: Deque[str] = deque()
_job_handlers: Dict[str, JobHandler] = {}
_queue_lock: Optional[asyncio.Lock] = None
_queue_event: Optional[asyncio.Event] = None
_worker_task: Optional[asyncio.Task] = None


def register_job_handler(command: str, handler: JobHandler) -> None:
    _job_handlers[command] = handler


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_params(raw_params: List[Dict[str, Any]]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for entry in raw_params or []:
        if not isinstance(entry, dict):
            continue
        if "key" in entry and "value" in entry:
            key = entry.get("key")
            if isinstance(key, str) and key:
                params[key] = entry.get("value")
            continue
        if len(entry) == 1:
            key, value = next(iter(entry.items()))
            if isinstance(key, str) and key:
                params[key] = value
            continue
        for key, value in entry.items():
            if isinstance(key, str) and key and key not in params:
                params[key] = value
    return params


def _normalize_ignore_classes(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned or None
    if isinstance(value, str):
        cleaned = [item.strip() for item in value.split(",") if item.strip()]
        return cleaned or None
    return None


def _sanitize_pod_title(raw: Any) -> str:
    if not raw:
        return ""
    name = os.path.basename(str(raw)).strip()
    name = name.replace(" ", "")
    name = name.replace("/", "").replace("\\", "")
    return name


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _resolve_tts_provider(model_name: Any) -> Optional[str]:
    if not isinstance(model_name, str):
        return None
    normalized = model_name.strip().lower()
    if normalized in {"kokoro-tts", "kokoro"}:
        return "kokoro"
    if normalized in {"openai", "openai-tts"}:
        return "openai"
    if normalized in {"dia", "dia-tts"}:
        return "dia"
    return None


def _ensure_async_state() -> None:
    global _queue_lock, _queue_event
    if _queue_lock is None:
        _queue_lock = asyncio.Lock()
    if _queue_event is None:
        _queue_event = asyncio.Event()


def _queue_position(job_id: str) -> int:
    try:
        return list(_job_queue).index(job_id)
    except ValueError:
        return -1


def _set_progress(job: Job, progress: int, message: str | None = None) -> None:
    job.progress = max(0, min(100, int(progress)))
    if message is not None:
        job.progress_msg = message


async def _ensure_worker() -> None:
    global _worker_task
    _ensure_async_state()
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_queue_worker())


async def _queue_worker() -> None:
    if _queue_event is None or _queue_lock is None:
        return
    while True:
        await _queue_event.wait()
        while True:
            async with _queue_lock:
                if not _job_queue:
                    _queue_event.clear()
                    break
                job_id = _job_queue.popleft()
                job = _jobs.get(job_id)
                if job is None or job.status != "queued":
                    continue
                job.status = "running"
                job.started_at = _utc_now()
            job.task = asyncio.create_task(_run_job(job))
            try:
                await job.task
            except asyncio.CancelledError:
                if job.status != "cancelled":
                    job.status = "cancelled"
                    job.finished_at = _utc_now()
                    _set_progress(job, 100, "Cancelled.")
            except Exception:
                logger.exception("Job worker failed for %s", job.id)
                job.status = "fail"
                job.finished_at = _utc_now()
                _set_progress(job, 100, "Job failed.")


async def _run_job(job: Job) -> None:
    handler = _job_handlers.get(job.command)
    if handler is None:
        job.status = "fail"
        job.finished_at = _utc_now()
        _set_progress(job, 100, "Unknown command.")
        return

    try:
        result = handler(job)
        if asyncio.iscoroutine(result):
            await result
        if job.status == "running":
            job.status = "success"
            job.finished_at = _utc_now()
            _set_progress(job, 100, "Completed.")
    except asyncio.CancelledError:
        job.status = "cancelled"
        job.finished_at = _utc_now()
        _set_progress(job, 100, "Cancelled.")
        raise
    except Exception:
        logger.exception("Job %s failed", job.id)
        job.status = "fail"
        job.finished_at = _utc_now()
        _set_progress(job, 100, "Job failed.")


async def _handle_parse_epub(job: Job) -> None:
    params = _normalize_params(job.params)
    book_title = params.get("book_title")
    if not isinstance(book_title, str) or not book_title.strip():
        job.status = "fail"
        job.finished_at = _utc_now()
        _set_progress(job, 100, "Missing book_title parameter.")
        return

    chunk_size_raw = params.get("chunk_size")
    chunk_size = None
    if chunk_size_raw is not None:
        try:
            chunk_size = max(1, int(chunk_size_raw))
        except (TypeError, ValueError):
            job.status = "fail"
            job.finished_at = _utc_now()
            _set_progress(job, 100, "Invalid chunk_size parameter.")
            return

    ignore_classes = _normalize_ignore_classes(params.get("ignore_classes"))
    ai_extract_text = _to_bool(params.get("ai_extract_text"))

    def publish_progress(progress: int, message: str | None = None) -> None:
        _set_progress(job, progress, message or "")

    try:
        from agent.epub_agent import run_epub_agent
    except ImportError as exc:
        job.status = "fail"
        job.finished_at = _utc_now()
        _set_progress(job, 100, f"EPUB parser unavailable: {exc}")
        return

    try:
        kwargs: Dict[str, Any] = {"ai_extract_text": ai_extract_text}
        if chunk_size is not None:
            kwargs["chunk_size"] = chunk_size
        if ignore_classes is not None:
            kwargs["ignore_classes"] = ignore_classes
        await asyncio.to_thread(
            run_epub_agent,
            book_title,
            publish_progress=publish_progress,
            **kwargs,
        )
    except Exception as exc:
        logger.exception("parse_epub job failed for %s", job.id)
        job.status = "fail"
        job.finished_at = _utc_now()
        _set_progress(job, 100, str(exc) or "EPUB parsing failed.")
        return

    job.status = "success"
    job.finished_at = _utc_now()
    _set_progress(job, 100, "Completed.")


async def _handle_tts(job: Job) -> None:
    params = _normalize_params(job.params)
    book_title_raw = params.get("book_title")
    book_title = _sanitize_pod_title(book_title_raw)
    if not book_title:
        job.status = "fail"
        job.finished_at = _utc_now()
        _set_progress(job, 100, "Missing book_title parameter.")
        return

    model_name = params.get("model_name") or "kokoro-tts"
    provider_name = _resolve_tts_provider(model_name)
    if provider_name is None:
        job.status = "fail"
        job.finished_at = _utc_now()
        _set_progress(job, 100, "Unsupported model_name parameter.")
        return

    voice_value = params.get("voice")
    voice = str(voice_value).strip() if isinstance(voice_value, str) and voice_value.strip() else "af_jessica"

    speed_raw = params.get("speed", 1.0)
    try:
        speed = float(speed_raw)
    except (TypeError, ValueError):
        job.status = "fail"
        job.finished_at = _utc_now()
        _set_progress(job, 100, "Invalid speed parameter.")
        return

    overwrite = _to_bool(params.get("overwrite"))

    base_dir = os.getenv("PODS_BASE")
    if not base_dir:
        job.status = "fail"
        job.finished_at = _utc_now()
        _set_progress(job, 100, "PODS_BASE is not configured.")
        return

    base_path = Path(base_dir).expanduser() / book_title
    json_path = base_path / "book.json"
    if not json_path.exists():
        job.status = "fail"
        job.finished_at = _utc_now()
        _set_progress(job, 100, "Parsed EPUB JSON not found.")
        return

    try:
        book_text = json_path.read_text(encoding="utf-8")
        book_data = json.loads(book_text)
    except OSError as exc:
        job.status = "fail"
        job.finished_at = _utc_now()
        _set_progress(job, 100, f"Failed to read EPUB JSON: {exc}")
        return
    except json.JSONDecodeError as exc:
        job.status = "fail"
        job.finished_at = _utc_now()
        _set_progress(job, 100, f"Invalid EPUB JSON: {exc}")
        return

    def publish_progress(progress: int, message: str | None = None) -> None:
        _set_progress(job, progress, message or "")

    try:
        from epub_to_pod import convert_epub_to_pod
    except ImportError as exc:
        job.status = "fail"
        job.finished_at = _utc_now()
        _set_progress(job, 100, f"TTS engine unavailable: {exc}")
        return

    _set_progress(job, 0, "Starting TTS...")
    try:
        await asyncio.to_thread(
            convert_epub_to_pod,
            book_data=book_data,
            output_dir=base_path,
            voice=voice,
            lang="en-us",
            speed=speed,
            overwrite=overwrite,
            provider_name=provider_name,
            publish_progress=publish_progress,
        )
    except Exception as exc:
        logger.exception("tts job failed for %s", job.id)
        job.status = "fail"
        job.finished_at = _utc_now()
        _set_progress(job, 100, str(exc) or "TTS failed.")
        return

    job.status = "success"
    job.finished_at = _utc_now()
    _set_progress(job, 100, "Completed.")


async def _prune_jobs_locked() -> None:
    if len(_job_order) <= MAX_JOBS:
        return
    remove_limit = min(PRUNE_COUNT, len(_job_order))
    remove_ids: List[str] = []
    for job_id in _job_order:
        job = _jobs.get(job_id)
        if job is None or job.status != "running":
            remove_ids.append(job_id)
        if len(remove_ids) >= remove_limit:
            break
    if not remove_ids:
        return
    remove_set = set(remove_ids)
    original_order = list(_job_order)
    original_queue = list(_job_queue)
    _job_order.clear()
    _job_order.extend([job_id for job_id in original_order if job_id not in remove_set])
    _job_queue.clear()
    _job_queue.extend([job_id for job_id in original_queue if job_id not in remove_set])
    for job_id in remove_ids:
        job = _jobs.pop(job_id, None)
        if job is None:
            continue
        if job.task and not job.task.done():
            job.task.cancel()
        if job.status in {"queued", "running"}:
            job.status = "cancelled"
            job.finished_at = _utc_now()
            _set_progress(job, 100, "Cancelled.")


@router.get("/api/job/{job_id}")
async def get_job(
    job_id: str,
    session: SessionContainer = Depends(verify_session()),
) -> Dict[str, Any]:
    _ensure_async_state()
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    data = job.to_dict()
    data["queue_pos"] = _queue_position(job_id)
    return data


@router.post("/api/job")
async def create_job(
    payload: JobPayload,
    session: SessionContainer = Depends(verify_session()),
) -> Dict[str, str]:
    _ensure_async_state()
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        command=payload.command,
        params=payload.params,
        status="queued",
        progress=0,
        progress_msg="Queued.",
    )
    if _queue_lock is None:
        raise HTTPException(status_code=500, detail="Job queue is unavailable.")
    async with _queue_lock:
        _jobs[job_id] = job
        _job_queue.append(job_id)
        _job_order.append(job_id)
        await _prune_jobs_locked()
        if _queue_event is not None:
            _queue_event.set()
    await _ensure_worker()
    return {"id": job_id}


@router.delete("/api/job/{job_id}")
async def cancel_job(
    job_id: str,
    session: SessionContainer = Depends(verify_session()),
) -> Dict[str, str]:
    _ensure_async_state()
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if _queue_lock is None:
        raise HTTPException(status_code=500, detail="Job queue is unavailable.")
    async with _queue_lock:
        if job.status not in {"success", "fail", "cancelled"}:
            job.status = "cancelled"
            job.finished_at = _utc_now()
            _set_progress(job, 100, "Cancelled.")
        try:
            _job_queue.remove(job_id)
        except ValueError:
            pass
    if job.task and not job.task.done():
        job.task.cancel()
    return {"id": job_id, "status": job.status}


register_job_handler("parse_epub", _handle_parse_epub)
register_job_handler("tts", _handle_tts)
