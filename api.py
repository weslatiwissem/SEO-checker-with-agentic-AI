"""
FastAPI backend for the SEO Health Agent web UI.

Drop this file into your existing seo_agent project root (alongside main.py)
and run it with:

    pip install fastapi uvicorn
    uvicorn api:app --reload --port 8000

Audits take a while (multiple LLM calls, sometimes a real Lighthouse call),
so this exposes an async job pattern instead of one long blocking request:

    POST /api/audit          -> starts a job, returns {"job_id": "..."}
    GET  /api/audit/{job_id} -> {"status": "running"|"done"|"error", ...}
    GET  /api/history/{url}  -> past audit scores for a domain

The frontend polls GET /api/audit/{job_id} every couple seconds until status
is "done" or "error".
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import run_full_audit
from agent import memory

app = FastAPI(title="SEO Health Agent API")

# Allow the Next.js dev server (and any origin during local development) to
# call this API. Tighten this to your real frontend origin before deploying
# anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store. Fine for local/single-process use; if you ever run
# multiple backend workers, swap this for Redis or a database instead.
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


class AuditRequest(BaseModel):
    url: str
    mode: Literal["quick", "deep", "auto"] = "auto"
    competitor_url: str | None = None


class AuditJobResponse(BaseModel):
    job_id: str


def _run_job(job_id: str, url: str, mode: str, competitor_url: str | None) -> None:
    logs: list[str] = []

    def log_fn(message: str) -> None:
        logs.append(message)
        with _jobs_lock:
            _jobs[job_id]["logs"] = list(logs)

    try:
        report = run_full_audit(url, competitor_url=competitor_url, mode=mode, log_fn=log_fn)
        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["report"] = report
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)


@app.post("/api/audit", response_model=AuditJobResponse)
def start_audit(req: AuditRequest) -> AuditJobResponse:
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "url": req.url,
            "started_at": time.time(),
            "logs": [],
            "report": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_run_job, args=(job_id, req.url, req.mode, req.competitor_url), daemon=True
    )
    thread.start()

    return AuditJobResponse(job_id=job_id)


@app.get("/api/audit/{job_id}")
def get_audit(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return job


@app.get("/api/history/{domain}")
def get_history(domain: str, limit: int = 10) -> dict:
    rows = memory.get_history(domain, limit=limit)
    return {"domain": memory.domain_of(domain), "history": rows}


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}
