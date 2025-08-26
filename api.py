# api.py
import os
import uvicorn
import time as _time
from typing import List, Dict, Any

from fastapi import FastAPI, Body, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# 공통 로직(파이프라인, 로거) 임포트
from core import pipeline, logger

# =============================================================
# 1. FastAPI 앱 설정
# =============================================================
api = FastAPI(title="Helpdesk RAG API", version="0.1.0")

class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = _time.time()
        logger.info("api_request", extra={"extra_data": {"path": request.url.path}})
        try:
            response = await call_next(request)
            dur = round((_time.time() - start)*1000)
            logger.info("api_response", extra={"extra_data": {"status": response.status_code, "ms": dur}})
            return response
        except Exception as e:
            logger.exception("api_error", extra={"extra_data": {"error": str(e)}})
            raise

api.add_middleware(AuditMiddleware)

# =============================================================
# 2. API 엔드포인트
# =============================================================
class ChatIn(BaseModel): message: str
class ChatOut(BaseModel): reply: str; intent: str; sources: List[Dict[str, Any]]= []

@api.get("/health")
def health(): return {"ok":True}

@api.post("/chat", response_model=ChatOut)
def chat(payload: ChatIn = Body(...)):
    out = pipeline(payload.message)
    return ChatOut(reply=out.get("result",""), intent=out.get("intent",""), sources=out.get("sources", []))

# =============================================================
# 3. 엔트리포인트
# =============================================================
if __name__ == "__main__":
    import argparse
    default_host = os.getenv("API_SERVER_HOST", "0.0.0.0")
    default_port = int(os.getenv("API_PORT", 8000))

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", default=default_port, type=int)
    args = parser.parse_args()

    uvicorn.run(api, host=args.host, port=args.port)