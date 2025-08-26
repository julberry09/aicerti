import os
import json
import logging
from typing import TypedDict, List, Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Body, Request
from pydantic import BaseModel
import uvicorn
import httpx
import time as _time

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from starlette.middleware.base import BaseHTTPMiddleware

# =============================================================
# 1. 공통 설정 / 환경 변수
# =============================================================
load_dotenv()

logger = logging.getLogger("helpdesk-bot")


# 구조화 로그(JSON)
LOG_DIR = Path("./logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)
class _ConsoleFormatter(logging.Formatter):
    def format(self, record):
        base = {"level": record.levelname, "name": record.name, "msg": record.getMessage()}
        if hasattr(record, "extra_data"):
            base.update(record.extra_data)
        return json.dumps(base, ensure_ascii=False)

console_handler = logging.StreamHandler()
console_handler.setFormatter(_ConsoleFormatter())
file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
file_handler.setFormatter(_ConsoleFormatter())
logger.handlers = []
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Azure OpenAI 환경변수
AOAI_ENDPOINT    = os.getenv("AOAI_ENDPOINT", "")
AOAI_API_KEY     = os.getenv("AOAI_API_KEY", "")
AOAI_API_VERSION = os.getenv("AOAI_API_VERSION", "2024-10-21")
AOAI_DEPLOY_GPT4O_MINI = os.getenv("AOAI_DEPLOY_GPT4O_MINI", "gpt-4o-mini")
AOAI_DEPLOY_GPT4O = os.getenv("AOAI_DEPLOY_GPT4O", "gpt-4o")
AOAI_DEPLOY_EMBED_3_SMALL = os.getenv("AOAI_DEPLOY_EMBED_3_SMALL", "text-embedding-3-small")

# Webhook 알림
NOTIFY_WEBHOOK_URL = os.getenv("NOTIFY_WEBHOOK_URL", "")
NOTIFY_ENABLED = bool(NOTIFY_WEBHOOK_URL)

# 경로
KB_DIR = Path("./kb")
INDEX_DIR = Path("./index")
INDEX_NAME = "faiss_index"

# 샘플 데이터
OWNER_FALLBACK = {
    "인사시스템-사용자관리": {"owner": "홍길동", "email": "owner.hr@example.com", "phone": "010-1234-5678"},
    "재무시스템-정산화면": {"owner": "김재무", "email": "owner.fa@example.com", "phone": "010-2222-3333"},
    "포털-공지작성": {"owner": "박운영", "email": "owner.ops@example.com", "phone": "010-9999-0000"},
}
EMPLOYEE_DIR = {
    "kim.s": {"name": "김선니", "dept": "IT운영", "phone": "010-1111-2222", "status": "active"},
    "lee.a": {"name": "이알파", "dept": "보안", "phone": "010-3333-4444", "status": "active"},
}

# =============================================================
# 2. 유틸
# =============================================================
def _is_running_in_streamlit() -> bool:
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

def notify(event: str, data: dict | None = None):
    payload = {"event": event, "data": data or {}}
    try:
        logger.info("notify", extra={"extra_data": {"event": event, "data": data or {}}})
        if NOTIFY_ENABLED:
            import requests
            requests.post(NOTIFY_WEBHOOK_URL, json=payload, timeout=3)
    except Exception as e:
        logger.warning(f"노티 실패: {e}")

# =============================================================
# 3. RAG 유틸리티
# =============================================================
def _make_embedder() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_deployment=AOAI_DEPLOY_EMBED_3_SMALL,
        api_key=AOAI_API_KEY,
        azure_endpoint=AOAI_ENDPOINT,
        api_version=AOAI_API_VERSION,
    )

def _load_docs_from_kb() -> List[Document]:
    docs: List[Document] = []
    if not KB_DIR.exists():
        KB_DIR.mkdir(parents=True, exist_ok=True)
    for p in KB_DIR.rglob("*"):
        if p.is_file():
            try:
                suf = p.suffix.lower()
                if suf == ".pdf":
                    docs.extend(PyPDFLoader(str(p)).load())
                elif suf == ".csv":
                    docs.extend(CSVLoader(file_path=str(p), encoding="utf-8").load())
                elif suf in [".txt", ".md"]:
                    docs.extend(TextLoader(str(p), encoding="utf-8").load())
                elif suf == ".docx":
                    docs.extend(Docx2txtLoader(str(p)).load())
            except Exception as e:
                logger.warning(f"문서 로드 실패: {p} - {e}")
    return docs

def build_or_load_vectorstore() -> FAISS:
    embed = _make_embedder()
    if (INDEX_DIR / f"{INDEX_NAME}.faiss").exists():
        return FAISS.load_local(str(INDEX_DIR / INDEX_NAME), embeddings=embed, allow_dangerous_deserialization=True)

    raw_docs = _load_docs_from_kb()
    if not raw_docs:
        seed_text = """사내 헬프데스크 안내
- ID 발급: 신규 입사자는 HR 포털에서 '계정 신청' 양식을 제출. 승인 후 IT가 계정 생성.
- 비밀번호 초기화: SSO 포털의 '비밀번호 재설정' 기능 사용. 본인인증 필요.
- 담당자 조회: 포털 상단 검색창에 화면/메뉴명을 입력하면 담당자 카드가 표시됨.
- 근무시간: 평일 09:00~18:00, 점심 12:00~13:00
- 긴급 연락: it-help@example.com / 02-123-4567
"""
        raw_docs = [Document(page_content=seed_text, metadata={"source": "seed-faq.txt"})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(raw_docs)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs = FAISS.from_documents(chunks, embed)
    vs.save_local(str(INDEX_DIR / INDEX_NAME))
    return vs

_vectorstore: Optional[FAISS] = None
def retriever(k: int = 4):
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_or_load_vectorstore()
    return _vectorstore.as_retriever(search_kwargs={"k": k})

def make_llm(model: str = AOAI_DEPLOY_GPT4O_MINI, temperature: float = 0.2) -> AzureChatOpenAI:
    if not (AOAI_ENDPOINT and AOAI_API_KEY):
        raise RuntimeError("AOAI_ENDPOINT/AOAI_API_KEY 미설정. .env 또는 환경변수 설정 필요.")
    return AzureChatOpenAI(
        azure_deployment=model,
        api_version=AOAI_API_VERSION,
        azure_endpoint=AOAI_ENDPOINT,
        api_key=AOAI_API_KEY,
        temperature=temperature,
    )

# =============================================================
# 4. LangGraph (도구 + 노드)
# =============================================================
class BotState(TypedDict):
    question: str
    intent: str
    result: str
    sources: List[Dict[str, Any]]
    tool_output: Dict[str, Any]

# ---- 도구 ----
def tool_reset_password(payload: Dict[str, Any]) -> Dict[str, Any]:
    user = payload.get("user") or ""
    found = EMPLOYEE_DIR.get(user)
    if not found:
        return {
            "ok": False,
            "message": "사번/계정이 확인되지 않습니다. 포털의 '비밀번호 찾기'에서 사번/사내메일로 시도해주세요.",
            "steps": ["SSO 포털 접속 > 비밀번호 재설정", "본인인증(휴대폰/이메일)", "새 비밀번호 설정"],
            "contact": {"email": "it-help@example.com", "phone": "02-123-4567"},
        }
    return {
        "ok": True,
        "message": f"{found['name']}님의 비밀번호 초기화 절차",
        "steps": ["SSO 포털 접속 > 비밀번호 재설정", "본인인증", "새 비밀번호 설정"],
        "contact": {"email": "it-help@example.com", "phone": "02-123-4567"},
    }

def tool_request_id(payload: Dict[str, Any]) -> Dict[str, Any]:
    ticket_no = f"REQ-{int(_time.time())}"
    return {
        "ok": True,
        "message": "ID 발급 신청이 접수되었습니다.",
        "ticket": ticket_no,
        "next": ["HR 포털 신청", "부서장 승인", "IT 계정 생성"],
        "sla": "1~2일",
    }

def tool_owner_lookup(payload: Dict[str, Any]) -> Dict[str, Any]:
    screen = payload.get("screen") or ""
    info = OWNER_FALLBACK.get(screen)
    if not info:
        return {"ok": False, "message": f"'{screen}' 담당자 정보를 찾지 못했습니다."}
    return {"ok": True, "screen": screen, "owner": info}

# ---- 노드 ----
def node_classify(state: BotState) -> BotState:
    llm = make_llm()
    sys_prompt = (
        "당신은 사내 헬프데스크 라우터입니다. "
        "사용자 입력을 reset_password, request_id, owner_lookup, rag_qa 중 하나로 분류하세요. "
        "JSON(intent, arguments)으로만 답하세요."
    )
    msg = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": state["question"]}]
    out = llm.invoke(msg).content
    intent, args = "rag_qa", {}
    try:
        data = json.loads(out)
        intent = data.get("intent", "rag_qa")
        args = data.get("arguments", {}) or {}
    except Exception:
        pass
    return {**state, "intent": intent, "tool_output": args}

def node_reset_pw(state: BotState) -> BotState:
    args = state.get("tool_output", {}) or {}
    res = tool_reset_password(args)
    return {**state, "tool_output": res}

def node_request_id(state: BotState) -> BotState:
    args = state.get("tool_output", {}) or {}
    res = tool_request_id(args)
    return {**state, "tool_output": res}

def node_owner_lookup(state: BotState) -> BotState:
    args = state.get("tool_output", {}) or {}
    res = tool_owner_lookup(args)
    return {**state, "tool_output": res}

def node_rag(state: BotState) -> BotState:
    r = retriever(k=4)
    docs = r.get_relevant_documents(state["question"])
    context = "\n\n".join([f"[{i+1}] {d.page_content[:1200]}" for i, d in enumerate(docs)])
    sources = [{"index": i+1, "source": d.metadata.get("source","unknown"), "page": d.metadata.get("page")} for i,d in enumerate(docs)]
    llm = make_llm(model=AOAI_DEPLOY_GPT4O)
    sys_prompt = "너는 사내 헬프데스크 상담원이다. 컨텍스트를 기반으로 실행 가능한 답변을 한국어로 작성해라."
    user_prompt = f"질문:\n{state['question']}\n\n컨텍스트:\n{context}"
    out = llm.invoke([{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}]).content
    return {**state, "result": out, "sources": sources}

def node_finalize(state: BotState) -> BotState:
    if state["intent"] in ["reset_password", "request_id", "owner_lookup"]:
        res = state.get("tool_output", {})
        if state["intent"] == "reset_password":
            if res.get("ok"):
                text = "✅ 비밀번호 초기화 안내\n\n" + "\n".join(f"{i+1}. {s}" for i,s in enumerate(res.get("steps", [])))
            else:
                text = "❗" + res.get("message","실패")
        elif state["intent"] == "request_id":
            text = f"🆔 ID 발급 신청\n상태: {'접수됨' if res.get('ok') else '실패'}\n티켓: {res.get('ticket','-')}"
        else:
            if res.get("ok"):
                o = res.get("owner", {})
                text = f"👤 '{res.get('screen')}' 담당자\n- 이름: {o.get('owner')}\n- 이메일: {o.get('email')}\n- 연락처: {o.get('phone')}"
            else:
                text = "❗" + res.get("message","조회 실패")
        return {**state, "result": text}
    return state

def build_graph():
    g = StateGraph(BotState)
    g.add_node("classify", node_classify)
    g.add_node("reset_password", node_reset_pw)
    g.add_node("request_id", node_request_id)
    g.add_node("owner_lookup", node_owner_lookup)
    g.add_node("rag", node_rag)
    g.add_node("finalize", node_finalize)
    g.set_entry_point("classify")

    def _route(state: BotState):
        m = state["intent"]
        if m == "reset_password": return "reset_password"
        if m == "request_id": return "request_id"
        if m == "owner_lookup": return "owner_lookup"
        return "rag"

    g.add_conditional_edges("classify", _route, {
        "reset_password":"finalize","request_id":"finalize","owner_lookup":"finalize","rag":"rag"})
    g.add_edge("finalize", END); g.add_edge("rag", END)
    return g.compile()

_graph = None
def pipeline(question: str) -> Dict[str, Any]:
    global _graph
    logger.info("pipeline_in", extra={"extra_data": {"q": question}})
    if _graph is None:
        _graph = build_graph()
    state: BotState = {"question": question, "intent":"", "result":"", "sources":[], "tool_output":{}}
    out = _graph.invoke(state)
    logger.info("pipeline_out", extra={"extra_data": {"intent": out.get("intent","")}})
    return out

# =============================================================
# 5. FastAPI
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

class ChatIn(BaseModel): message: str
class ChatOut(BaseModel): reply: str; intent: str; sources: List[Dict[str, Any]]= []

@api.get("/health")
def health(): return {"ok":True}

@api.post("/chat", response_model=ChatOut)
def chat(payload: ChatIn = Body(...)):
    out = pipeline(payload.message)
    return ChatOut(reply=out.get("result",""), intent=out.get("intent",""), sources=out.get("sources", []))

# =============================================================
# 6. Streamlit UI
# =============================================================
def run_streamlit_ui():
    import streamlit as st

    st.set_page_config(page_title="사내 헬프데스크 RAG", page_icon="💬", layout="wide")
    st.title("💼 사내 헬프데스크 챗봇 (RAG + LangGraph)")

    with st.sidebar:
        st.header("📚 지식베이스(KB)")
        if st.button("인덱스 재빌드"):
            try:
                # Clear previous index
                for ext in [".faiss", ".pkl"]:
                    p = INDEX_DIR / f"{INDEX_NAME}{ext}"
                    if p.exists():
                        p.unlink()
                st.info("인덱스 재생성 중...")
                build_or_load_vectorstore()
                st.success("완료!")
            except Exception as e:
                st.error(f"실패: {e}")
        uploaded = st.file_uploader("문서 업로드 (PDF/CSV/TXT/DOCX)", type=["pdf","csv","txt","md","docx"], accept_multiple_files=True)
        if uploaded:
            KB_DIR.mkdir(parents=True, exist_ok=True)
            for f in uploaded:
                dest = KB_DIR / f.name
                with open(dest, "wb") as w:
                    w.write(f.read())
            st.success(f"{len(uploaded)}개 문서 저장됨. '인덱스 재빌드'를 눌러 반영하세요.")

        st.divider()
        api_host = os.getenv("API_CLIENT_HOST", "localhost")
        api_port = int(os.getenv("API_PORT", 8000))
        api_base_url = f"http://{api_host}:{api_port}"

        # st.header("⚙ 설정")
        # 변수를 사용하여 토글 UI의 텍스트를 동적으로 생성합니다.
        use_api = st.toggle(f"백엔드 API 사용 ({api_base_url}/chat)", value=False)
        #use_api = st.toggle("백엔드 API 사용 (http://localhost:8000/chat)", value=False)
        st.caption("비활성화 시 로컬 파이프라인 직접 호출")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, content in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(content)

    if q := st.chat_input("무엇을 도와드릴까요? 예: '비밀번호 초기화', '인사시스템-사용자관리 담당자'"):



        st.session_state.chat.append(("user", q))
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            with st.spinner("처리 중..."):
                try:
                    if use_api:
                        with httpx.Client(timeout=30.0) as client:
                            resp = client.post(api_base_url, json={"message": q})
                            resp.raise_for_status()
                            data = resp.json()
                            reply = data.get("reply","")
                            sources = data.get("sources", [])
                    else:
                        out = pipeline(q)
                        reply = out.get("result","")
                        sources = out.get("sources", [])
                    st.markdown(reply)
                    if sources:
                        with st.expander("🔎 참조 소스"):
                            for s in sources:
                                line = f"- [{s.get('index')}] {s.get('source')}"
                                if s.get("page") is not None:
                                    line += f" (page {s['page']})"
                                st.write(line)
                    st.session_state.chat.append(("assistant", reply))
                except Exception as e:
                    st.error(f"오류: {e}")

# Auto-run Streamlit UI only when truly inside Streamlit runner

# =============================================================
# 7. 엔트리포인트
# =============================================================
if _is_running_in_streamlit():
    run_streamlit_ui()

# CLI entry for FastAPI
if __name__ == "__main__":
    import argparse
    default_host = os.getenv("API_SERVER_HOST", "0.0.0.0")
    default_port = int(os.getenv("API_PORT", 8000))

    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true", help="Run FastAPI server")
    parser.add_argument("--host", default=default_host)
    parser.add_argument("--port", default=default_port, type=int)
    args = parser.parse_args()

    if args.api:
        uvicorn.run(api, host=args.host, port=args.port)
    else:
        print("Usage:")
        print("  FastAPI : python app.py --api")
        print("  UI      : streamlit run app.py")
