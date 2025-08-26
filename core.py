# core.py
import os
import json
import logging
import time as _time
from typing import TypedDict, List, Dict, Any, Optional
from pathlib import Path

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# =============================================================
# 1. 공통 설정 / 환경 변수
# =============================================================
load_dotenv()

# 로거 설정
logger = logging.getLogger("helpdesk-bot")
if not logger.handlers:
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
# 2. RAG 유틸리티
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
                if suf == ".pdf": docs.extend(PyPDFLoader(str(p)).load())
                elif suf == ".csv": docs.extend(CSVLoader(file_path=str(p), encoding="utf-8").load())
                elif suf in [".txt", ".md"]: docs.extend(TextLoader(str(p), encoding="utf-8").load())
                elif suf == ".docx": docs.extend(Docx2txtLoader(str(p)).load())
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
- 담당자 조회: 포털 상단 검색창에 화면/메뉴명을 입력하면 담당자 카드가 표시됨."""
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
# 3. LangGraph (도구 + 노드)
# =============================================================
class BotState(TypedDict):
    question: str; intent: str; result: str
    sources: List[Dict[str, Any]]; tool_output: Dict[str, Any]

def tool_reset_password(payload: Dict[str, Any]) -> Dict[str, Any]:
    user = payload.get("user") or ""
    found = EMPLOYEE_DIR.get(user)
    if not found:
        return {"ok": False, "message": "사번/계정이 확인되지 않습니다."}
    return {"ok": True, "message": f"{found['name']}님의 비밀번호 초기화 절차 안내", "steps": ["SSO 포털 접속 > 비밀번호 재설정", "본인인증", "새 비밀번호 설정"]}

def tool_request_id(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, "message": "ID 발급 신청이 접수되었습니다.", "ticket": f"REQ-{int(_time.time())}"}

def tool_owner_lookup(payload: Dict[str, Any]) -> Dict[str, Any]:
    screen = payload.get("screen") or ""
    info = OWNER_FALLBACK.get(screen)
    if not info:
        return {"ok": False, "message": f"'{screen}' 담당자 정보를 찾지 못했습니다."}
    return {"ok": True, "screen": screen, "owner": info}

def node_classify(state: BotState) -> BotState:
    llm = make_llm()
    sys_prompt = ("당신은 사내 헬프데스크 라우터입니다. 사용자 입력을 reset_password, request_id, owner_lookup, rag_qa 중 하나로 분류하세요. JSON(intent, arguments)으로만 답하세요.")
    msg = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": state["question"]}]
    out = llm.invoke(msg).content
    intent, args = "rag_qa", {}
    try:
        data = json.loads(out)
        intent = data.get("intent", "rag_qa")
        args = data.get("arguments", {}) or {}
    except Exception: pass
    return {**state, "intent": intent, "tool_output": args}

def node_reset_pw(state: BotState) -> BotState: return {**state, "tool_output": tool_reset_password(state.get("tool_output", {}))}
def node_request_id(state: BotState) -> BotState: return {**state, "tool_output": tool_request_id(state.get("tool_output", {}))}
def node_owner_lookup(state: BotState) -> BotState: return {**state, "tool_output": tool_owner_lookup(state.get("tool_output", {}))}

def node_rag(state: BotState) -> BotState:
    docs = retriever(k=4).get_relevant_documents(state["question"])
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
            text = f"✅ 비밀번호 초기화 안내\n\n" + "\n".join(f"{i+1}. {s}" for i,s in enumerate(res.get("steps", []))) if res.get("ok") else f"❗{res.get('message','실패')}"
        elif state["intent"] == "request_id":
            text = f"🆔 ID 발급 신청\n상태: {'접수됨' if res.get('ok') else '실패'}\n티켓: {res.get('ticket','-')}"
        else:
            text = f"👤 '{res.get('screen')}' 담당자\n- 이름: {res.get('owner', {}).get('owner')}\n- 이메일: {res.get('owner', {}).get('email')}\n- 연락처: {res.get('owner', {}).get('phone')}" if res.get("ok") else f"❗{res.get('message','조회 실패')}"
        return {**state, "result": text}
    return state

def build_graph():
    g = StateGraph(BotState)
    g.add_node("classify", node_classify); g.add_node("reset_password", node_reset_pw); g.add_node("request_id", node_request_id)
    g.add_node("owner_lookup", node_owner_lookup); g.add_node("rag", node_rag); g.add_node("finalize", node_finalize)
    g.set_entry_point("classify")
    g.add_conditional_edges("classify", lambda s: s["intent"], {"reset_password":"finalize", "request_id":"finalize", "owner_lookup":"finalize", "rag":"rag"})
    g.add_edge("finalize", END); g.add_edge("rag", END)
    return g.compile()

_graph = None
def pipeline(question: str) -> Dict[str, Any]:
    global _graph
    logger.info("pipeline_in", extra={"extra_data": {"q": question}})
    if _graph is None: _graph = build_graph()
    state: BotState = {"question": question, "intent":"", "result":"", "sources":[], "tool_output":{}}
    out = _graph.invoke(state)
    logger.info("pipeline_out", extra={"extra_data": {"intent": out.get("intent","")}})
    return out