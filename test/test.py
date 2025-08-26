import json                # JSON(자바스크립트 객체 표기법) 데이터 인코딩/디코딩 처리 모듈
import re                  # 정규표현식(Regular Expression) 처리 모듈
import pytest              # 파이썬 테스트 프레임워크(Pytest) - 단위/통합 테스트 실행에 사용
from fastapi.testclient import TestClient  # FastAPI용 테스트 클라이언트 (API 호출 테스트)
import app as appmod       # app.py 모듈을 가져와서 내부 함수/객체를 활용
 
# =============================================================
# Fixtures & Helpers
# =============================================================
@pytest.fixture(scope="module")
def client():
    """FastAPI 테스트 클라이언트 Fixture"""
    return TestClient(appmod.api)


def force_recompile_graph():
    """
    노드를 monkeypatch한 후 LangGraph가 다시 빌드되도록 강제합니다.
    """
    appmod._graph = None

# =============================================================
# 1. API 기본 동작 테스트 (API Contract Tests)
# =============================================================
def test_health_ok(client):
    """/health 엔드포인트가 정상적으로 200 OK와 {"ok": True}를 반환하는지 테스트합니다."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_chat_bad_request(client):
    """'message' 필드가 없는 잘못된 요청에 대해 422 Unprocessable Entity를 반환하는지 테스트합니다."""
    r = client.post("/chat", json={"session_id": "bad_request"})
    assert r.status_code == 422


# =============================================================
# 2. 인텐트별 상세 로직 테스트 (Node-level Intent Flow Tests)
# =============================================================
def test_reset_password_flow(client, monkeypatch):
    """'reset_password' 인텐트 흐름을 테스트합니다."""
    # node_classify를 패치하여 'reset_password' 인텐트로 강제 라우팅
    # Monkeypatch classifier to route to reset_password with a known user
    def fake_classify(state):
        state["intent"] = "reset_password"
        state["tool_output"] = {"user": "kim.s"}
        return state

    monkeypatch.setattr(appmod, "node_classify", fake_classify, raising=True)
    force_recompile_graph()

    # API 호출 및 결과 검증
    r = client.post("/chat", json={"message": "비밀번호 초기화 해줘"})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "reset_password"
    assert "비밀번호 초기화" in data["reply"]
    # should include ordered steps
    assert "1." in data["reply"] and "2." in data["reply"]


def test_request_id_flow(client, monkeypatch):
    """'request_id' 인텐트 흐름을 테스트합니다."""
    # node_classify를 패치하여 'request_id' 인텐트로 강제 라우팅
    def fake_classify(state):
        state["intent"] = "request_id"
        state["tool_output"] = {"name": "홍길동", "dept": "IT운영"}
        return state

    monkeypatch.setattr(appmod, "node_classify", fake_classify, raising=True)
    force_recompile_graph()

    # API 호출 및 결과 검증
    r = client.post("/chat", json={"message": "신규 계정 발급 신청"})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "request_id"
    assert "ID 발급 신청" in data["reply"]
    # ticket format REQ-<digits>
    assert re.search(r"REQ-\d+", data["reply"]) # 티켓 번호 형식 검증


def test_owner_lookup_flow(client, monkeypatch):
    """'owner_lookup' 인텐트 흐름을 테스트합니다."""
    # node_classify를 패치하여 'owner_lookup' 인텐트로 강제 라우팅
    def fake_classify(state):
        state["intent"] = "owner_lookup"
        state["tool_output"] = {"screen": "인사시스템-사용자관리"}
        return state

    monkeypatch.setattr(appmod, "node_classify", fake_classify, raising=True)
    force_recompile_graph()

    # API 호출 및 결과 검증
    r = client.post("/chat", json={"message": "인사시스템 사용자관리 담당자"})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "owner_lookup"
    assert "담당자" in data["reply"]
    assert any(k in data["reply"] for k in ["홍길동", "owner.hr@example.com"])


def test_rag_flow_with_fake_answer(client, monkeypatch):
    """'rag_qa' 인텐트 흐름을 테스트합니다."""
    # node_classify를 패치하여 'rag_qa' 인텐트로 강제 라우팅
    def fake_classify(state):
        state["intent"] = "rag_qa"
        state["tool_output"] = {}
        return state

    # node_rag를 패치하여 실제 LLM/RAG 호출을 회피하고 고정된 답변을 반환
    def fake_rag(state):
        answer = "핵심 요약: 사내 규정에 따라 신청 양식을 제출하고 승인 후 계정이 생성됩니다."
        sources = [
            {"index": 1, "source": "seed-faq.txt", "page": None},
            {"index": 2, "source": "kb/guide.pdf", "page": 3},
        ]
        state["result"] = answer
        state["sources"] = sources
        return state

    monkeypatch.setattr(appmod, "node_classify", fake_classify, raising=True)
    monkeypatch.setattr(appmod, "node_rag", fake_rag, raising=True)
    force_recompile_graph()

    # API 호출 및 결과 검증
    r = client.post("/chat", json={"message": "ID 발급 절차 알려줘"})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "rag_qa"
    assert "핵심 요약" in data["reply"]
    assert isinstance(data.get("sources"), list) and len(data["sources"]) >= 1

# =============================================================
# import types # 파이썬 내부 객체 타입 관련 모듈 (FunctionType, GeneratorType, ModuleType 등 제공)
# 3. 통합 테스트: 전체 파이프라인 동작 테스트 (End-to-End Pipeline Tests)
# def test_chat_with_monkeypatched_pipeline(client, monkeypatch):
#     """
#     외부 LLM/AOAI 호출 없이도 테스트가 가능하도록 pipeline을 스텁으로 교체.
#     """
#     def fake_pipeline(question: str):
#         # 간단한 라우팅 흉내 + 결과 스텁
#         if "비밀번호" in question:
#             return {
#                 "intent": "reset_password",
#                 "result": "✅ 비밀번호 초기화 안내\n\n1. SSO 포털 접속 > 비밀번호 재설정\n2. 본인인증\n3. 새 비밀번호 설정",
#                 "sources": [],
#                 "tool_output": {"ok": True},
#             }
#         elif "담당자" in question:
#             return {
#                 "intent": "owner_lookup",
#                 "result": "👤 '인사시스템-사용자관리' 담당자\n- 이름: 홍길동\n- 이메일: owner.hr@example.com",
#                 "sources": [],
#                 "tool_output": {"ok": True},
#             }
#         else:
#             return {"intent": "rag_qa", "result": "일반 안내입니다.", "sources": [], "tool_output": {}}

#     # pipeline monkeypatch
#     monkeypatch.setattr(appmod, "pipeline", fake_pipeline)

#     payload = {"message": "비밀번호 초기화 방법 알려줘"}
#     res = client.post("/chat", json=payload)
#     assert res.status_code == 200
#     data = res.json()
#     assert data["intent"] == "reset_password"
#     assert "비밀번호" in data["reply"]


# def test_pipeline_smoke(monkeypatch):
#     """
#     pipeline 자체를 직접 호출해 스모크 테스트 (LLM 호출은 스킵).
#     LangGraph 내부 invoke를 더미 함수로 바꿔 최소 동작만 검증.
#     """
#     class FakeCompiledGraph:
#         def invoke(self, state):
#             return {"intent": "rag_qa", "result": "스모크 테스트 OK", "sources": []}

#     def fake_build_graph():
#         return FakeCompiledGraph()

#     # 그래프/LLM 호출 우회
#     monkeypatch.setattr(appmod, "build_graph", fake_build_graph)
#     # 전역 그래프 초기화
#     if hasattr(appmod, "_graph"):
#         appmod._graph = None

#     out = appmod.pipeline("테스트 질문")
#     assert out["result"] == "스모크 테스트 OK"