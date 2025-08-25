# -*- coding: utf-8 -*-
"""
Pytest test suite for the Helpdesk RAG chatbot

Goals
- Exercise FastAPI endpoints deterministically (no external AOAI calls)
- Validate each intent path: reset_password, request_id, owner_lookup, rag_qa
- Show how to monkeypatch LangGraph node functions for reliable tests
"""
import re
import pytest
from fastapi.testclient import TestClient

import app as appmod


@pytest.fixture(scope="module")
def client():
    return TestClient(appmod.api)


def force_recompile_graph():
    """Ensure LangGraph is rebuilt after monkeypatching nodes."""
    appmod._graph = None


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_reset_password_flow(client, monkeypatch):
    # 1) Monkeypatch classifier to route to reset_password with a known user
    def fake_classify(state):
        state["intent"] = "reset_password"
        state["tool_output"] = {"user": "kim.s"}
        return state

    monkeypatch.setattr(appmod, "node_classify", fake_classify, raising=True)
    force_recompile_graph()

    r = client.post("/chat", json={"message": "비밀번호 초기화 해줘", "session_id": "t"})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "reset_password"
    assert "비밀번호 초기화" in data["reply"]
    # should include ordered steps
    assert "1." in data["reply"] and "2." in data["reply"]


def test_request_id_flow(client, monkeypatch):
    def fake_classify(state):
        state["intent"] = "request_id"
        state["tool_output"] = {"name": "홍길동", "dept": "IT운영"}
        return state

    monkeypatch.setattr(appmod, "node_classify", fake_classify, raising=True)
    force_recompile_graph()

    r = client.post("/chat", json={"message": "신규 계정 발급 신청", "session_id": "t"})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "request_id"
    assert "ID 발급 신청" in data["reply"]
    # ticket format REQ-<digits>
    assert re.search(r"REQ-\d+", data["reply"])


def test_owner_lookup_flow(client, monkeypatch):
    def fake_classify(state):
        state["intent"] = "owner_lookup"
        state["tool_output"] = {"screen": "인사시스템-사용자관리"}
        return state

    monkeypatch.setattr(appmod, "node_classify", fake_classify, raising=True)
    force_recompile_graph()

    r = client.post("/chat", json={"message": "인사시스템 사용자관리 담당자", "session_id": "t"})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "owner_lookup"
    assert "담당자" in data["reply"]
    # default fallback owner name from app.py OWNER_FALLBACK
    assert any(k in data["reply"] for k in ["홍길동", "owner.hr@example.com"])


def test_rag_flow_with_fake_answer(client, monkeypatch):
    # Replace classify -> rag_qa
    def fake_classify(state):
        state["intent"] = "rag_qa"
        state["tool_output"] = {}
        return state

    # Replace RAG node to avoid FAISS/embeddings and LLM calls
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

    r = client.post("/chat", json={"message": "ID 발급 절차 알려줘", "session_id": "t"})
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "rag_qa"
    assert "핵심 요약" in data["reply"]
    # sources should be returned and have at least one entry
    assert isinstance(data.get("sources"), list) and len(data["sources"]) >= 1
