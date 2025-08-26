# ui.py
import os
import streamlit as st
import httpx

# 공통 로직 및 KB/인덱스 관리 함수 임포트
from core import pipeline, KB_DIR, INDEX_DIR, INDEX_NAME, build_or_load_vectorstore

def main():
    st.set_page_config(page_title="사내 헬프데스크 RAG", page_icon="💬", layout="wide")
    st.title("💼 사내 헬프데스크 챗봇 (RAG + LangGraph)")

    with st.sidebar:
        st.header("📚 지식베이스(KB)")
        if st.button("인덱스 재빌드"):
            try:
                for ext in [".faiss", ".pkl"]:
                    p = INDEX_DIR / f"{INDEX_NAME}{ext}"
                    if p.exists(): p.unlink()
                with st.spinner("인덱스 재생성 중..."):
                    build_or_load_vectorstore()
                st.success("완료!")
            except Exception as e:
                st.error(f"실패: {e}")
        
        uploaded = st.file_uploader("문서 업로드 (PDF/CSV/TXT/DOCX)", type=["pdf","csv","txt","md","docx"], accept_multiple_files=True)
        if uploaded:
            KB_DIR.mkdir(parents=True, exist_ok=True)
            for f in uploaded:
                with open(KB_DIR / f.name, "wb") as w:
                    w.write(f.read())
            st.success(f"{len(uploaded)}개 문서 저장됨. '인덱스 재빌드'를 눌러 반영하세요.")

        st.divider()
        api_host = os.getenv("API_CLIENT_HOST", "localhost")
        api_port = int(os.getenv("API_PORT", 8000))
        api_base_url = f"http://{api_host}:{api_port}"

        use_api = st.toggle(f"백엔드 API 사용 ({api_base_url}/chat)", value=True)
        st.caption("비활성화 시 로컬 파이프라인 직접 호출")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for role, content in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(content)

    if q := st.chat_input("무엇을 도와드릴까요?"):
        st.session_state.chat.append(("user", q))
        with st.chat_message("user"): st.markdown(q)

        with st.chat_message("assistant"):
            with st.spinner("처리 중..."):
                try:
                    if use_api:
                        with httpx.Client(timeout=30.0) as client:
                            resp = client.post(f"{api_base_url}/chat", json={"message": q})
                            resp.raise_for_status()
                            data = resp.json()
                            reply = data.get("reply",""); sources = data.get("sources", [])
                    else:
                        out = pipeline(q)
                        reply = out.get("result",""); sources = out.get("sources", [])
                    
                    st.markdown(reply)
                    if sources:
                        with st.expander("🔎 참조 소스"):
                            for s in sources:
                                line = f"- [{s.get('index')}] {s.get('source')}"
                                if s.get("page") is not None: line += f" (page {s['page']})"
                                st.write(line)
                    st.session_state.chat.append(("assistant", reply))
                except Exception as e:
                    st.error(f"오류: {e}")

if __name__ == "__main__":
    main()