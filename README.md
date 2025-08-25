# 💼 사내 헬프데스크 챗봇 (RAG + LangGraph + Streamlit/FastAPI)
이 프로젝트는 LangChain과 Streamlit을 활용하여 구축된 사내 헬프데스크 챗봇 애플리케이션입니다. 
RAG(검색 증강 생성) 기술을 사용하여 사내 문서 기반의 답변을 제공하고, 특정 요청에 대해서는 미리 정의된 기능을 실행하는 에이전트 역할을 수행합니다.
사내 ID 발급, 비밀번호 초기화, 특정 화면 담당자 조회 등 **헬프데스크 업무**를 돕는 대화형 RAG 챗봇의 **기본 골격**입니다.  
- UI: **Streamlit**
- Backend API: **FastAPI**
- 오케스트레이션: **LangGraph** (인텐트 라우팅)
- 검색: **FAISS** + **LangChain** (PDF/CSV/TXT/DOCX)
- 모델: **Azure OpenAI (AOAI)**

---

## ✨ 요구사항 내역

-대화형 서비스: Streamlit UI를 통해 사용자 친화적인 챗봇 대화 환경을 제공합니다.
-RAG (Retrieval-Augmented Generation): 사내 매뉴얼이나 문서를 기반으로 정확하고 관련성 높은 답변을 생성합니다. (예제 데이터 포함)
-LangChain/LangGraph 활용: 복잡한 사용자 의도를 분석하고, 질의응답과 특정 기능 실행(툴 사용)을 유연하게 처리합니다.
-기능 실행 (Tools):
-ID 발급: 신규 직원의 사내 ID 발급 절차를 안내합니다.
-비밀번호 초기화: 계정 비밀번호 초기화 방법을 안내합니다.
-담당자 정보: 특정 업무(예: 특정 화면)의 담당자 정보를 제공합니다.
-패키징: requirements.txt를 통해 필요한 모든 의존성을 관리하여 쉽게 환경을 구성할 수 있습니다. (Docker를 활용한 배포 환경 구축 가능)
-프로젝트**Coder**: 산출물 파일 생성(`write_file`) 및 간단 파이썬 표현식 검증(`py_eval`)

## 📄 프로젝트 구조

```
project-root/
├─ README.md
├─ app.py                # Streamlit UI & FastAPI (동일 파일 내)
├─ requirements.txt
├─ kb/                   # 문서 넣는 곳 (PDF/CSV/TXT/DOCX)
├─ index/                # FAISS 인덱스 저장소
└─ test/
   └─ test.py
```

---

## 🧱 필요한 라이브러리 설치 (requirements.txt)

```
# Core app
streamlit>=1.37.0
fastapi>=0.110.0
uvicorn[standard]>=0.30.0
httpx>=0.27.0
python-dotenv>=1.0.1
pydantic>=2.8.2

# LLMs / RAG
langchain>=0.2.14
langchain-community>=0.2.12
langchain-openai>=0.1.22
langgraph>=0.2.33
faiss-cpu>=1.8.0.post1
tiktoken>=0.7.0

# Loaders
pypdf>=4.2.0
docx2txt>=0.8
pandas>=2.2.2

# Tests
pytest>=8.2.0
```
## 🔑 환경 변수 설정 - Azure OpenAI 리소스 생성등 (.env 또는 시스템 환경변수)

```bash
AOAI_ENDPOINT=https://123
AOAI_API_KEY=123
AOAI_API_VERSION=2024-10-21

AOAI_DEPLOY_GPT4O_MINI=gpt-4o-mini
AOAI_DEPLOY_GPT4O=gpt-4o
AOAI_DEPLOY_EMBED_3_LARGE=text-embedding-3-large
AOAI_DEPLOY_EMBED_3_SMALL=text-embedding-3-small
AOAI_DEPLOY_EMBED_ADA=text-embedding-ada-002
```

> **Azure 배포명**(deployment name)을 위 값에 맞춰 주세요. (모델명이자 배포명으로 사용 중)

---

## 🚀 실습 실행 순서
### 실습 기본 설치

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
cd 63
pip install -r requirements.txt
```

### FastAPI 백엔드 실행
```bash
python app.py --api --port 8001 &
```

### Streamlit UI 실행
```bash
streamlit run app.py --server.port=8502

```
```bash
python app.py --api --port 8001&
streamlit run app.py --server.port=8507
```

### 사용 방법
- 사이드바에서 문서를 업로드하고 **인덱스 재빌드**를 눌러 반영합니다.
- 예시 질문:
  - `사내 ID 발급해줘` / `신규 입사자 ID 신청 방법`
  - `비밀번호 초기화`
  - `인사시스템-사용자관리 담당자`
  - `포털 공지 작성 권한 신청`

---

## 🌐 기능 개요

### LangGraph로 인텐트 라우팅
- labels: `reset_password`, `request_id`, `owner_lookup`, `rag_qa`
- LLM이 JSON으로 인텐트와 슬롯(예: `screen`, `user`, `dept`)을 추출
- 각 툴 노드에서 처리를 수행 (데모: 딕셔너리 기반 모의 처리)

### RAG
- `./kb` 폴더의 문서를 불러와 청크 후 **FAISS** 인덱스 생성
- PDF/CSV/TXT/DOCX 지원 (최소 구현)
- 답변 시 참조 소스(파일/페이지) 목록 표시

### UI/백엔드
- Streamlit: 대화형 UI, KB 업로드 & 재인덱스
- FastAPI: `/chat` 엔드포인트 (POST, `{"message": "...", "session_id": "..."}`)

---

## 📊 테스트

```bash
python -m pytest test/test.py
```

---

## 🐳 Docker 패키징 (예시)

### 6.1 FastAPI 이미지

```dockerfile
# Dockerfile.api
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py /app/
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "app.py", "--api", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.2 Streamlit 이미지

```dockerfile
# Dockerfile.ui
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py /app/
COPY kb /app/kb
COPY index /app/index
ENV PYTHONUNBUFFERED=1
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

### 6.3 docker-compose

```yaml
version: "3.9"
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    environment:
      - AOAI_ENDPOINT=${AOAI_ENDPOINT}
      - AOAI_API_KEY=${AOAI_API_KEY}
      - AOAI_API_VERSION=${AOAI_API_VERSION}
      - AOAI_DEPLOY_GPT4O_MINI=${AOAI_DEPLOY_GPT4O_MINI}
      - AOAI_DEPLOY_GPT4O=${AOAI_DEPLOY_GPT4O}
      - AOAI_DEPLOY_EMBED_3_SMALL=${AOAI_DEPLOY_EMBED_3_SMALL}
    ports:
      - "8000:8000"
    restart: unless-stopped

  ui:
    build:
      context: .
      dockerfile: Dockerfile.ui
    depends_on:
      - api
    ports:
      - "8501:8501"
    restart: unless-stopped
```

> 실제 배포에서는 KB/인덱스를 **볼륨**으로 마운트하고, 인증/권한/로그 수집 등을 추가하세요.

---

## 📦 커스터마이즈 가이드

- **툴 연결**: `tool_reset_password`, `tool_request_id`, `tool_owner_lookup`를 실제 사내 API(IAM, ITSM, CMDB 등)로 교체  
- **권한/감사**: FastAPI 미들웨어로 **JWT**, **IP 화이트리스트**, **감사로그** 추가  
- **세션 메모리**: Redis 등 외부 스토리지 사용  
- **문서 파서 강화**: unstructured, tika 등을 활용해 더 많은 포맷 지원  
- **모델 선택**: 분류는 `gpt-4o-mini`, 답변은 `gpt-4o`처럼 **소·대 모델 혼용** 최적화

---

## ✅ 자주 묻는 질문(FAQ)

- **KB가 비어 있어도 동작하나요?**  
  네, 최소한의 seed FAQ로 작동합니다. 문서를 넣고 재인덱스하면 품질이 향상됩니다.

- **Azure 배포명이 다른데요?**  
  `AOAI_DEPLOY_*` 값에 실제 **배포명**을 넣어주세요.

- **Streamlit에서 API 모드 사용이 안돼요**  
  사이드바에서 토글 켜고 `http://localhost:8000`이 떠 있는지 확인해주세요.


## 🚀 Docker 실행 가이드

###  단독 컨테이너 실행

#### FastAPI (백엔드 API)
```bash
docker build -f Dockerfile.api -t helpdesk-api .
docker run --rm -p 8000:8000 --env-file .env helpdesk-api
```

#### Streamlit (UI)
```bash
docker build -f Dockerfile.ui -t helpdesk-ui .
docker run --rm -p 8501:8501 --env-file .env helpdesk-ui
```

### docker-compose 실행 (API + UI 동시 구동)
```bash
docker compose up --build
```

- UI: [http://localhost:8501](http://localhost:8501)  
- API: [http://localhost:8000](http://localhost:8000)  
- `./kb`, `./index` 폴더는 **볼륨 마운트** 되어 컨테이너 재기동 시에도 데이터 유지됩니다.

### .dockerignore
이미지 최적화를 위해 `.dockerignore`를 포함했습니다.

### 정리
- `.env` 값만 실제 환경에 맞게 교체하면 됩니다.  
- `docker-compose.yml`이 `.env`를 자동으로 읽어 환경변수를 각 서비스에 주입합니다.  
- `app.py`는 `load_dotenv()` 호출로 .env를 명시적으로 불러옵니다.  

