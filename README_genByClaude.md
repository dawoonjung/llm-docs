# LLM0707 - LLM 교육과정 실습 프로젝트 🚀

LangChain, RAG, 평가, 에이전트 등 다양한 LLM 기술을 학습하는 4일간의 실습 프로젝트입니다.

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [기능](#-기능)
- [프로젝트 구조](#-프로젝트-구조)
- [기술 스택](#-기술-스택)
- [설치 및 실행](#-설치-및-실행)
- [교육 과정](#-교육-과정)
- [실습 애플리케이션](#-실습-애플리케이션)
- [데이터](#-데이터)
- [환경 설정](#-환경-설정)
- [기여하기](#-기여하기)

## 🎯 프로젝트 개요

이 프로젝트는 **LLM(대형 언어 모델) 교육과정**을 위한 실습 프로젝트로, 다음과 같은 주요 기술들을 단계적으로 학습할 수 있습니다:

- **LangChain**: LLM 애플리케이션 개발 프레임워크
- **RAG (Retrieval-Augmented Generation)**: 검색 증강 생성 시스템
- **벡터 데이터베이스**: ChromaDB, Qdrant를 활용한 임베딩 저장
- **프롬프트 엔지니어링**: Few-shot, Chain of Thought 등 고급 프롬프팅 기법
- **평가 시스템**: RAGAS, Langfuse를 활용한 LLM 평가
- **에이전트**: 도구 호출 및 다국어 처리 에이전트

## ✨ 기능

### 핵심 기능
- 📚 **단계별 학습**: 4일간의 체계적인 교육 과정
- 🤖 **RAG 챗봇**: 근로기준법 관련 질문 답변 시스템
- 🔍 **하이브리드 검색**: 다양한 검색 전략 구현
- 📊 **평가 메트릭**: 검색 및 생성 성능 평가
- 🌐 **다국어 지원**: 한국어/영어 문서 처리
- 💬 **스트리밍 응답**: 실시간 응답 생성

### 지원 기능
- 🎨 **Gradio 웹 인터페이스**: 사용자 친화적인 웹 UI
- 📈 **모니터링**: Langfuse를 통한 프롬프트 관리 및 평가
- 🔄 **대화 히스토리**: 컨텍스트 기반 대화 관리
- 🛠️ **도구 통합**: 다양한 외부 도구 연동

## 🏗️ 프로젝트 구조

```
llm0707/
├── 📁 data/                          # 데이터 및 분석 자료
│   ├── tesla_10k_sections_split.pkl, tesla_10k_sections.pkl, tesla_10k_toc_based_docs.pkl
│   ├── tsla-20241231-gen.pdf, transformer.pdf
│   ├── housing_leasing_law.pdf, personal_info_law.pdf, labor_law.pdf
│   ├── Tesla_EN.md, Rivian_EN.md, 리비안_KR.md, 테슬라_KR.md
│   ├── evaluation_result.csv, ragas_testset.csv, testset.xlsx, kbo_teams_2023.csv
│   ├── housing_faq.txt, housing_faq_formatted.json, housing_faq_formatted_with_summary.json
│   ├── etf_list.csv, etf_info.zip
│   ├── restaurant_menu.txt, restaurant_wine.txt
│   ├── korean_docs_final.jsonl
│   ├── 📁 etf/                       # ETF 관련 데이터
│   │   └── etf_list.csv
│   ├── 📁 etf_info/                  # ETF 상세 정보 (csv 다수)
│   │   └── etf_info_*.csv, test_etf_info_*.csv
│   ├── 📁 docling_output/            # OCR/분석 결과
│   │   ├── tsla_analysis_ocr.pkl, tsla_analysis_ocr.md
│   │   ├── transformer_analysis.md, labor_law.txt
│   └── .DS_Store
├── 📁 chroma_db/                     # ChromaDB 벡터 저장소
│   ├── chroma.sqlite3
│   └── <uuid>/data_level0.bin, ...   # 여러 컬렉션별 벡터 데이터
├── 📁 langchain_qdrant/              # Qdrant 벡터 저장소
│   ├── meta.json, .lock
│   └── 📁 collection/
│       └── 📁 labor_law/
│           └── storage.sqlite
├── 📁 src/                           # (예시) 소스 코드
│   ├── graph.py
│   └── __pycache__/
├── 📁 data copy/                     # (백업/실험용) 데이터 복사본
│   ├── restaurant_menu.txt, restaurant_wine.txt
├── 📓 DAY01_001_Langchain_Components.ipynb
├── 📓 DAY01_002_LangSmith_LCEL.ipynb
├── 📓 DAY01_003_Gradio_Chatbot.ipynb
├── 📓 DAY01_004_RAG.ipynb
├── 📓 DAY01_005_RunnableConfig_Fallback.ipynb
├── 📓 DAY02_001_Prompt_Engineering_Fewshot.ipynb
├── 📓 DAY02_002_Prompt_Engineering_CoT.ipynb
├── 📓 DAY02_003_Langfuse_Pompt_Management.ipynb
├── 📓 DAY02_004_Chat_History.ipynb
├── 📓 DAY02_005_Housing_FAQ_Bot.ipynb
├── 📓 DAY03_001_RAG_Evalution.ipynb
├── 📓 DAY03_002_Retrieval_Metrics.ipynb
├── 📓 DAY03_003_Hybrid_Search.ipynb
├── 📓 DAY03_004_Query_Expansion.ipynb
├── 📓 DAY03_005_Rerank_Compression.ipynb
├── 📓 DAY03_006_Generation_Metrics.ipynb
├── 📓 DAY04_001_LLM-as-Judge-LangChain.ipynb
├── 📓 DAY04_002_Langfuse_Evaluation.ipynb
├── 📓 DAY04_003_ToolCalling_Agent.ipynb
├── 📓 DAY04_004_LangChain_Tools.ipynb
├── 📓 DAY04_005_Multilingual_RAG.ipynb
├── 📓 DAY04_006_MCP.ipynb
├── 📓 DAY05_001_LangGraph_StateGraph.ipynb
├── 📓 DAY05_002_ETF_Data_Collection.ipynb
├── 📓 DAY05_002_ETF_Data_Collection_Windows.ipynb
├── 📓 DAY05_003_ETF_Text2SQL.ipynb
├── 📓 DAY05_004_ETF_Text2SQL_RAG.ipynb
├── 📓 DAY05_005_ETF_Text2SQL_Cardinality.ipynb
├── 📓 DAY05_006_ETF_Recommendation.ipynb
├── 📓 DAY05_007_LangGraph_MessageGraph.ipynb
├── 📓 DAY05_008_LangGraph_ReAct.ipynb
├── 📓 DAY06_001_LangGraph_Memory.ipynb
├── 📓 DAY06_002_LangGraph_HITL.ipynb
├── 📓 DAY06_003_LangGraph_SubGraph.ipynb
├── 📓 DAY06_005_LangGraph_Multi-Agent.ipynb
├── 📓 DAY06_006_LangGraph_SelfRAG.ipynb
├── 📓 DAY06_007_LangGraph_CRAG.ipynb
├── 📓 DAY07_001_LangGraph_Leagal_Agent.ipynb
├── 📓 DAY07_002_Unstructured_Docling.ipynb
├── 📓 DAY07_003_Unstructured_RAG_10-K.ipynb
├── 📓 DAY07_004_neo4j_intro.ipynb
├── 📓 DAY07_005_neo4j_LangChain.ipynb
├── 📓 DAY07_006_neo4j_Structred_ETF.ipynb
├── 📓 DAY07_007_neo4j_GraphRAG_10-K.ipynb
├── 🐍 gradio_rag_app.py              # RAG 챗봇 애플리케이션
├── 🐍 main.py                        # 메인 실행 파일
├── 📋 pyproject.toml                 # 프로젝트 설정
├── 📄 README.md                      # 프로젝트 문서
├── checkpoints.db                    # LangGraph 체크포인트 DB
├── langgraph.json                    # LangGraph 워크플로우 설정
├── .env, .gitignore, .python-version, uv.lock, .DS_Store
```

## 🛠️ 기술 스택

### 핵심 프레임워크
- **LangChain**: LLM 애플리케이션 개발 프레임워크
- **LangGraph**: 그래프 기반 워크플로우
- **Gradio**: 웹 인터페이스 구축

### 벡터 데이터베이스
- **ChromaDB**: 경량 벡터 데이터베이스
- **Qdrant**: 고성능 벡터 검색 엔진

### LLM 제공자
- **OpenAI**: GPT 모델 통합
- **Google GenAI**: Gemini 모델 지원
- **HuggingFace**: 오픈소스 모델 통합
- **Ollama**: 로컬 LLM 실행

### 평가 및 모니터링
- **Langfuse**: 프롬프트 관리 및 평가
- **RAGAS**: RAG 시스템 평가 메트릭

## 🚀 설치 및 실행

### 1. 저장소 클론
```bash
git clone <repository-url>
cd llm0707
```

### 2. 가상환경 설정 (uv 사용)
```bash
# uv 설치 (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 가상환경 생성 및 활성화
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
uv sync
```

### 3. 환경 변수 설정
```bash
# .env 파일 생성
cp .env.example .env

# 필요한 API 키 설정
OPENAI_API_KEY=your_openai_api_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_HOST=your_langfuse_host
```

### 4. 애플리케이션 실행
```bash
# RAG 챗봇 실행
uv run python gradio_rag_app.py

# 또는 기본 애플리케이션 실행
uv run python main.py
```

## 📚 교육 과정

### 📅 DAY 01 - LangChain 기초
1. **Langchain Components** - LangChain 구성요소 학습
2. **LangSmith & LCEL** - 개발 도구 및 체인 구성
3. **Gradio Chatbot** - 웹 인터페이스 구축
4. **RAG 시스템** - 검색 증강 생성 기초
5. **설정 및 폴백** - 시스템 안정성 확보

### 📅 DAY 02 - 프롬프트 엔지니어링
1. **Few-shot Prompting** - 예시 기반 프롬프팅
2. **Chain of Thought** - 단계별 사고 과정
3. **Langfuse 프롬프트 관리** - 프롬프트 버전 관리
4. **Chat History** - 대화 컨텍스트 관리
5. **Housing FAQ Bot** - 실제 사용 사례 구현

### 📅 DAY 03 - RAG 고도화
1. **RAG 평가** - 시스템 성능 측정
2. **검색 메트릭** - 검색 품질 평가
3. **하이브리드 검색** - 다양한 검색 전략
4. **쿼리 확장** - 검색 성능 향상
5. **재순위 및 압축** - 결과 최적화
6. **생성 메트릭** - 답변 품질 평가

### 📅 DAY 04 - 평가 및 에이전트
1. **LLM-as-Judge** - LLM 기반 평가 시스템
2. **Langfuse 평가** - 종합 평가 도구
3. **Tool Calling Agent** - 도구 호출 에이전트
4. **LangChain Tools** - 다양한 도구 활용
5. **다국어 RAG** - 글로벌 서비스 구현
6. **MCP (Model Context Protocol)** - 고급 컨텍스트 관리

### 📅 DAY 05 - LangGraph 심화 & ETF 실전
1. **LangGraph StateGraph** - 상태 기반 대화/워크플로우 설계
2. **ETF 데이터 수집/정제** - 크롤링, 상세정보 수집, CSV 저장
3. **ETF Text2SQL** - 자연어→SQL 변환, RAG 기반 질의응답
4. **ETF 추천 시스템** - 사용자 프로필 기반 맞춤형 ETF 추천
5. **MessageGraph/병렬처리** - 메시지 그래프, 리듀서, 병렬 실행
6. **LangGraph ReAct** - Reason+Act 패턴 실습

### 📅 DAY 06 - LangGraph 고급: 메모리, 멀티에이전트, Self-RAG, CRAG
1. **LangGraph Memory** - 단기/장기 메모리, 체크포인트, 대화 상태 관리
2. **HITL/서브그래프** - Human-in-the-loop, 복합 워크플로우
3. **멀티에이전트** - Supervisor 패턴, 전문화된 에이전트 협업
4. **Self-RAG** - 자기반영 기반 RAG, 환각/유용성 평가
5. **CRAG** - Corrective RAG, 지식 정제 및 외부 검색 결합

### 📅 DAY 07 - Unstructured RAG & Graph 기반 RAG
1. **Unstructured Docling** - 비정형 문서(10-K 등) 분할/정제/임베딩
2. **Unstructured RAG** - 대용량 문서 기반 RAG 파이프라인
3. **Neo4j + LangChain** - 지식그래프 구축, 쿼리, RAG 결합
4. **GraphRAG** - 그래프 기반 RAG, 복합 질의응답

## 🤖 실습 애플리케이션

### RAG 챗봇 (`gradio_rag_app.py`)

근로기준법 관련 질문에 답변하는 실용적인 RAG 챗봇입니다.

#### 주요 기능
- 📋 **근로기준법 질의응답**: 법률 문서 기반 정확한 답변
- 💬 **대화 히스토리**: 이전 대화 맥락을 고려한 답변
- 🔄 **스트리밍 응답**: 실시간 답변 생성
- 🔍 **MMR 검색**: 다양성을 고려한 검색 결과

#### 사용 예시
```python
# 애플리케이션 실행
uv run python gradio_rag_app.py

# 웹 브라우저에서 http://localhost:7860 접속
```

#### 예시 질문
- "연차휴가는 언제부터 사용할 수 있나요?"
- "최저임금은 얼마인가요?"
- "해고예고기간은 얼마나 되나요?"
- "출산휴가는 얼마나 받을 수 있나요?"

## 📊 데이터

### 문서 데이터
- **Tesla/Rivian 정보**: 영어/한국어 마크다운 문서
- **근로기준법**: PDF 형태의 법률 문서
- **주택 FAQ**: 텍스트 및 JSON 형태의 질의응답 데이터

### 평가 데이터
- **RAGAS 테스트셋**: RAG 시스템 평가용 데이터
- **평가 결과**: CSV 형태의 성능 측정 결과
- **KBO 팀 데이터**: 예시 구조화 데이터

## ⚙️ 환경 설정

### 필수 환경 변수
```bash
# OpenAI API
OPENAI_API_KEY=sk-...

# Langfuse (옵션)
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# 기타 API 키 (필요시)
GOOGLE_API_KEY=...
TAVILY_API_KEY=...
```

### Python 버전
- **Python 3.12+** (`.python-version` 파일 참조)

### 의존성 관리
- **uv**: 빠른 Python 패키지 관리자 사용
- **pyproject.toml**: 프로젝트 설정 및 의존성 정의

## 🤝 기여하기

### 개발 환경 설정
```bash
# 저장소 포크 및 클론
git clone <your-fork-url>
cd llm0707

# 개발 환경 설정
uv sync --dev

# 브랜치 생성
git checkout -b feature/your-feature
```

### 커밋 가이드라인
- **feat**: 새로운 기능 추가
- **fix**: 버그 수정
- **docs**: 문서 업데이트
- **refactor**: 코드 리팩토링
- **test**: 테스트 추가/수정

### 이슈 및 PR
- 이슈 생성 시 명확한 설명과 재현 방법 제공
- PR 시 변경사항에 대한 상세한 설명 포함
- 코드 리뷰 후 병합 진행

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 문의

프로젝트 관련 문의사항이나 버그 리포트는 GitHub Issues를 통해 제출해 주세요.

---

**Happy Learning! 🎓**