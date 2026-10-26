# gachon_admission_agentic_rag

가천대학교 2027학년도 입학 자료(HWP 1개, PDF 4개)를 근거로 수험생·학부모의 질문에 답하는 **Agentic RAG 챗봇**입니다.
답변마다 근거가 된 문서·쪽과 원문 줄을 함께 보여 줍니다.

```
Q: 정시모집 인공지능학과의 모집군과 일반전형1·일반전형2 모집인원은 각각 얼마인가요?
A: 정시모집 인공지능학과는 다군이며, 일반전형 1은 46명, 일반전형 2는 21명입니다.

[근거]
- 2027학년도 수시 모집요강 97쪽 · 2027학년도 수시모집안내: "계열: 자연 | 모집단위: 인공지능학과 | 일반전형 1 다군: 46 | 일반전형 2 다군: 21"
```

> AI 부트캠프(가천대) 「AI Agent 개발」 중간 평가 과제로 시작해, 평가 이후 근거 인용 기능을 더해 다듬은 버전입니다.

## 주요 특징

- **배포용(암호화) HWP 파싱**: `langchain-hwp-hwpx-loader`의 pyhwp 백엔드로 문서를 열고, 표마다 셀 좌표를 받아 병합 셀을 채운 행 단위 텍스트로 복원합니다.
- **PDF 표 구조 복원**: Upstage Document Parse가 돌려주는 표 HTML(rowspan/colspan)을 그리드로 펼쳐 `열 머리글: 값` 형태의 행으로 만듭니다. Upstage 키가 없으면 `pymupdf4llm`으로 자동 전환합니다.
- **★ 면접일정표 좌표 파싱**: 파서들이 인접한 두 행을 합쳐 학과별 날짜가 섞이는 문제가 있어, PyMuPDF로 글자 좌표를 읽어 "★의 세로 위치 → 학과, 가로 위치 → 날짜·오전/오후"로 직접 매칭합니다.
- **전형명 태깅**: 생김새가 비슷한 전형별 표(수능최저학력기준 등)를 섞지 않도록, 펼친 표 행마다 바로 위 "OO 전형" 제목을 붙입니다.
- **하이브리드 검색**: 벡터 검색(`text-embedding-3-large`, Chroma)과 BM25(Kiwi 형태소)를 RRF로 합치고, 작은 청크로 찾은 뒤 그 청크가 속한 쪽 전체를 LLM에 넘깁니다(small-to-big). 한 문서가 결과를 독차지하지 않도록 문서당 섹션 수를 제한합니다.
- **Agentic RAG**: 질문으로 먼저 검색한 결과를 LangChain Agent(`gpt-5.4-mini`, Responses API)에 넘기고, Agent가 근거가 부족하다고 판단할 때만 질의를 바꿔 추가 검색합니다. 계산은 `calculate` 도구로 합니다.
- **근거 인용**: LLM은 실제로 쓴 검색 결과의 근거 ID만 고르고, 문서명·쪽과 원문 줄은 코드가 실제 섹션에서 가져와 붙입니다. 출처가 틀리게 적힐 수 없습니다.

## 동작 흐름

```
HWP / PDF ─► 파싱(표 구조 복원, ★표 좌표 파싱, 전형명 태깅)
          ─► 분할(검색용 700자 청크 + 답변용 쪽 단위 섹션)
          ─► 임베딩·Chroma 저장 / BM25 색인
질문 ─► 하이브리드 검색(상위 6개 섹션) ─► Agent 판단 ─┬─► 답변 + 근거 ID
                                                    └─► (부족하면) 추가 검색
     ─► 근거 ID 검증 후 문서명·쪽·원문 줄 첨부
```

## 성능

자체 검증 세트로 측정했습니다. 문항당 3회 호출해 2회 이상 맞으면 정답으로 보고, LLM 심판(`gpt-5.4-mini`)이 판정했습니다. 검증 세트는 정답을 응시자 쪽이 보지 않도록 별도 세션에서 원문과 렌더링한 표 이미지로 정답을 확정해 만들었습니다.

| 세트 | 문항 수 | 정확도 | 평균 응답 시간 |
|---|---|---|---|
| 검증 v1 (친절·구어체 질문 짝) | 83 | 97.6% | 1.81s |
| 검증 v2 (RAG 평가 지표별 설계) | 80 | 96.2% | 2.34s |
| 최종 테스트 (튜닝에 쓰지 않은 세트, 1회만 채점) | 50 | 94.0% | 2.20s |

- 위 수치는 근거 인용 기능과 시스템 프롬프트 규칙 일부(간접 정황 단정 금지, 답변불가 시 수치 덧붙이기 금지)를 넣기 **전** 버전으로 측정했습니다.
- 응답 시간은 질문 임베딩·검색·답변 생성 시간을 포함하고, 문서 파싱·임베딩·DB 구축 시간은 제외했습니다.

### 알려진 한계

- 셀 병합으로 의미가 표현된 표(예: 수학 칸이 "기하·미적분"과 "확률과 통계"에 걸쳐 합쳐져 "선택과목 제한 없음"을 뜻하는 경우)는 펼치는 과정에서 그 의미가 사라집니다.
- 여러 전형의 값을 모두 모아 더해야 하는 집계형 질문에 약합니다.
- 근거 인용의 원문 줄은 PDF 원문 그대로가 아니라 표를 펼쳐 만든 텍스트입니다.

## 실행 방법

### Google Colab

1. `gachon_admission_agentic_rag.ipynb`를 Colab에 업로드합니다.
2. 왼쪽 파일 패널에 `data/`의 데이터 파일 5개를 올립니다(`/content` 아래 3단계 이내면 폴더 위치는 자동으로 찾습니다).
   - `01_2027학년도_재외국민_모집요강.hwp`
   - `02_2027학년도_지역의사선발전형_안내.pdf`
   - `03_2027학년도_수시_모집요강.pdf`
   - `04_2027학년도_정시_모집요강.pdf`
   - `05_2027학년도_수시_면접고사_일정.pdf`
3. 🔑 보안 비밀에 `OPENAI_API_KEY`(필수), `UPSTAGE_API_KEY`(권장)를 등록하고 노트북 액세스를 켭니다. 노트북 상단의 키 입력 칸에 직접 넣어도 됩니다.
4. 런타임 → 모두 실행. 이후 `answer("질문")`으로 질문합니다.

처음 실행할 때 문서 파싱·임베딩으로 몇 분 걸리고, 결과는 `rag_index/`에 캐시되어 다음 실행부터는 바로 불러옵니다.

### 로컬

```bash
pip install "langchain>=1.2,<2" "langchain-openai>=1.1,<2" "langchain-community>=0.4,<0.5" "langchain-classic>=1.0,<2" "langchain-chroma>=1.0,<2" "langchain-text-splitters>=1.0,<2" "langchain-hwp-hwpx-loader[distribution]==0.2.1" rank_bm25 kiwipiepy "pymupdf>=1.26" pymupdf-layout "pymupdf4llm>=0.2" python-dotenv
```

프로젝트 루트의 `.env`에 키를 넣고 노트북을 실행합니다.

```
OPENAI_API_KEY=...
UPSTAGE_API_KEY=...
```

## 폴더 구조

```
gachon_admission_agentic_rag.ipynb  # 실행용 노트북 (work/rag.py에서 생성)
data/                               # 2027학년도 가천대 입학 자료 5개 (HWP 1, PDF 4)
work/rag.py                         # 노트북 원본 소스 (# %% 셀 구분)
work/py2nb.py                       # rag.py → gachon_admission_agentic_rag.ipynb 변환
work/validate.py                    # 검증 세트 채점 (LLM 심판, 3회 호출 다수결, 검색 recall/precision)
work/retrieval_eval.py              # 검색 단계만 평가 (LLM 호출 없음)
validation/                         # 검증·최종 테스트 세트 (CSV)
```

노트북을 고칠 때는 `work/rag.py`를 수정한 뒤 변환합니다.

```bash
python work/py2nb.py
```

검증 세트 채점:

```bash
python work/validate.py --file validation/val_pairs_v2.csv --trials 3 --workers 6
```

## 사용 기술

LangChain 1.x (`create_agent`) · OpenAI `gpt-5.4-mini` / `text-embedding-3-large` · Upstage Document Parse · Chroma · BM25 + Kiwi · PyMuPDF / pymupdf4llm · langchain-hwp-hwpx-loader (pyhwp)

## 데이터 출처

`data/`의 입학 자료는 가천대학교 입학처가 공개한 2027학년도 모집요강·안내 자료이며, 저작권은 가천대학교에 있습니다.
학습·실습 목적으로 함께 담았고, 최신·공식 정보는 가천대학교 입학처 홈페이지에서 확인하세요.
