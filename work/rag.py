# %% [markdown]
# # 가천대학교 2027학년도 입학 안내 Agentic RAG 챗봇
#
# - 데이터: 제공된 5개 문서(HWP 1개, PDF 4개)만 사용합니다. 웹 검색·외부 조회 도구는 없습니다.
# - 흐름: 문서 파싱(HWP 표 병합셀 복원, PDF는 Upstage Document Parse 표 구조 복원) → 분할 → 임베딩·Chroma 저장 → 벡터+BM25 하이브리드 검색
#   → LangChain Agent가 검색 결과를 보고 추가 검색 여부를 판단해 답변.
# - 사용법: 위에서부터 모든 셀을 실행한 뒤 `answer(question)`을 호출합니다. (인덱싱은 최초 1회만 수행)
# - API Key는 제출본에서 비워 두었습니다. 아래 키 입력 칸(또는 Colab 보안 비밀·`.env`)에 넣고 실행하세요.
# - 데이터: `dataset.zip`을 푼 폴더(5개 파일)를 작업 폴더(예: `/content`) 아래 3단계 이내에서 자동으로 찾습니다.

# %%
!pip install -q "langchain>=1.2,<2" "langchain-openai>=1.1,<2" "langchain-community>=0.4,<0.5" "langchain-classic>=1.0,<2" "langchain-chroma>=1.0,<2" "langchain-text-splitters>=1.0,<2" "langchain-hwp-hwpx-loader[distribution]==0.2.1" rank_bm25 kiwipiepy "pymupdf>=1.26" pymupdf-layout "pymupdf4llm>=0.2" python-dotenv

# %%
import os
from pathlib import Path

# ---- API Key 입력 (제출본에서는 비워 둠) ----
# OPENAI_API_KEY: 필수
# UPSTAGE_API_KEY: 권장 (있으면 PDF 표를 Upstage Document Parse로 파싱 → 표 구조가 더 정확함, 없으면 pymupdf4llm 사용)
OPENAI_API_KEY = ""
UPSTAGE_API_KEY = ""

for key, value in (("OPENAI_API_KEY", OPENAI_API_KEY), ("UPSTAGE_API_KEY", UPSTAGE_API_KEY)):
    if value:
        os.environ[key] = value
    if not os.environ.get(key):
        try:
            from google.colab import userdata  # Colab 보안 비밀(🔑)에 등록한 경우
            os.environ[key] = userdata.get(key) or ""
        except Exception:
            pass
try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))  # .env 파일이 있으면 사용 (이미 설정된 값은 덮어쓰지 않음)
except Exception:
    pass
assert os.environ.get("OPENAI_API_KEY"), "위 OPENAI_API_KEY에 키를 입력하세요."
print("UPSTAGE_API_KEY:", "있음" if os.environ.get("UPSTAGE_API_KEY") else "없음 (pymupdf4llm으로 파싱)")

# ---- 데이터 위치: 5개 파일이 들어 있는 폴더를 자동으로 찾음 (필요하면 DATA_DIR을 직접 지정) ----
DATA_DIR = None
DOC_FILES = [
    "01_2027학년도_재외국민_모집요강.hwp",
    "02_2027학년도_지역의사선발전형_안내.pdf",
    "03_2027학년도_수시_모집요강.pdf",
    "04_2027학년도_정시_모집요강.pdf",
    "05_2027학년도_수시_면접고사_일정.pdf",
]


def find_data_dir(max_depth: int = 3) -> Path:
    import unicodedata
    target = unicodedata.normalize("NFC", DOC_FILES[2])
    for root in (Path.cwd(), Path("/content"), Path.cwd().parent):
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            depth = len(Path(dirpath).relative_to(root).parts)
            # 깊이 제한, Colab의 Google Drive 마운트·샘플 폴더는 건너뜀
            dirnames[:] = [] if depth >= max_depth else [d for d in dirnames if d not in ("drive", "sample_data") and not d.startswith(".")]
            if any(unicodedata.normalize("NFC", f) == target for f in filenames):
                return Path(dirpath)
    raise FileNotFoundError("데이터 폴더를 찾지 못했습니다. DATA_DIR에 5개 파일이 있는 폴더 경로를 지정하세요.")


DATA_DIR = Path(DATA_DIR) if DATA_DIR else find_data_dir()
INDEX_DIR = Path("rag_index")
INDEX_DIR.mkdir(exist_ok=True)
print("DATA_DIR =", DATA_DIR)

# %% [markdown]
# ## 1. Load — HWP·PDF 파싱
# - HWP(배포용 문서): `langchain-hwp-hwpx-loader`(pyhwp 백엔드)의 elements 모드로 표마다 셀 좌표를 받아,
#   병합 셀 값을 채운 행 단위 텍스트(`열1 | 열2 | ...`)로 복원합니다. 문서 전체가 표 레이아웃이라 이 방식이 가장 읽기 좋습니다.
# - PDF: Upstage Document Parse(키가 있을 때)로 쪽마다 추출하고, 표 HTML은 병합 셀을 채운 `열 머리글: 값` 행 텍스트로 펼칩니다.
#   Upstage 키가 없으면 `pymupdf4llm`(표→마크다운)으로 대신 추출합니다.
#   면접일정의 ★ 표는 글자 좌표로 직접 읽어 "모집단위 면접일: 날짜 오전·오후" 문장으로 풀어 씁니다.

# %%
import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser

import pymupdf
import pymupdf4llm
import requests
from langchain_core.documents import Document
from langchain_hwp_hwpx import HwpHwpxLoader


def doc_title(filename: str) -> str:
    # '03_2027학년도_수시_모집요강.pdf' -> '2027학년도 수시 모집요강'
    return Path(filename).stem.split("_", 1)[1].replace("_", " ")


def hwp_table_rows(cells: list[dict], n_rows: int, n_cols: int) -> list[str]:
    grid = [[None] * n_cols for _ in range(n_rows)]
    for i, c in enumerate(cells):
        text = " ".join(c["text"].split())
        for r in range(c["row"], c["row"] + c["rowspan"]):
            for k in range(c["col"], c["col"] + c["colspan"]):
                # 세로 병합은 짧은 머리글성 값만 아래 행에 복제 (긴 본문 셀은 첫 행에만)
                if r == c["row"] or len(text) <= 30:
                    grid[r][k] = (i, text)
    rows = []
    for row in grid:
        vals, last = [], None
        for v in row:
            if v and v[1] and v[1] != "[IMAGE]" and v[0] != last:  # 가로 병합 셀은 한 번만
                vals.append(v[1])
            last = v[0] if v else None
        if vals:
            rows.append(" | ".join(vals))
    return rows


def load_hwp(path: Path) -> list[Document]:
    title = doc_title(path.name)
    sections = []
    for d in HwpHwpxLoader(str(path), mode="elements", distribution_backend="pyhwp").load():
        m = d.metadata
        if m["element_type"] != "table":
            continue  # 본문은 표지 문구뿐
        rows = hwp_table_rows(json.loads(m["cells"]), m["row_count"], m["col_count"])
        if not rows:
            continue
        head = rows[0]  # 표 첫 행 = 섹션 제목(예: '전형일정')
        # 큰 표는 행 묶음(약 2,500자)으로 나눠 섹션으로 만들고, 각 묶음 앞에 섹션 제목을 붙인다.
        buf = []
        for row in rows[1:] or rows:
            if buf and sum(len(x) for x in buf) + len(row) > 2500:
                sections.append((head, buf))
                buf = []
            buf.append(row)
        sections.append((head, buf))
    docs = []
    for i, (head, rows) in enumerate(sections, 1):
        docs.append(Document(f"{head}\n" + "\n".join(rows),
                             metadata={"source": path.name, "title": title, "loc": f"{head} (섹션 {i})"}))
    return docs


def star_schedule_lines(page, caption: str) -> list[str]:
    """면접일정 ★ 표를 글자 좌표로 읽는다: ★의 y → 모집단위 행, x → (날짜, 오전/오후, 전형) 열.
    마크다운 변환은 인접한 두 행을 하나로 합치는 경우가 있어 좌표로 직접 읽는다."""
    words = page.get_text("words")
    ampm = [w for w in words if w[4] in ("오전", "오후")]
    if not ampm:
        return []
    cx = lambda w: (w[0] + w[2]) / 2
    cy = lambda w: (w[1] + w[3]) / 2
    y_ampm = min(w[1] for w in ampm)
    x_left = min(w[0] for w in ampm) - 5

    def phrases(top):  # 같은 줄에서 가까운 단어를 이어 붙여 구절로 ('10월' + '31일(토)')
        out = []
        for w in sorted((w for w in words if round(w[1]) == top and w[0] >= x_left), key=lambda w: w[0]):
            if out and w[0] - out[-1][1] < 6:
                out[-1] = (out[-1][0], w[2], out[-1][2] + " " + w[4])
            else:
                out.append((w[0], w[2], w[4]))
        return [((a + b) / 2, t) for a, b, t in out]

    # 오전/오후 줄 바로 위 35pt 안의 줄들 = 날짜 줄(가까운 것), 전형명 줄(그 위)
    tops = sorted({round(w[1]) for w in words if y_ampm - 35 < w[1] < y_ampm - 2 and w[0] >= x_left}, reverse=True)
    levels = [phrases(t) for t in tops[:2]]
    nearest = lambda ps, x: min(ps, key=lambda p: abs(p[0] - x))[1] if ps else ""
    cols = [(cx(w), nearest(levels[0], cx(w)) if levels else "", w[4], nearest(levels[1], cx(w)) if len(levels) > 1 else "")
            for w in ampm]

    rows = {}  # 행 라벨(모집단위): 표 왼쪽 열, 오전/오후 줄 아래
    for w in words:
        if w[2] < x_left and w[1] > y_ampm and w[4] != "★":
            key = round(cy(w))
            rows[key] = (rows[key] + " " + w[4]) if key in rows else w[4]
    slots = {}
    for w in words:
        if w[4] != "★" or w[1] < y_ampm:
            continue
        ry = min(rows, key=lambda y: abs(y - cy(w)))
        if abs(ry - cy(w)) > 6:
            continue
        _, date, half, group = min(cols, key=lambda c: abs(c[0] - cx(w)))
        slots.setdefault((ry, group), {}).setdefault(date, []).append(half)
    lines = []
    for (ry, group), by_date in sorted(slots.items()):
        who = f"{group} 전형 " if group else ""
        when = ", ".join(f"{d} {'·'.join(h)}" for d, h in by_date.items())
        lines.append(f"{caption} {who}{rows[ry]} 면접일: {when}")
    return lines


def with_star_lines(nontable_text: str, page) -> str:
    """★ 표가 있는 쪽: 표 대신 좌표로 읽은 '모집단위 면접일' 문장을 붙인다."""
    heading = next((ln.strip("# *") for ln in nontable_text.splitlines() if ln.strip("# *")), "")
    return nontable_text + "\n\n" + "\n".join(star_schedule_lines(page, f"[{heading.strip()} 면접고사 일정]"))


def clean_md(md: str) -> str:
    md = re.sub(r"</?(mark|u|sup|sub|span)[^>]*>", "", md)
    md = md.replace("**", "").replace("\t", " ")
    md = re.sub(r"\n\s*(?:\d{1,3}|- \d+ -)\s*\n", "\n", md)  # 쪽 번호만 있는 줄
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def load_pdf_pymupdf(path: Path) -> list[tuple[int, str, str]]:
    pdf = pymupdf.open(path)
    pages = []
    for page in pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False):
        pno, md = page["metadata"]["page_number"], page["text"]
        if "★" in md:
            md = with_star_lines("\n".join(ln for ln in md.splitlines() if not ln.lstrip().startswith("|")), pdf[pno - 1])
        pages.append((pno, md, ""))
    return pages


# ---- Upstage Document Parse (UPSTAGE_API_KEY가 있을 때 사용: 표 구조(rowspan/colspan)를 HTML로 정확히 돌려줌) ----
class _TableHTML(HTMLParser):
    """<table> HTML -> [[(text, rowspan, colspan, is_head), ...], ...] (중첩 표의 글자는 바깥 셀에 포함)"""

    def __init__(self):
        super().__init__()
        self.rows, self.depth, self.cell, self.in_head = [], 0, None, False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self.depth += 1
        elif self.depth == 1 and tag == "thead":
            self.in_head = True
        elif self.depth == 1 and tag == "tbody":
            self.in_head = False
        elif self.depth == 1 and tag == "tr":
            self.rows.append([])
        elif self.depth == 1 and tag in ("td", "th"):
            self.cell = [[], int(a.get("rowspan") or 1), int(a.get("colspan") or 1), self.in_head or tag == "th"]
        elif tag == "br" and self.cell:
            self.cell[0].append(" ")

    def handle_endtag(self, tag):
        if tag == "table":
            self.depth -= 1
        elif self.depth == 1 and tag in ("td", "th") and self.cell:
            if not self.rows:
                self.rows.append([])
            self.rows[-1].append((" ".join("".join(self.cell[0]).split()), *self.cell[1:]))
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell[0].append(data)


def html_table_lines(html: str) -> list[str]:
    """표를 '열 머리글: 값 | ...' 행 단위 텍스트로 펼친다 (병합 셀은 해당 칸마다 채움)."""
    p = _TableHTML()
    p.feed(html)
    grid, heads = {}, set()
    for r, row in enumerate(p.rows):
        c = 0
        for i, (text, rs, cs, is_head) in enumerate(row):
            while (r, c) in grid:
                c += 1
            for dr in range(rs):
                for dc in range(cs):
                    grid[(r + dr, c + dc)] = ((r, i), text)
            if is_head:
                heads.update(r + dr for dr in range(rs))
            c += cs
    n_rows = max((k[0] for k in grid), default=-1) + 1
    n_cols = max((k[1] for k in grid), default=-1) + 1
    labels = []
    for c in range(n_cols):  # 열 머리글 = thead 행들의 글자를 위에서부터 이어 붙임
        parts = []
        for r in sorted(heads):
            t = grid.get((r, c), (None, ""))[1]
            if t and t not in parts:
                parts.append(t)
        labels.append(" ".join(parts))
    lines = [" | ".join(l for l in labels if l)] if heads else []
    for r in range(n_rows):
        if r in heads:
            continue
        vals, last = [], None
        for c in range(n_cols):
            cid, t = grid.get((r, c), (None, ""))
            if t and cid != last:
                vals.append(f"{labels[c]}: {t}" if labels[c] and labels[c] != t else t)
            last = cid
        if vals:
            lines.append(" | ".join(vals))
    return lines


def upstage_elements(path: Path) -> list[dict]:
    cache = INDEX_DIR / f"upstage_{path.stem}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    src = pymupdf.open(path)
    parts = [(s, min(s + 20, len(src))) for s in range(0, len(src), 20)]  # 20쪽씩 나눠 병렬 요청

    def parse(part):
        s, e = part
        sub = pymupdf.open()
        sub.insert_pdf(src, from_page=s, to_page=e - 1)
        for attempt in range(3):
            try:
                res = requests.post("https://api.upstage.ai/v1/document-digitization",
                                    headers={"Authorization": f"Bearer {os.environ['UPSTAGE_API_KEY']}"},
                                    files={"document": (path.name, sub.tobytes(), "application/pdf")},
                                    data={"model": "document-parse", "ocr": "auto",
                                          "output_formats": "['html', 'markdown']", "coordinates": "false"},
                                    timeout=300)
                res.raise_for_status()
                return [{**el, "page": el["page"] + s} for el in res.json()["elements"]]
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3)

    with ThreadPoolExecutor(4) as ex:
        elements = [el for els in ex.map(parse, parts) for el in els]
    cache.write_text(json.dumps(elements, ensure_ascii=False), encoding="utf-8")
    return elements


def load_pdf_upstage(path: Path) -> list[tuple[int, str, str]]:
    pdf = pymupdf.open(path)
    by_page, headers = {}, {}
    for el in upstage_elements(path):
        if el["category"] == "header" and "모집요강" not in el["content"]["markdown"]:
            headers[el["page"]] = el["content"]["markdown"].replace("\t", " ")  # 쪽 머리글 (예: '학생부위주(교과)')
        if el["category"] in ("header", "footer"):
            continue
        if el["category"] == "table":
            by_page.setdefault(el["page"], []).append(("table", "\n".join(html_table_lines(el["content"]["html"]))))
        else:
            by_page.setdefault(el["page"], []).append(("text", el["content"]["markdown"]))
    pages = []
    for pno in sorted(by_page):
        items = by_page[pno]
        if "★" in pdf[pno - 1].get_text():
            md = with_star_lines("\n\n".join(t for k, t in items if k == "text"), pdf[pno - 1])
        else:
            md = "\n\n".join(t for _, t in items)
        pages.append((pno, md, headers.get(pno, "")))
    return pages


PDF_PARSER = "upstage" if os.environ.get("UPSTAGE_API_KEY") else "pymupdf4llm"
print("PDF 파서:", PDF_PARSER)


def load_pdf(path: Path) -> list[Document]:
    title = doc_title(path.name)
    pages = load_pdf_upstage(path) if PDF_PARSER == "upstage" else load_pdf_pymupdf(path)
    docs, current, since = [], "", 0
    for pno, md, header in pages:
        text = clean_md(md)
        # 전형 설명은 2~3쪽에 걸치는데 'OO 전형' 제목은 첫 쪽에만 있다 -> 다음 2쪽까지 이어 붙인다.
        # 비슷한 표(전형별 수능최저 등)를 섞지 않도록, 표에서 펼친 행마다 바로 위 전형 제목을 붙인다.
        if since >= 2:
            current = ""
        since += 1
        start, lines = current, []
        for ln in text.splitlines():
            m = re.match(r"^#+\s*([^#\n]{2,80}?전형(?:\s?\d)?)\s*$", ln)
            if m:
                current, since = m.group(1).strip(), 1
            lines.append(f"({current}) {ln}" if current and ": " in ln and " | " in ln else ln)
        text = "\n".join(lines)
        loc = " · ".join(x for x in (f"{pno}쪽", header, start if start == current else current) if x)
        if len(text) >= 20:
            docs.append(Document(text, metadata={"source": path.name, "title": title, "loc": loc}))
    return docs


def load_all() -> list[Document]:
    files = {unicodedata.normalize("NFC", p.name): p for p in DATA_DIR.iterdir()}
    docs = []
    for name in DOC_FILES:
        path = files[unicodedata.normalize("NFC", name)]
        docs += load_hwp(path) if path.suffix.lower() == ".hwp" else load_pdf(path)
    return docs


SECTIONS_PATH = INDEX_DIR / f"sections_{PDF_PARSER}.json"
if SECTIONS_PATH.exists():  # 파싱 결과 캐시 (재실행 시 빠르게)
    sections = [Document(d["text"], metadata=d["meta"]) for d in json.loads(SECTIONS_PATH.read_text(encoding="utf-8"))]
else:
    sections = load_all()
    SECTIONS_PATH.write_text(json.dumps([{"text": d.page_content, "meta": d.metadata} for d in sections],
                                        ensure_ascii=False), encoding="utf-8")
print("섹션 수:", len(sections))

# %% [markdown]
# ## 2. Split — 검색용 작은 청크 + 답변용 섹션(쪽/표 묶음)
# 검색은 작은 청크(약 700자)로 정확하게 하고, LLM에는 그 청크가 속한 섹션(PDF 한 쪽, HWP 표 묶음) 전체를 넘겨
# 표 머리글·주석 같은 주변 맥락을 잃지 않게 합니다(small-to-big).

# %%
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

chunk_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120, separators=["\n\n", "\n", " ", ""])
section_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=300, separators=["\n\n", "\n", " ", ""])


def label(meta) -> str:
    return f"[{meta['title']} | {meta['loc']}]"


parents = {}  # parent_id -> 답변용 섹션 텍스트 (앞에 근거 ID를 붙여 LLM이 출처로 인용할 수 있게 함)
sources = {}  # 근거 ID(parent_id 앞 6자리) -> (문서·위치 라벨, 섹션 본문)
chunks = []
for sec in sections:
    # 너무 긴 섹션은 3,500자 단위로 나눈다
    for k, part in enumerate(section_splitter.split_text(sec.page_content)):
        meta = dict(sec.metadata)
        pid = hashlib.sha1(f"{meta['source']}|{meta['loc']}|{k}|{part}".encode()).hexdigest()
        parents[pid] = f"[근거 {pid[:6]}] {label(meta)}\n{part}"
        sources[pid[:6]] = (f"{meta['title']} {meta['loc']}", part)
        for piece in chunk_splitter.split_text(part):
            text = f"{label(meta)}\n{piece}"  # 문서명·위치를 청크에 넣어 BM25에도 걸리게 함
            chunks.append(Document(text, metadata={**meta, "parent_id": pid,
                                                   "chunk_id": hashlib.sha1(text.encode()).hexdigest()}))
# 같은 내용 청크 중복 제거
chunks = list({c.metadata["chunk_id"]: c for c in chunks}.values())
print("청크 수:", len(chunks), "/ 섹션 수:", len(parents))

# %% [markdown]
# ## 3. Embed & Store — OpenAI 임베딩 + Chroma (캐시 사용)

# %%
from langchain_chroma import Chroma
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings

base_emb = OpenAIEmbeddings(model="text-embedding-3-large")
embeddings = CacheBackedEmbeddings.from_bytes_store(
    base_emb, LocalFileStore(str(INDEX_DIR / "emb_cache")), namespace=base_emb.model,
    query_embedding_cache=True, key_encoder="sha256",
)
vectorstore = Chroma(collection_name="gachon_2027", embedding_function=embeddings,
                     persist_directory=str(INDEX_DIR / "chroma"), collection_metadata={"hnsw:space": "cosine"})
chunk_ids = [c.metadata["chunk_id"] for c in chunks]
if set(vectorstore.get(include=[])["ids"]) != set(chunk_ids):  # 최초 1회(또는 내용 변경 시)만 색인
    vectorstore.reset_collection()
    for i in range(0, len(chunks), 200):
        vectorstore.add_documents(chunks[i:i + 200], ids=chunk_ids[i:i + 200])
print("저장된 벡터 수:", len(vectorstore.get(include=[])["ids"]))

# %% [markdown]
# ## 4. Retrieve — 벡터 + BM25(Kiwi 형태소) 하이브리드, RRF 병합

# %%
from concurrent.futures import ThreadPoolExecutor

from kiwipiepy import Kiwi
from langchain_community.retrievers import BM25Retriever

kiwi = Kiwi()


def kiwi_tokenize(text: str) -> list[str]:
    return [t.form for t in kiwi.tokenize(text) if not t.tag.startswith(("J", "E", "S")) or t.tag in ("SL", "SN", "SH")]


K = 8
N_SECTIONS = 6   # 1차 검색(질문 그대로)에서 넘기는 섹션 수
N_FOLLOWUP = 3   # Agent 추가 검색에서 넘기는 섹션 수
PER_DOC = 3  # 한 문서가 결과를 독차지하지 않도록 문서당 섹션 수 제한 (문서를 넘나드는 질문 대비)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": K})
bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=kiwi_tokenize, k=K)
_pool = ThreadPoolExecutor(4)


def hybrid_search(query: str, n_sections: int = N_SECTIONS) -> list[str]:
    """벡터·BM25 결과를 RRF로 합친 뒤, 상위 청크가 속한 섹션을 n_sections개 돌려준다."""
    vec = _pool.submit(vector_retriever.invoke, query)
    bm = _pool.submit(bm25_retriever.invoke, query)
    scores, source = {}, {}
    for results in (vec.result(), bm.result()):
        for rank, d in enumerate(results):
            pid = d.metadata["parent_id"]
            scores[pid] = scores.get(pid, 0) + 1 / (60 + rank)
            source[pid] = d.metadata["source"]
    ranked = sorted(scores, key=scores.get, reverse=True)
    top, per_doc = [], {}
    for pid in ranked:
        if per_doc.get(source[pid], 0) < PER_DOC:
            top.append(pid)
            per_doc[source[pid]] = per_doc.get(source[pid], 0) + 1
    top += [pid for pid in ranked if pid not in top]  # 다른 문서 결과가 없으면 남은 칸을 순위대로 채움
    return [parents[pid] for pid in top[:n_sections]]


# %% [markdown]
# ## 5. Agentic RAG — 검색 도구 + Agent
# Agent는 먼저 받은 검색 결과로 답할 수 있는지 판단하고, 근거가 부족하면 `search_documents`로 질의를 바꿔 추가 검색합니다.

# %%
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

AGENT_MODEL = "gpt-5.4-mini"


@tool
def search_documents(query: str) -> str:
    """가천대학교 2027학년도 입학 자료(재외국민 모집요강, 지역의사선발전형 안내, 수시 모집요강, 정시 모집요강,
    수시 면접고사 일정)에서 관련 내용을 검색한다. 전형명·모집단위·항목명 같은 핵심 키워드 위주의 짧은 질의를 넣는다."""
    # Agent의 추가 검색은 이미 1차 검색 결과를 본 뒤의 보충이므로 섹션 수를 줄여 다음 LLM 입력을 가볍게 한다
    return "\n\n---\n\n".join(hybrid_search(query, N_FOLLOWUP))


@tool
def calculate(expression: str) -> str:
    """사칙연산 수식을 계산한다 (예: '30*0.4+20', '(78-16)/78*100'). 숫자에 쉼표·단위는 넣지 않는다."""
    if not re.fullmatch(r"[\d\s.+\-*/()%]+", expression):
        return "오류: 숫자와 + - * / ( ) 만 사용할 수 있습니다."
    return f"{expression} = {eval(expression):,.4f}".rstrip("0").rstrip(".")


SYSTEM_PROMPT = """당신은 가천대학교 입학처 상담 챗봇입니다. 반드시 검색된 문서 내용만 근거로 2027학년도 입학 관련 질문에 답합니다.

문서 구성: 재외국민 모집요강 / 지역의사선발전형 안내(전국 공통 제도·법령 안내) / 수시 모집요강 / 정시 모집요강 / 수시 면접고사 일정.
가천대의 전형별 모집인원·수능최저학력기준·일정 등 가천대 고유 정보는 수시·정시·재외국민 모집요강에 있다.
검색 결과 앞의 [문서명 | 위치] 표시(쪽, 쪽 머리글, 전형명)로 어느 문서·전형의 내용인지 확인한다.
각 검색 결과 맨 앞의 [근거 xxxxxx]는 그 결과의 근거 ID다.

규칙
1. 질문이 어느 모집(수시·정시·재외국민 등)과 전형에 대한 것인지 먼저 파악하고, 그 문서·전형의 내용으로 답한다.
   수시 내용을 정시 질문에, 다른 전형의 값을 이 전형에 쓰지 않는다. 질문에 모집 시기가 없으면 질문 내용에 맞는 문서를 고른다.
2. 받은 검색 결과로 근거가 부족하면 키워드를 바꿔 search_documents로 다시 검색한다. 서로 다른 정보가 필요하면 각각 검색하고,
   독립적인 검색은 한 번에 병렬로 호출한다.
3. 2027학년도 기준으로 답한다. 표에 2026학년도(전년도)와 2027학년도가 나란히 있으면 2027학년도 값을 쓴다.
4. 날짜는 요일·시각·마감 조건(예: "18:00까지", "우체국 소인 유효")까지, 수치는 단위까지 문서 그대로 답한다.
   답에 해당하는 대상·제외 대상·예외 조건이 문서에 있으면 함께 적는다.
   지원 가능 여부·자격 요건을 물으면 결론과 함께, 문서의 해당 요건 항목에 적힌 세부 기준(인정 범위, 합산 여부, 기준일, 제외 대상)을 모두 적는다.
5. 계산(합계, 비율, 점수 환산 등)이 필요하면 암산하지 말고 calculate 도구를 쓴다.
   문서에 점수 산출 수식이 있으면 그 수식을 그대로 쓴다(가중치를 백분율로 바꾸거나 다른 식으로 바꾸지 않는다).
6. 문서에 질문이 묻는 바로 그 내용이 없으면 추측하지 말고 "제공된 문서에서 확인할 수 없습니다."라고 답한다.
   받은 검색 결과에 질문과 관련된 항목 자체가 보이지 않을 때만, 문서 용어로 바꾼 키워드로 한 번 더 검색해 확인한다.
   다른 전형·다른 연도의 값을 대신 제시하거나 일반 상식으로 단정하지 않는다.
   "확인할 수 없습니다"라고 답할 때는 질문과 다른 항목의 금액·날짜·인원 등 구체적인 수치를 덧붙이지 않는다.
7. 일상 표현은 문서 용어로 바꿔 이해하고 검색한다
   (예: 수능 최저 → 수능최저학력기준, 학종 → 학생부종합, 교과 → 학생부교과, 과 옮기기 → 전과, 원서비 → 전형료, 추합 → 충원합격, 지균 → 지역균형).
8. 답변은 한국어로 간결하게(보통 1~4문장) 쓰고, 질문이 묻는 핵심 정보를 빠짐없이 명확히 제시한다. 근거 문서명은 쓰지 않아도 된다.
9. 예/아니오로 답할 질문은 첫 마디에 결론(예/아니요)을 쓰고, 뒤의 설명이 그 결론과 모순되지 않게 한다.
   두 모집·전형을 각각 묻는 질문은 각 값을 해당 문서·전형 표시가 붙은 내용에서 따로 확인하고, 같은 값이라고 가정하지 않는다.
10. 모든 답변에서, 문서에 직접 적혀 있지 않은 사실을 간접 정황(표의 다른 행, 비슷한 모집단위·전형의 기준 등)으로 추론해 단정하지 않는다.
11. 답변의 마지막 줄에는 답을 쓰는 데 실제로 사용한 검색 결과의 근거 ID만 "근거: id1, id2" 형식으로 적는다.
    답을 문서에서 확인할 수 없으면 "근거: 없음"으로 적는다. 본문에는 근거 ID를 쓰지 않는다."""

agent = create_agent(
    # 문항당 최대 대기 20초 안에 끝나도록 LLM 호출 타임아웃·재시도를 짧게 둔다
    model=ChatOpenAI(model=AGENT_MODEL, reasoning={"effort": "low"}, use_responses_api=True, timeout=8, max_retries=1),
    tools=[search_documents, calculate],
    system_prompt=SYSTEM_PROMPT,
)
fallback_llm = ChatOpenAI(model=AGENT_MODEL, reasoning={"effort": "low"}, use_responses_api=True, timeout=8, max_retries=1)

# %% [markdown]
# ## 6. 답변 함수 — `answer(question) -> str`
# 답변 끝에 근거 출처(문서명·쪽)와 원문 줄을 붙입니다. LLM은 사용한 근거 ID만 고르고,
# 출처 표기와 원문 인용은 코드가 실제 검색 섹션에서 가져오므로 문서명·쪽이 틀리게 적히지 않습니다.

# %%
def best_lines(text: str, reply: str, n: int = 2) -> list[str]:
    """섹션에서 답변과 숫자·단어가 가장 많이 겹치는 줄을 골라 원문 인용으로 쓴다."""
    # 한 자리 숫자('일반전형 1'의 1 등)는 어디에나 있어 다른 행까지 끌어오므로 두 자리 이상만 본다
    pattern = r"[0-9]{2,}|[가-힣A-Za-z]{2,}"
    words = set(re.findall(pattern, reply))
    scored = []
    for ln in text.splitlines():
        if " | " in ln and ": " not in ln:
            continue  # 표 머리글 행(열 이름만 있고 값이 없음)은 인용하지 않는다
        toks = re.findall(pattern, ln)
        hits = set(toks) & words
        score = sum((3 if t.isdigit() else 1) for t in hits)
        if score:
            scored.append((score, hits, ln.strip()))
    scored.sort(key=lambda x: -x[0])
    # 두 번째 이후 줄은 가장 잘 맞는 줄과 비슷하게 맞고, 앞 줄에 없던 답 내용을 새로 담을 때만 붙인다
    # (비슷하게 생긴 다른 모집단위 행 인용 방지)
    picked, covered = [], set()
    for s, hits, ln in scored[:n]:
        if not picked or (s >= scored[0][0] * 0.8 and hits - covered):
            picked.append(ln)
            covered |= hits
    # 검색용으로 행 앞에 붙인 '(OO 전형) ' 표시는 사용자에게 보여 줄 때 뗀다 (전형명은 출처 위치에 이미 있음)
    picked = [re.sub(r"^\([^\n]*?전형(?:\s?\d)?\)\s", "", ln) for ln in picked]
    return [ln if len(ln) <= 200 else ln[:200] + "…" for ln in picked]


def with_sources(reply: str) -> str:
    """LLM이 마지막 줄에 적은 근거 ID를 실제 섹션과 대조해, 문서명·위치와 원문 줄을 코드로 붙인다."""
    m = re.search(r"\n?\s*근거\s*[:：]\s*(.*)\s*$", reply)
    if not m:
        return reply
    body = reply[:m.start()].rstrip()
    ids = [i for i in re.findall(r"[0-9a-f]{6}", m.group(1)) if i in sources]  # 실제로 있는 섹션만 인정
    if not ids:
        return body
    lines = ["", "[근거]"]
    for sid in dict.fromkeys(ids):
        loc, text = sources[sid]
        quotes = best_lines(text, body)
        lines.append(f"- {loc}" + (": " + " / ".join(f'"{q}"' for q in quotes) if quotes else ""))
    return body + "\n" + "\n".join(lines)


def answer(question: str) -> str:
    # 질문으로 1차 검색한 결과를 함께 넘겨 LLM 왕복을 줄인다. 근거가 부족하면 Agent가 직접 추가 검색한다.
    prefetched = "\n\n---\n\n".join(hybrid_search(question))
    content = (f"질문: {question}\n\n[질문으로 미리 검색한 결과]\n{prefetched}\n\n"
               "위 결과만으로 답할 근거가 충분하면 바로 답하고, 부족하거나 다른 곳의 정보가 더 필요하면 search_documents로 추가 검색하세요.")
    try:
        # recursion_limit: 검색 반복이 길어지지 않도록 Agent 단계 수 상한 (LLM 호출 최대 약 5회)
        result = agent.invoke({"messages": [{"role": "user", "content": content}]}, config={"recursion_limit": 10})
        reply = result["messages"][-1].text
    except Exception:
        # Agent가 단계 상한·오류로 끝나면 미리 검색한 결과만으로 바로 답한다 (빈 답·예외 방지)
        reply = fallback_llm.invoke([("system", SYSTEM_PROMPT), ("user", f"질문: {question}\n\n[검색 결과]\n{prefetched}")]).text
    return with_sources(reply)


# %% [markdown]
# ## 7. 실행 예시

# %%
if __name__ == "__main__":
    import time
    for q in ["수시 원서접수 기간은 언제인가요?",
              "정시 합격자 등록 기간을 알려주세요.",
              "재외국민 전형 면접고사일은 언제인가요?"]:
        t0 = time.time()
        print(f"Q: {q}\nA: {answer(q)}  ({time.time() - t0:.1f}s)\n")
