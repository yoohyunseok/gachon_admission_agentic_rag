"""검증 세트(validation/*.csv)로 실행용 노트북(gachon_admission_agentic_rag.ipynb)을 채점한다.
노트북 코드 셀을 위에서부터 실행(!·% 줄과 __main__ 블록 제외)한 뒤 answer()를 호출하고, LLM 심판이 판정한다.
문항당 trials회 호출, 과반 정답이면 정답, 평균 응답시간 측정.

  python work/validate.py --trials 1 --workers 6          # 빠른 정확도 점검
  python work/validate.py                                 # 3회·순차 (시간 측정)
  python work/validate.py --style casual --split tune     # 불친절 질문만
  python work/validate.py --ids 12,31
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
WORK = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "gachon_admission_agentic_rag.ipynb"
load_dotenv(ROOT / ".env")
JUDGE_MODEL = os.getenv("JUDGE_MODEL") or "gpt-5.4-mini"

JUDGE_PROMPT = """당신은 RAG 시스템 시험의 채점관입니다. [질문]에 대한 [응시자 답변]이 [정답]과 일치하는지 판정하세요.

판정 기준
1. 예상 정답과 표현이 달라도 핵심 정보와 질문에서 요구한 조건이 일치하면 정답.
2. 부분점수는 없다. 질문에서 요구한 핵심 정보(수치·날짜·대상·예외 조건 등 답변에 필요한 정보)와 조건을 모두 충족해야 정답.
3. 표기 차이는 허용: 1월 4일 = 01.04 / 18:00 = 오후 6시 / 30% = 30퍼센트.
4. 정답에 없는 부가 설명이 있어도 정답과 모순되지 않으면 허용. 단, 여러 후보 값을 나열해 얼버무리거나 핵심 정보가 틀리면 오답.
5. 정답이 '문서에서 확인할 수 없음'인 경우: 응시자가 확인할 수 없다/근거가 없다고 밝히면 정답. 구체적 값을 지어내면 오답.
6. 정답의 괄호 안 내용은 보충 설명이다. 괄호 밖 핵심을 맞히면 정답으로 본다.
7. 정답이 구체적인 값인데 응시자가 '확인할 수 없다'고 답하면 오답.

[질문]
{question}

[정답]
{answer}

[응시자 답변]
{prediction}

JSON으로만 답하세요: {{"correct": true 또는 false, "reason": "한 문장 근거"}}"""


def load_answer_fn():
    os.chdir(WORK)
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ns = {"__name__": "rag"}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            src = "\n".join(ln for ln in "".join(cell["source"]).splitlines() if not ln.lstrip().startswith(("!", "%")))
            exec(compile(src, f"{NOTEBOOK.name}[cell {i}]", "exec"), ns)
    return ns


def retrieval_scores(ns, item):
    """검색 단계 지표: gold 근거('03:46;01:전형일정')가 1차 검색 섹션에 들어 있는지(recall), 섹션 중 근거 비율(precision)."""
    gold = item.get("gold", "").strip()
    if not gold or gold == "none":
        return None
    secs = ns["hybrid_search"](item["question"])
    titles = {f[:2]: ns["doc_title"](f) for f in ns["DOC_FILES"]}

    def match(sec, g):
        num, _, where = g.partition(":")
        head = sec.split("\n", 1)[0]
        if titles.get(num, "?") not in head:
            return False
        if num == "01":
            return where in sec
        return re.search(rf"\| {re.escape(where)}쪽", head) is not None

    golds = [g.strip() for g in gold.split(";") if g.strip()]
    found = [any(match(s, g) for s in secs) for g in golds]
    useful = [any(match(s, g) for g in golds) for s in secs]
    return sum(found) / len(golds), sum(useful) / len(secs)


def judge(client, item, prediction):
    if not prediction or not prediction.strip():
        return False, "빈 답변"
    msg = JUDGE_PROMPT.format(question=item["question"], answer=item["answer"], prediction=prediction)
    err = None
    for attempt in range(3):
        try:
            res = client.chat.completions.create(model=JUDGE_MODEL, messages=[{"role": "user", "content": msg}],
                                                 response_format={"type": "json_object"})
            out = json.loads(res.choices[0].message.content)
            return bool(out.get("correct")), out.get("reason", "")
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 * (attempt + 1))
    return False, f"채점 오류: {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "validation" / "val_pairs.csv"))
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--style", default=None)
    ap.add_argument("--out", default=None, help="결과 JSON 저장 경로")
    args = ap.parse_args()

    with open(args.file, encoding="utf-8-sig") as f:
        items = list(csv.DictReader(f))
    if args.split:
        items = [it for it in items if it["split"] == args.split]
    if args.style:
        items = [it for it in items if it["style"] == args.style]
    if args.ids:
        keep = set(args.ids.split(","))
        items = [it for it in items if it["id"] in keep]

    t_load = time.time()
    ns = load_answer_fn()
    answer_fn = ns["answer"]
    print(f"[validate] 노트북 로딩 {time.time() - t_load:.1f}s, 문항 {len(items)}개, judge={JUDGE_MODEL}")
    client = OpenAI()
    need = args.trials // 2 + 1

    def run(it):
        trials = []
        for _ in range(args.trials):
            t0 = time.time()
            try:
                pred = str(answer_fn(it["question"]))
            except Exception as e:  # noqa: BLE001
                pred = f"[예외] {e}"
            dt = time.time() - t0
            ok, reason = judge(client, it, pred)
            trials.append((ok, dt, pred, reason))
        it["_ret"] = retrieval_scores(ns, it)
        return it, trials

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(run, items))

    n_ok, times, by = 0, [], {}
    for it, trials in results:
        n = sum(ok for ok, *_ in trials)
        ok = n >= need
        n_ok += ok
        avg = sum(dt for _, dt, *_ in trials) / len(trials)
        times.append(avg)
        for key in (f"type:{it['type']}", f"style:{it['style']}") + ((f"metric:{it['metric']}",) if it.get("metric") else ()):
            s = by.setdefault(key, [0, 0])
            s[0] += ok
            s[1] += 1
        ret = it["_ret"]
        if not ok or n < len(trials) or avg > 5:
            r = f" 검색 recall={ret[0]:.2f} precision={ret[1]:.2f}" if ret else ""
            print(f"[{it['id']}] {'O' if ok else 'X'} ({n}/{len(trials)}) {avg:.1f}s {it['style']}/{it['type']}{r}  {it['question']}")
            print(f"      정답: {it['answer']}   (근거: {it['source']})")
            for i, (tok, dt, pred, reason) in enumerate(trials, 1):
                print(f"      답{i}({'o' if tok else 'x'}, {dt:.1f}s): {pred.strip()[:300]}")
                if not tok:
                    print(f"      사유{i}: {reason}")
    print("=" * 60)
    print(f"정확도: {n_ok}/{len(results)} ({n_ok / len(results) * 100:.1f}%)  평균 {sum(times) / len(times):.2f}s  "
          f"최대 {max(times):.1f}s  5초 초과 {sum(t > 5 for t in times)}개")
    rets = [it["_ret"] for it, _ in results if it["_ret"]]
    if rets:
        print(f"검색 단계(1차 검색 상위 섹션, gold 있는 {len(rets)}문항): 평균 recall {sum(r[0] for r in rets) / len(rets):.2f}  "
              f"전부 찾음 {sum(r[0] == 1 for r in rets)}/{len(rets)}  평균 precision {sum(r[1] for r in rets) / len(rets):.2f}")
    for k, (a, b) in sorted(by.items()):
        print(f"  - {k:22s} {a}/{b}")
    if args.out:
        Path(args.out).write_text(json.dumps([{"id": it["id"], "trials": t} for it, t in results], ensure_ascii=False, indent=1),
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
