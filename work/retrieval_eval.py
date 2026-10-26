"""검색 단계만 평가 (LLM 호출 없음): gold 근거가 상위 N개 섹션에 들어오는 비율.
  python work/retrieval_eval.py validation/val_pairs_v2.csv [N ...]
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import validate  # noqa: E402

path = sys.argv[1]
ns_list = [int(x) for x in sys.argv[2:]] or [4, 6, 8]
items = [r for r in csv.DictReader(open(Path(path).resolve(), encoding="utf-8-sig")) if r.get("gold", "none") != "none"]
ns = validate.load_answer_fn()
for n in ns_list:
    ns["N_SECTIONS"] = n
    orig = ns["hybrid_search"]
    full, recall, misses = 0, 0.0, []
    for it in items:
        r = validate.retrieval_scores({**ns, "hybrid_search": lambda q: orig(q, n)}, it)
        recall += r[0]
        full += r[0] == 1
        if r[0] < 1:
            misses.append(f"  {it['id']} ({it['style']}) gold={it['gold']}  {it['question']}")
    print(f"N={n}: 평균 recall {recall / len(items):.2f}, 전부 찾음 {full}/{len(items)}")
    if n == ns_list[-1]:
        print("\n".join(misses))
