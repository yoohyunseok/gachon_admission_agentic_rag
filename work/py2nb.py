"""rag.py(# %% 셀 구분)를 실행용 노트북으로 변환한다.  python work/py2nb.py [출력경로]
출력경로를 생략하면 저장소 루트의 gachon_admission_agentic_rag.ipynb에 쓴다."""
import json
import re
import sys
from pathlib import Path

src = (Path(__file__).parent / "rag.py").read_text(encoding="utf-8")
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "gachon_admission_agentic_rag.ipynb"
cells = []
for block in re.split(r"^# %%", src, flags=re.M)[1:]:
    head, _, body = block.partition("\n")
    body = body.strip("\n")
    if head.strip() == "[markdown]":
        text = "\n".join(re.sub(r"^# ?", "", ln) for ln in body.splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": body.splitlines(keepends=True)})
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", out, len(cells), "cells")
