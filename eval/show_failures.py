import json
import sys
from pathlib import Path

path = sys.argv[1]                       # e.g. eval/results_effort_low_r1.json
mode = sys.argv[2] if len(sys.argv) > 2 else "vector"

lines = Path("eval/dataset.jsonl").read_text(encoding="utf-8").splitlines()
ref = {json.loads(l)["question"]: json.loads(l) for l in lines if l.strip()}
rows = json.loads(Path(path).read_text(encoding="utf-8"))[mode]

for r in rows:
    if r["correct"] and r["faithful"]:
        continue
    item = ref[r["question"]]
    print("=" * 80)
    print("Q:", r["question"])
    print(f"type={r['type']} correct={r['correct']} faithful={r['faithful']} rank={r['rank']}")
    print("REFERENCE:", item["reference_answer"] or "(unanswerable)")
    print("ANSWER:", r["answer"][:600])