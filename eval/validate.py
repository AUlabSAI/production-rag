import json
from pathlib import Path

REQUIRED = {"question", "reference_answer", "source", "pages", "answerable"}
ok = True

for n, line in enumerate(
    Path("eval/dataset.jsonl").read_text(encoding="utf-8").splitlines(), 1
):
    if not line.strip():
        continue
    try:
        item = json.loads(line)
    except json.JSONDecodeError as e:
        print(f"line {n}: invalid JSON ({e.msg} at col {e.colno})")
        ok = False
        continue
    missing = REQUIRED - item.keys()
    if missing:
        print(f"line {n}: missing keys {missing}")
        ok = False
    elif item["answerable"] and (not item["source"] or not item["pages"]):
        print(f"line {n}: answerable question needs source and pages")
        ok = False

print("dataset OK" if ok else "fix the lines above")