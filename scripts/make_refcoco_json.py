"""Convert HuggingFace RefCOCO to the JSON format eval_refcoco.py expects."""
import json, re
from datasets import load_dataset

ds = load_dataset("lmms-lab/RefCOCO", split="val")

samples = []
for item in ds:
    fname = item["file_name"]
    m = re.search(r"(\d{12})", fname)
    if not m:
        continue
    image_id = int(m.group(1))
    samples.append({
        "image_id": image_id,
        "ann_id": int(item["question_id"]),
        "bbox": item["bbox"],
        "sentences": item["answer"],
        "split": "val",
    })

out = "data/refcoco/refcoco_val.json"
with open(out, "w") as f:
    json.dump(samples, f)
print(f"Saved {len(samples)} samples to {out}")
