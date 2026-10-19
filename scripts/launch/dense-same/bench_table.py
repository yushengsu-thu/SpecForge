"""Markdown tables of the acceptance-length suites under /scratch/dense/bench/results/<label>/<mode>_t<temp>/summary.json,
next to the values published in jiapingW/Qwen3-8b-dspark-SpecForge (README / bench/results.json)."""
import argparse, json, os

PUBLISHED = {  # request macro tau, from jiapingW's README
    ("0", "V16"): {"gsm8k": 9.720633, "math500": 9.375822, "humaneval": 7.481950, "mbpp": 6.988848, "mt-bench": 5.159163, "pooled": 8.899194},
    ("0", "V8"): {"gsm8k": 6.452161, "math500": 6.356439, "humaneval": 5.608338, "mbpp": 5.399994, "mt-bench": 4.232717, "pooled": 6.113925},
    ("1", "V16"): {"gsm8k": 8.984335, "math500": 8.186249, "humaneval": 6.683424, "mbpp": 6.427295, "mt-bench": 4.657792, "pooled": 8.098585},
    ("1", "V8"): {"gsm8k": 6.234437, "math500": 5.858557, "humaneval": 5.345947, "mbpp": 5.141272, "mt-bench": 3.972278, "pooled": 5.827545},
}
ROWS = ["gsm8k", "math500", "humaneval", "mbpp", "mt-bench", "pooled"]

p = argparse.ArgumentParser(); p.add_argument("--results", required=True); a = p.parse_args()
labels = sorted(d for d in os.listdir(a.results) if os.path.isdir(os.path.join(a.results, d)) and not d.startswith("smoke"))
for temp in ("0", "1"):
    for mode in ("V16", "V8"):
        cols, data = [], {}
        for lab in labels:
            f = os.path.join(a.results, lab, f"{mode}_t{temp}", "summary.json")
            if os.path.exists(f):
                s = json.load(open(f)); cols.append(lab)
                data[lab] = {**{k: v["request_macro_tau"] for k, v in s["datasets"].items()}, "pooled": s["pooled"]["request_macro_tau"]}
        if not cols:
            continue
        print(f"\n### temperature {temp}, D16/{mode} (request macro tau)\n")
        print("| benchmark | " + " | ".join(cols) + " | jiapingW published |")
        print("|---|" + "---|" * (len(cols) + 1))
        for r in ROWS:
            vals = [f"{data[c].get(r):.3f}" if data[c].get(r) is not None else "-" for c in cols]
            print(f"| {r} | " + " | ".join(vals) + f" | {PUBLISHED[(temp, mode)][r]:.3f} |")
