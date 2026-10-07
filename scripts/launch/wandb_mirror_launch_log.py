#!/usr/bin/env python3
"""Mirror SpecForge launch.log step lines into W&B (live). Phase A first, then phase B when its log appears."""
import os, re, sys, time, json, subprocess
import yaml, wandb

ENTITY = "miles_training"
PROJECT = "specforge-2 layer moe"
RUNS = [
    dict(
        run_id="qwen3.8-27b-dspark-moe-2layer-regen-mixture-v1-1ep-h200",
        recipe="/scratch/qwen38-moe-2layer-1ep-h200.yaml",
        total=4958,
        phase="A",
    ),
    dict(
        run_id="qwen3.8-27b-dspark-moe-2layer-regen-mixture-v1-cont2ep-from-1ep-h200",
        recipe="/scratch/qwen38-moe-2layer-cont2ep-h200.yaml",
        total=9916,
        phase="B",
    ),
]
STOP = "/scratch/wandb_live.stop"
STATE_DIR = "/scratch/wandb_live_state"
os.makedirs(STATE_DIR, exist_ok=True)
POLL = 300
IDLE_FINISH_S = 1800
SAFE = {
    "__builtins__": {},
    "nan": float("nan"),
    "inf": float("inf"),
    "True": True,
    "False": False,
    "None": None,
}


def log(msg):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg, flush=True)


def parse_new_steps(path, after):
    out = {}
    with open(path, errors="replace") as f:
        for line in f:
            if not line.startswith("step "):
                continue
            m = re.match(r"step (\d+): (\{.*\})\s*$", line)
            if not m:
                continue
            s = int(m.group(1))
            if s <= after or s in out:
                continue
            try:
                d = eval(m.group(2), SAFE)
            except Exception:
                continue
            out[s] = {
                k: v
                for k, v in d.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
    return out


def trainer_alive(recipe):
    r = subprocess.run(
        ["pgrep", "-f", "specforge.cli train.*" + os.path.basename(recipe)],
        capture_output=True,
        text=True,
    )
    return bool(r.stdout.strip())


def run_config(r):
    cfg = {
        "phase": r["phase"],
        "total_steps": r["total"],
        "hardware": "8x H200 141GB, devbox specforge-qwen-h200",
        "target": "Qwen/Qwen3.8-27B bf16 (the NVFP4 build needs Blackwell)",
        "code": "SpecForge kan/moe-3-qwen38 @ abf003e9",
        "base_recipe": "Kan -1ep-v3 (phase A) / -cont2ep-from-1ep-v3 (phase B); only change: draft num_hidden_layers 5 -> 2",
    }
    try:
        y = yaml.safe_load(open(r["recipe"]))
        cfg["recipe"] = y
        cfg["draft_config"] = json.load(open(y["model"]["draft_model_config"]))
    except Exception as e:
        cfg["recipe_error"] = str(e)
    return cfg


def mirror(r):
    run_id = r["run_id"]
    phase = r["phase"]
    logp = f"/scratch/outputs/{run_id}/launch.log"
    state_p = os.path.join(STATE_DIR, run_id + ".json")
    state = json.load(open(state_p)) if os.path.exists(state_p) else {"last": 0}
    while not os.path.exists(logp):
        if os.path.exists(STOP):
            return
        log(f"[{phase}] waiting for {logp}")
        time.sleep(POLL)
    wid = re.sub(r"[^A-Za-z0-9_\-]", "-", run_id)
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=wid,
        name=run_id,
        resume="allow",
        config=run_config(r),
        tags=["qwen3.8-27b", "dspark", "moe", "2layer", "phase-" + phase, "h200"],
        settings=wandb.Settings(console="off"),
    )
    log(f"[{phase}] wandb run {run.url} (resuming after step {state['last']})")
    idle_since = None
    while True:
        new = parse_new_steps(logp, state["last"])
        for s in sorted(new):
            run.log(new[s], step=s)
            state["last"] = s
        if new:
            json.dump(state, open(state_p, "w"))
            idle_since = None
            log(f"[{phase}] logged {len(new)} steps, now at {state['last']}")
        done = state["last"] >= r["total"]
        alive = trainer_alive(r["recipe"])
        if not alive and not new:
            idle_since = idle_since or time.time()
        if (
            done
            or (not alive and idle_since and time.time() - idle_since > IDLE_FINISH_S)
            or os.path.exists(STOP)
        ):
            run.summary["final_step"] = state["last"]
            run.finish(exit_code=0 if done else 1)
            log(f"[{phase}] finished at step {state['last']} (done={done})")
            return
        time.sleep(POLL)


for r in RUNS:
    if os.path.exists(STOP):
        break
    mirror(r)
log("all runs mirrored; exiting")
