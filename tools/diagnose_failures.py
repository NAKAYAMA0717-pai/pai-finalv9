"""失敗した rollout 動画を VLM に見せて、失敗の種類をJSONで診断する。

VLM は2通りから選べる (--backend):
  hf         Colab の GPU 上で、公開されているオープンな VLM を動かす
             (既定。Qwen2.5-VL。課金・API キー不要)
  anthropic  Anthropic API (環境変数 ANTHROPIC_API_KEY が必要)

使い方 (Colab):
    !python tools/diagnose_failures.py --videos /content/drive/MyDrive/\
smolvla_task/videos --pattern "r3_object_nas10_libero_object_*_NG.mp4" \
--out diagnoses_r3.json
"""
import argparse
import base64
import bisect
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

MODEL = os.environ.get("DIAG_MODEL", "claude-sonnet-5-5")
N_FRAMES = 10
TYPES = ["wrong_target", "no_approach", "touch_no_grasp",
         "grasp_dropped", "carry_incomplete", "other"]
# LIBERO-plus Object: 物体ごとの最初の task_id (ノートブック 10.0 で求めた値)
OBJ_FIRST = [0, 32, 79, 100, 122, 127, 138, 170, 210, 235]
OBJ_NAMES = ["alphabet soup", "bbq sauce", "butter", "chocolate pudding",
             "cream cheese", "ketchup", "milk", "orange juice",
             "salad dressing", "tomato sauce"]

SYSTEM = """You are inspecting a failed robot-manipulation rollout in a
simulator. The robot arm must pick up ONE target object and put it in the
basket (the woven basket on the left). You get frames in time order.
Judge only from what is visible. First write short per-frame notes, then
decide. Reply with a single JSON object and nothing else:
{"notes": ["frame 1: ...", "frame 2: ...", ...],
 "failure_type": one of %s,
 "target_visible": true/false,
 "contacted_target": true/false,
 "grasped": true/false,
 "summary_ja": "1 sentence in Japanese describing what the arm did",
 "confidence": number between 0 and 1}
Definitions:
 wrong_target     the gripper goes to, or pushes, a DIFFERENT object
 no_approach      the gripper hovers or wanders and never reaches the target
 touch_no_grasp   the gripper reaches the target but never holds it
 grasp_dropped    the target is held, then falls somewhere other than basket
 carry_incomplete the target is held and moved toward the basket, but is
                  not released into it before time runs out
 other            none of these
Set "grasped" true only if the target is visibly lifted with the gripper.
If the frames are too small or ambiguous, lower the confidence.""" % TYPES

PROPOSE_SYSTEM = """You help improve a robot policy (SmolVLA fine-tuned with
LoRA). You get, per object, the number of test rollouts, how many failed, the
failure rate, the diagnosed failure types (from a small vision model that is
often wrong, so trust fail_rate more than the types), and how many
demonstration episodes each object already has in the training set.
Propose where to add demonstrations: more for objects with a higher
failure rate. Reply with one
JSON object only:
{"add_demos": {"<object name>": <integer number of extra episodes>},
 "other_changes": ["short suggestions, e.g. inference or training settings"],
 "rationale_ja": "2-3 sentences in Japanese"}
Use the object names exactly as they appear in failures_by_object.
Only add demos for objects that failed. Keep the total extra episodes <= 60.
Example: {"add_demos": {"ketchup": 20, "milk": 10}, "other_changes": [],
"rationale_ja": "..."}"""


def object_of(task_id: int) -> str:
    return OBJ_NAMES[bisect.bisect_right(OBJ_FIRST, task_id) - 1]


def parse_name(path: Path) -> dict:
    m = re.match(r"(?P<label>.+?)_(?P<nas>default|nas\d+)_"
                 r"(?P<suite>libero_\w+?)_t(?P<tid>\d+)_(?P<res>OK|NG)$",
                 path.stem)
    if not m:
        raise ValueError(f"unexpected file name: {path.name}")
    d = m.groupdict()
    d["tid"] = int(d["tid"])
    d["file"] = path.name
    return d


def extract_frames(video: Path, n: int = N_FRAMES) -> list[bytes]:
    """動画から等間隔に n コマを PNG で取り出す (ffmpeg)。"""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=nb_frames", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True).stdout.strip()
    total = int(out)
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n):
            k = round((total - 1) * i / (n - 1))
            png = Path(tmp) / f"{i}.png"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vf",
                 f"select=eq(n\\,{k}),scale=448:448", "-frames:v", "1",
                 str(png)], check=True)
            frames.append(png.read_bytes())
    return frames


class AnthropicBackend:
    def __init__(self, model):
        import anthropic  # キーは環境変数 ANTHROPIC_API_KEY から読まれる
        self.client, self.model = anthropic.Anthropic(), model

    def ask(self, system, parts, max_tokens=700):
        content = []
        for kind, val in parts:
            if kind == "text":
                content.append({"type": "text", "text": val})
            else:
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": base64.b64encode(val).decode()}})
        r = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": content}])
        return "".join(b.text for b in r.content
                       if getattr(b, "text", None))


class HFBackend:
    """Colab の GPU 上で Qwen2.5-VL を動かす (無料・オフライン推論)。"""

    def __init__(self, model):
        import torch
        from transformers import (AutoProcessor,
                                  Qwen2_5_VLForConditionalGeneration)
        self.proc = AutoProcessor.from_pretrained(model)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, torch_dtype=torch.bfloat16, device_map="cuda")

    def ask(self, system, parts, max_tokens=500):
        import io
        from PIL import Image
        content, images = [], []
        for kind, val in parts:
            if kind == "text":
                content.append({"type": "text", "text": val})
            else:
                content.append({"type": "image"})
                images.append(Image.open(io.BytesIO(val)).convert("RGB"))
        msgs = [{"role": "system",
                 "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content}]
        text = self.proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=images or None,
                           return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                  do_sample=False)
        out = out[:, inputs.input_ids.shape[1]:]
        return self.proc.batch_decode(out, skip_special_tokens=True)[0]


def extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON in reply: " + text[:200])
    return json.loads(m.group(0))


def ask(bk, system: str, parts: list, max_tokens: int = 500) -> dict:
    """JSON が読めなければ1回だけ聞き直す。"""
    for _ in range(2):
        text = bk.ask(system, parts, max_tokens)
        try:
            return extract_json(text)
        except (ValueError, json.JSONDecodeError):
            continue
    return {"parse_error": text[:200]}


def diagnose(bk, video: Path) -> dict:
    meta = parse_name(video)
    obj = object_of(meta["tid"])
    parts = [("text",
              f"Target object: {obj}. Task: pick up the {obj} and place it"
              f" in the basket. The rollout FAILED (timed out). "
              f"{N_FRAMES} frames follow, first to last.")]
    for i, png in enumerate(extract_frames(video), start=1):
        parts.append(("text", f"frame {i}/{N_FRAMES}"))
        parts.append(("image", png))
    res = ask(bk, SYSTEM, parts)
    if res.get("failure_type") not in TYPES:
        res["failure_type"] = "other"
    return {**meta, "object": obj, **res}


def rollout_counts(videos: Path, rows: list[dict]) -> dict:
    """診断した動画と同じ (label, nas, suite) の、全 rollout 数を数える。"""
    keys = {(r["label"], r["nas"], r["suite"]) for r in rows}
    total, fail = {}, {}
    for label, nas, suite in keys:
        for f in videos.glob(f"{label}_{nas}_{suite}_t*_*.mp4"):
            m = parse_name(f)
            o = object_of(m["tid"])
            total[o] = total.get(o, 0) + 1
            fail[o] = fail.get(o, 0) + (m["res"] == "NG")
    return {o: {"rollouts": total[o], "failed": fail[o]} for o in total}


def propose(bk, rows: list[dict], train_counts: dict,
            counts: dict) -> dict:
    table = {}
    for o, c in counts.items():
        types = {}
        for r in rows:
            if r["object"] == o:
                types[r["failure_type"]] = types.get(r["failure_type"], 0) + 1
        table[o] = {**c, "fail_rate": round(c["failed"] / c["rollouts"], 2),
                    "diagnosed_types": types}
    text = json.dumps({"results_by_object": table,
                       "train_episodes_per_object": train_counts},
                      ensure_ascii=False)
    res = ask(bk, PROPOSE_SYSTEM, [("text", text)], 700)
    # 制約を守らせる: 失敗した物体だけ、合計 60 本まで
    failed = {o for o, c in table.items() if c["failed"] > 0}
    add = {k: int(v) for k, v in res.get("add_demos", {}).items()
           if k in failed and str(v).lstrip("-").isdigit() and int(v) > 0}
    total = sum(add.values())
    if total > 60:
        add = {k: int(v * 60 / total) for k, v in add.items()}
    res["add_demos"] = add
    res["input"] = table
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--pattern", default="*_NG.mp4")
    ap.add_argument("--out", default="diagnoses.json")
    ap.add_argument("--backend", choices=["hf", "anthropic"], default="hf")
    ap.add_argument("--model", default="")
    ap.add_argument("--train-counts", default="",
                    help='JSON: {"ketchup": 5, ...} 学習データの本数')
    args = ap.parse_args()

    out = Path(args.out)
    done = json.loads(out.read_text()) if out.is_file() else {"rows": []}
    seen = {r["file"] for r in done["rows"]}
    vids = sorted(Path(args.videos).glob(args.pattern))
    if not vids:
        raise SystemExit(f"動画が見つかりません: {args.videos} / "
                         f"{args.pattern}")
    print(f"対象 {len(vids)} 本 (診断済み {len(seen)} 本)")
    if args.backend == "hf":
        bk = HFBackend(args.model or "Qwen/Qwen2.5-VL-7B-Instruct")
    else:
        bk = AnthropicBackend(args.model or MODEL)
    for v in vids:
        if v.name in seen:
            continue
        row = diagnose(bk, v)
        done["rows"].append(row)
        out.write_text(json.dumps(done, ensure_ascii=False, indent=1))
        print(row["object"], row["tid"], row["failure_type"],
              row.get("summary_ja", row.get("parse_error")))

    if not done["rows"]:
        raise SystemExit("診断結果が空のため、提案は作りません")
    counts = json.loads(args.train_counts) if args.train_counts else {}
    done["proposal"] = propose(bk, done["rows"], counts,
                               rollout_counts(Path(args.videos), done["rows"]))
    out.write_text(json.dumps(done, ensure_ascii=False, indent=1))
    print(json.dumps(done["proposal"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
