"""コマごとの単純な質問(はい/いいえ)と、ルールで失敗の種類を決める診断。

diagnose_failures.py の改良版。変えた点:
  1. 作業領域を切り出して拡大した画像を使う
  2. 1コマずつ、4つの はい/いいえ だけを答えさせる
  3. 失敗の種類は、コマごとの答えからルールで決める (derive)
成功した動画にも同じ質問をして、「運べた」を検出できるか確かめられる。

使い方 (Colab, diagnose_failures.py と同じ場所に置く):
    !python diagnose_frames.py --videos /content/drive/MyDrive/\\
smolvla_task/videos --pattern "r1_spatial_nas10_libero_object_*.mp4" \\
--out /content/diag_frames_r1.json
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from diagnose_failures import (HFBackend, object_of,
                               parse_name)

N = 12
SYSTEM = """You are looking at ONE frame from a robot simulation. A robot
arm with a two-finger gripper must pick up a target object and put it in
the woven basket on the left. Choose the ONE option that best describes
the target object in this frame. Reply with a single letter only.
A) on the table, not touched by the gripper
B) held by the gripper (lifted into the air)
C) in the basket, or held directly above the basket
D) pushed, tipped over or moved, but not held
E) cannot tell"""
LETTERS = "ABCDE"


def crop_frames(video: Path, n: int = N) -> list[bytes]:
    """作業領域 (画像の上下を少し削る) を切り出し、拡大した PNG を返す。"""
    total = int(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True).stdout.strip())
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n):
            k = round((total - 1) * i / (n - 1))
            png = Path(tmp) / f"{i}.png"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vf",
                 f"select=eq(n\\,{k}),crop=256:176:0:30,"
                 f"scale=640:440:flags=lanczos", "-frames:v", "1",
                 str(png)], check=True)
            out.append(png.read_bytes())
    return out


def derive(states: list[str]) -> str:
    """コマごとの選択 (時間順) から、結果の種類を決める。"""
    c = {k: states.count(k) for k in LETTERS}
    if c["B"] >= 1 and c["C"] >= 1:
        return "reached_basket"
    if c["B"] >= 2:
        late = states[-3:]
        return "carry_incomplete" if "B" in late else "grasp_dropped"
    if c["D"] >= 2:
        return "touch_no_grasp"
    return "no_approach"


def run_video(bk, video: Path) -> dict:
    meta = parse_name(video)
    obj = object_of(meta["tid"])
    states = []
    for png in crop_frames(video):
        text = bk.ask(SYSTEM, [("text", f"Target object: {obj}."),
                               ("image", png)], 3).strip().upper()
        states.append(text[:1] if text[:1] in LETTERS else "E")
    return {**meta, "object": obj, "states": "".join(states),
            "derived": derive(states)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--pattern", default="*.mp4")
    ap.add_argument("--out", default="diag_frames.json")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()

    vids = sorted(Path(args.videos).glob(args.pattern))
    if not vids:
        raise SystemExit(f"動画が見つかりません: {args.pattern}")
    out = Path(args.out)
    done = json.loads(out.read_text()) if out.is_file() else {"rows": []}
    seen = {r["file"] for r in done["rows"]}
    print(f"対象 {len(vids)} 本 (診断済み {len(seen)} 本)")
    bk = HFBackend(args.model)
    for v in vids:
        if v.name in seen:
            continue
        row = run_video(bk, v)
        done["rows"].append(row)
        out.write_text(json.dumps(done, ensure_ascii=False, indent=1))
        print(row["res"], row["object"], row["tid"], row["states"], "->",
              row["derived"], flush=True)

    rows = done["rows"]
    ok = [r for r in rows if r["res"] == "OK"]
    ng = [r for r in rows if r["res"] == "NG"]
    hit = sum(r["derived"] == "reached_basket" for r in ok)
    fa = sum(r["derived"] == "reached_basket" for r in ng)
    print(f"\n成功 {len(ok)} 本のうち 運べたと検出: {hit}")
    print(f"失敗 {len(ng)} 本のうち 運べたと誤検出: {fa}")



if __name__ == "__main__":
    main()
