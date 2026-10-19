"""見直し間隔の設定を並べて、成功率と推論の負担を比べる (Colab 用)。

使い方 (ノートブックの環境で、変数を渡して呼ぶ):
    import sys; sys.path.insert(0, "/content")        # tools を置いた場所
    import replan_sweep as rs
    ids = rs.object_variant_ids(OBJECT_TASK_NAMES, per_obj=3)
    rows = rs.run_sweep(MERGED_MODEL_DIR, rs.SETTINGS, ids, ...)
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

# 比べる設定: name, モード, n_action_steps (chunk_size=50 以下), K_OPEN, K_HOLD
SETTINGS = [
    dict(name="fixed50", mode="fixed", nas=50),
    dict(name="fixed25", mode="fixed", nas=25),
    dict(name="fixed10", mode="fixed", nas=10),
    dict(name="fixed5", mode="fixed", nas=5),
    dict(name="adaptive_10_25", mode="grip", nas=50, k_open=10, k_hold=25),
]


def object_variant_ids(names, per_obj=3):
    """各物体の指示文に一致する task_id のうち、等間隔に per_obj 個を選ぶ。"""
    from libero.libero import benchmark
    with contextlib.redirect_stdout(io.StringIO()):
        b = benchmark.get_benchmark_dict()["libero_object"]()
    ids = {n: [] for n in names}
    for i in range(b.get_num_tasks()):
        lang = b.get_task(i).language.lower()
        for n in names:
            if lang.startswith(n):
                ids[n].append(i)
    out = {}
    for n, xs in ids.items():
        pos = [round(j * (len(xs) - 1) / (per_obj - 1))
               for j in range(per_obj)]
        out[n] = [xs[p] for p in dict.fromkeys(pos)]
    return out


def run_one(model_dir, st, task_ids, out_root, tools_dir, lerobot_dir,
            lerobot_src, libero_plus_dir, camera_mapping, seed,
            episode_length=None):
    out = Path(out_root) / st["name"]
    shutil.rmtree(out, ignore_errors=True)
    stats = out.parent / f"{st['name']}_stats.json"
    env = os.environ.copy()
    env.update(MUJOCO_GL="egl", TQDM_DISABLE="1", PYTHONWARNINGS="ignore",
               REPLAN_MODE=st["mode"], STATS_JSON=str(stats),
               K_OPEN=str(st.get("k_open", 10)),
               K_HOLD=str(st.get("k_hold", 25)),
               PRE_FILTER=st.get("pre", "none"),
               PRE_SIGMA=str(st.get("pre_sigma", 2.0)),
               PRE_AMOUNT=str(st.get("pre_amount", 1.5)),
               PYTHONPATH=os.pathsep.join(
                   [str(libero_plus_dir), str(lerobot_src),
                    env.get("PYTHONPATH", "")]))
    cmd = [
        "python", str(Path(tools_dir) / "eval_adaptive.py"),
        f"--policy.path={model_dir}", "--policy.device=cuda",
        "--policy.use_amp=false", f"--policy.n_action_steps={st['nas']}",
        "--env.type=libero", "--env.is_libero_plus=true",
        "--env.task=libero_object",
        "--env.task_ids=" + json.dumps(task_ids, separators=(",", ":")),
        "--env.camera_name_mapping="
        + json.dumps(camera_mapping, separators=(",", ":")),
        "--env.observation_height=256", "--env.observation_width=256",
        "--env.control_mode=relative", "--env.max_parallel_tasks=1",
        "--eval.batch_size=1", "--eval.n_episodes=1",
        "--eval.use_async_envs=false", "--eval.recording=false",
        f"--seed={seed}", f"--output_dir={out}",
    ]
    if episode_length:
        cmd.append(f"--env.episode_length={episode_length}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=lerobot_dir, env=env, capture_output=True,
                       text=True)
    wall = time.time() - t0
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
        return None
    info = json.loads(next(out.rglob("eval_info.json")).read_text())
    ok = {t["task_id"]: bool(t["metrics"]["successes"][0])
          for t in info["per_task"]}
    eps = json.loads(stats.read_text())["episodes"]
    ms = [x for e in eps for x in e["call_ms"]]
    return {
        "setting": st["name"], "n": len(ok), "success": sum(ok.values()),
        "per_task": ok,
        "calls_per_episode": sum(e["calls"] for e in eps) / len(eps),
        "steps_per_episode": sum(e["steps"] for e in eps) / len(eps),
        "ms_per_call": sum(ms) / max(len(ms), 1),
        "wall_s": round(wall, 1),
    }


def run_sweep(model_dir, settings, ids_by_object, out_root, save_json,
              **kw):
    """設定ごとに全 task_id を評価。済んだ設定は save_json から読んで飛ばす。"""
    save_json = Path(save_json)
    done = json.loads(save_json.read_text()) if save_json.is_file() else []
    seen = {r["setting"] for r in done}
    task_ids = [i for xs in ids_by_object.values() for i in xs]
    for st in settings:
        if st["name"] in seen:
            print("skip", st["name"])
            continue
        print("run ", st["name"], f"({len(task_ids)} rollouts)", flush=True)
        row = run_one(model_dir, st, task_ids, out_root, **kw)
        if row is None:
            break
        done.append(row)
        save_json.write_text(json.dumps(done, indent=1))
        print(f"  success {row['success']}/{row['n']}  "
              f"calls/ep {row['calls_per_episode']:.1f}  "
              f"{row['ms_per_call']:.0f} ms/call", flush=True)
    return done


def summarize(rows):
    """設定ごとの成功率・推論回数・1回の時間・1エピソードの推論時間を表にする。"""
    import pandas as pd
    d = pd.DataFrame([{
        "設定": r["setting"],
        "成功": f"{r['success']}/{r['n']}",
        "成功率": round(r["success"] / r["n"], 2),
        "推論回数/エピソード": round(r["calls_per_episode"], 1),
        "ms/回": round(r["ms_per_call"]),
        "推論時間/エピソード(s)": round(
            r["calls_per_episode"] * r["ms_per_call"] / 1000, 1),
        "動作1回あたりの余裕(ms)": None,
    } for r in rows])
    return d.drop(columns=["動作1回あたりの余裕(ms)"])
