"""lerobot-eval の前に差し込む、見直し間隔の制御と計測 (ライブラリは書き換えない)。

使い方 (lerobot-eval と同じ引数):
    REPLAN_MODE=fixed|grip  STATS_JSON=out.json  python eval_adaptive.py \\
        --policy.path=... --env.type=libero ...

  fixed : 何もしない。見直し間隔は --policy.n_action_steps で決まる
  grip  : つかむ前は K_OPEN 動作ごと、つかんでいる間は K_HOLD 動作ごとに
          見直す (--policy.n_action_steps=50 で動かすこと)
どのモードでも、推論の回数・1回の時間・グリッパーの開き具合を記録する。

おまけ(ぼやけ対策): PRE_FILTER=unsharp で、メインカメラの画像に、推論の直前に
アンシャープマスク(ぼかした画像との差を足して輪郭を強調する)をかける。
  PRE_SIGMA  : ぼかしの幅(大きいほど強く補正)   既定 2.0
  PRE_AMOUNT : 足す量(大きいほど強く補正)       既定 1.5
  PRE_KEYS   : 対象のカメラ名(カンマ区切り)      既定 front,image,camera1(メインカメラ)
  PRE_MARK   : 補正がかかったら、そのカメラ名を書き込むファイル(確認用)
"""
import atexit
import json
import os
import sys
import time
from pathlib import Path

MODE = os.environ.get("REPLAN_MODE", "fixed")
K_OPEN = int(os.environ.get("K_OPEN", "10"))    # つかむ前の見直し間隔
K_HOLD = int(os.environ.get("K_HOLD", "25"))    # つかんでいる間の見直し間隔
MIN_DROP = float(os.environ.get("MIN_DROP", "0.3"))  # 閉じたとみなす最小変化
STATS_JSON = os.environ.get("STATS_JSON", "")
PRE_FILTER = os.environ.get("PRE_FILTER", "none")
PRE_SIGMA = float(os.environ.get("PRE_SIGMA", "2.0"))
PRE_AMOUNT = float(os.environ.get("PRE_AMOUNT", "1.5"))
PRE_KEYS = os.environ.get("PRE_KEYS", "front,image,camera1").split(",")
PRE_MARK = os.environ.get("PRE_MARK", "")


def unsharp(x, sigma, amount):
    """x: (B, C, H, W) の画像 (0〜1)。x + amount * (x - ガウスぼかし(x))。"""
    import torch
    import torch.nn.functional as F
    r = max(1, int(3 * sigma))
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(k ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    kh = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    kv = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    y = F.pad(x, (r, r, 0, 0), mode="reflect")
    y = F.conv2d(y, kh, groups=c)
    y = F.pad(y, (0, 0, r, r), mode="reflect")
    y = F.conv2d(y, kv, groups=c)
    return (x + amount * (x - y)).clamp(0.0, 1.0)


class Replanner:
    """グリッパーの開き幅 w の履歴から、いつ見直すかを決める (torch 非依存)。"""

    def __init__(self):
        self.w0 = None
        self.wmin = None
        self.since = 0

    def holding(self, w: float) -> bool:
        """初期の開き w0 から、最小値までの半分以上閉じたら『つかんでいる』。"""
        if self.w0 is None:
            self.w0 = self.wmin = w
        self.wmin = min(self.wmin, w)
        drop = self.w0 - self.wmin
        return drop > MIN_DROP and (self.w0 - w) > 0.5 * drop

    def want_replan(self, w: float) -> bool:
        k = K_HOLD if self.holding(w) else K_OPEN
        return self.since >= k


def install():
    import torch
    from lerobot.policies.smolvla import modeling_smolvla as M
    from lerobot.utils.constants import ACTION, OBS_STATE

    P = M.SmolVLAPolicy
    orig_select, orig_chunk, orig_reset = (P.select_action,
                                           P._get_action_chunk, P.reset)
    episodes, cur = [], {}

    def new_cur():
        return {"steps": 0, "calls": 0, "call_ms": [], "w": []}

    def reset(self):
        if cur.get("steps"):
            episodes.append(dict(cur))
        cur.clear()
        cur.update(new_cur())
        self._rp = Replanner()
        return orig_reset(self)

    def chunk(self, *a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = orig_chunk(self, *a, **k)
        torch.cuda.synchronize()
        cur["call_ms"].append((time.perf_counter() - t) * 1e3)
        cur["calls"] += 1
        self._rp.since = 0
        return out

    def select(self, batch, noise=None, **kw):
        s = batch[OBS_STATE].reshape(-1, batch[OBS_STATE].shape[-1])[-1]
        w = float(s[6] - s[7])           # 2 本の指の開き (正規化された値)
        cur["w"].append(round(w, 3))
        cur["steps"] += 1
        if MODE == "grip" and self._rp.want_replan(w) \
                and len(self._queues[ACTION]) > 0:
            self._queues[ACTION].clear()  # 残りの行動を捨て、次で再計画
        else:
            self._rp.holding(w)           # 状態の更新だけ
        if PRE_FILTER == "unsharp":
            for key in list(batch.keys()):
                if (key.startswith("observation.images.")
                        and key.split(".")[-1] in PRE_KEYS):
                    batch[key] = unsharp(batch[key], PRE_SIGMA, PRE_AMOUNT)
                    if PRE_MARK and not Path(PRE_MARK).exists():
                        Path(PRE_MARK).write_text(key)   # かかった印
        out = orig_select(self, batch, noise, **kw)
        self._rp.since += 1
        return out

    P.reset, P._get_action_chunk, P.select_action = reset, chunk, select

    def dump():
        if cur.get("steps"):
            episodes.append(dict(cur))
        if STATS_JSON:
            Path(STATS_JSON).write_text(json.dumps(
                {"mode": MODE, "k_open": K_OPEN, "k_hold": K_HOLD,
                 "episodes": episodes}))

    atexit.register(dump)


if __name__ == "__main__":
    install()
    from lerobot.scripts.lerobot_eval import main
    sys.argv[0] = "lerobot-eval"
    main()
