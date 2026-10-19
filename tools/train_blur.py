"""lerobot-train に「ぼやけ」の水増しを足して学習する (ライブラリは書き換えない)。

使い方: lerobot-train の代わりに、同じ引数でこのファイルを python で呼ぶ。
    python train_blur.py --policy.path=... --dataset.repo_id=... (以下同じ)

学習データのメインカメラ画像に、確率 BLUR_P で次のどちらかをかける
(LIBERO-plus の「ぼやけ」と同じ種類: モーションブラー / ガウスぼかし)。
  BLUR_P    : ぼやけをかける確率          既定 0.5
  BLUR_KEYS : 対象のカメラ名(カンマ区切り) 既定 front,image,camera1
手首カメラ(wrist, image2, camera2)にはかけない。
"""
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

P_BLUR = float(os.environ.get("BLUR_P", "0.5"))
KEYS = set(os.environ.get("BLUR_KEYS", "front,image,camera1").split(","))
STATE = {"seen": 0, "blurred": 0, "keys": set()}


def gauss_kernel(sigma):
    r = max(1, int(3 * sigma))
    t = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    return (k[:, None] * k[None, :])


def motion_kernel(length, angle):
    n = length if length % 2 == 1 else length + 1
    k = torch.zeros(n, n)
    c = (n - 1) / 2
    for i in range(4 * n):
        t = -c + (2 * c) * i / (4 * n - 1)
        x = int(round(c + t * math.cos(angle)))
        y = int(round(c + t * math.sin(angle)))
        k[y, x] = 1.0
    return k / k.sum()


def random_blur_kernel():
    if random.random() < 0.7:     # モーションブラー(長さ5〜15, 向きはランダム)
        return motion_kernel(random.choice([5, 7, 9, 11, 13, 15]),
                             random.uniform(0, math.pi))
    return gauss_kernel(random.uniform(1.0, 3.0))   # ガウスぼかし


def apply_blur(x):
    """x: (..., C, H, W)。float(0〜1) か uint8。"""
    shape, dt = x.shape, x.dtype
    y = x.reshape(-1, *shape[-3:]).float()
    if dt == torch.uint8:
        y = y / 255.0
    k = random_blur_kernel()
    c = y.shape[1]
    r = k.shape[0] // 2
    w = k.to(y.device).view(1, 1, *k.shape).repeat(c, 1, 1, 1)
    y = F.conv2d(F.pad(y, (r, r, r, r), mode="reflect"), w, groups=c)
    if dt == torch.uint8:
        y = (y * 255.0).round().clamp(0, 255)
    return y.to(dt).reshape(shape)


def install():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    orig = LeRobotDataset.__getitem__

    def getitem(self, idx):
        item = orig(self, idx)
        STATE["seen"] += 1
        cams = sorted(k for k in item if k.startswith("observation.images."))
        if random.random() < P_BLUR:
            for key in cams:
                if key.split(".")[-1] in KEYS and torch.is_tensor(item[key]):
                    item[key] = apply_blur(item[key])
                    STATE["keys"].add(key)
                    STATE["blurred"] += 1
        n = STATE["seen"]
        if n in (1, 200, 2000):
            msg = (f"seen={n} blurred={STATE['blurred']} "
                   f"blurred_keys={sorted(STATE['keys'])} cameras={cams}")
            print("[train_blur]", msg, flush=True)
            with open("/content/train_blur_log.txt", "a") as f:
                f.write(msg + "\n")
            if n == 200 and STATE["blurred"] == 0:
                raise RuntimeError("train_blur: ぼやけがかかっていません。 " + msg)
        return item

    LeRobotDataset.__getitem__ = getitem


if __name__ == "__main__":
    install()
    from lerobot.scripts.lerobot_train import main
    sys.argv[0] = "lerobot-train"
    main()
