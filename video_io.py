# -*- coding: utf-8 -*-
"""動画の読み書き。ComfyUI 本体が依存している PyAV (av) だけで完結する。"""
import glob
import os
from fractions import Fraction

import numpy as np
import torch

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif", ".webp", ".flv", ".wmv"}


def expand_paths(text, base_dirs=()):
    """1行1パス。ファイル / フォルダ / ワイルドカード(*.mp4) に対応。相対パスは base_dirs からも探す"""
    out, seen = [], set()
    for line in text.splitlines():
        line = line.strip().strip('"').strip("'")
        if not line or line.startswith("#"):
            continue
        cands = [line] if os.path.isabs(line) else [line] + [os.path.join(b, line) for b in base_dirs]
        found = []
        for c in cands:
            if os.path.isdir(c):
                found = sorted(
                    f for f in glob.glob(os.path.join(c, "*"))
                    if os.path.splitext(f)[1].lower() in VIDEO_EXT
                )
            else:
                found = sorted(h for h in glob.glob(c) if os.path.isfile(h))
            if found:
                break
        if not found:
            raise FileNotFoundError(f"動画が見つかりません: {line}")
        for f in found:
            f = os.path.abspath(f)
            if f not in seen:
                seen.add(f)
                out.append(f)
    return out


def read_video(path, max_frames=0, skip_first=0):
    """戻り値: (frames (T,H,W,3) float32 0..1, fps)"""
    import av

    frames = []
    with av.open(path) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or s.guessed_rate or 24.0)
        for i, fr in enumerate(c.decode(video=0)):
            if i < skip_first:
                continue
            frames.append(fr.to_ndarray(format="rgb24"))
            if max_frames and len(frames) >= max_frames:
                break
    if not frames:
        raise ValueError(f"フレームを読み込めませんでした: {path}")
    arr = torch.from_numpy(np.stack(frames))
    return arr.float().div_(255.0), fps


def _to_uint8(frames):
    return (frames.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu().numpy()


def _iter_repeated(arr, repeats):
    for _ in range(max(1, int(repeats))):
        for i in range(arr.shape[0]):
            yield arr[i]


def write_mp4(path, frames, fps, repeats=1, crf=18, preset="medium"):
    import av

    arr = _to_uint8(frames)
    h, w = arr.shape[1:3]
    w2, h2 = w // 2 * 2, h // 2 * 2  # yuv420p は偶数サイズ必須
    rate = Fraction(float(fps)).limit_denominator(10000)
    with av.open(path, "w", options={"movflags": "+faststart"}) as c:
        st = c.add_stream("libx264", rate=rate)
        st.width, st.height, st.pix_fmt = w2, h2, "yuv420p"
        st.options = {"crf": str(int(crf)), "preset": preset}
        for img in _iter_repeated(arr, repeats):
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(img[:h2, :w2]), format="rgb24")
            for pkt in st.encode(vf):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)


def write_pil(path, frames, fps, fmt, repeats=1):
    from PIL import Image

    arr = _to_uint8(frames)
    imgs = [Image.fromarray(a) for a in _iter_repeated(arr, repeats)]
    dur = max(1, int(round(1000.0 / float(fps))))
    if fmt == "gif":
        dur = max(20, dur)  # GIF は 20ms 未満が安定しない
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=dur, loop=0, optimize=False)
    else:
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=dur, loop=0,
                     quality=90, method=4, lossless=False)
