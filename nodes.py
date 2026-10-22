# -*- coding: utf-8 -*-
import hashlib
import math
import os
import re
from fractions import Fraction

import torch

import folder_paths

from . import loop_core as core
from . import rife_backend as rb
from . import video_io as vio

CATEGORY = "AutoLoop"
SCALES = ["1.0", "0.5", "0.25", "2.0"]


# --------------------------------------------------------------------------- #
# 共通処理
# --------------------------------------------------------------------------- #
def _interp_fn(scale, fp16):
    engine = rb.get_engine()
    sc = float(scale)
    return lambda a, b, n: engine.interpolate(a, b, n, sc, fp16)


def fix_loop(images, fps, max_search_n, min_m, max_m, allow_start_cut, candidate_rank,
             top_candidates, motion_bias, rife_scale, fp16):
    interp = _interp_fn(rife_scale, fp16)
    prog = core.make_progress(max(top_candidates, 1) * 2 + 1)
    cands = core.search_loops(
        images, interp, max_search_n=max_search_n, min_m=min_m, max_m=max_m,
        allow_start_cut=allow_start_cut, top_candidates=top_candidates,
        motion_bias=motion_bias, progress=prog,
    )
    rank = max(1, min(int(candidate_rank), len(cands)))
    loop = core.assemble_loop(images, cands[rank - 1])
    info = f"入力 {images.shape[0]}フレーム @ {fps:.3f}fps\n" + core.describe_candidates(cands, rank, fps)
    return loop, info


def interpolate(images, mult, rife_scale, fp16, cyclic, dedupe, dedupe_thr, cut_on, cut_thr):
    mult = int(mult)
    if mult < 2:
        return images, {"cuts": 0, "dups": 0}
    interp = _interp_fn(rife_scale, fp16)
    prog = core.make_progress(images.shape[0] + 1)
    return core.interpolate_sequence(
        images, mult, interp, cyclic=cyclic, dedupe=dedupe, thr=dedupe_thr,
        cut_on=cut_on, cut_thr=cut_thr, progress=prog,
    )


def _sanitize(s):
    return re.sub(r"[^\w\-]+", "_", s).strip("_") or "video"


# --------------------------------------------------------------------------- #
# 1. 複数動画のローダー(リスト出力)
# --------------------------------------------------------------------------- #
class AL_LoadVideoBatch:
    """複数の動画を読み込み、IMAGE のリストとして出力する。
    後段のノードは ComfyUI の仕様で動画ごとに自動で繰り返し実行される。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "paths": ("STRING", {"multiline": True, "default": "",
                                 "tooltip": "1行に1つ。動画ファイル / フォルダ / ワイルドカード(例: D:/clips/*.mp4)。"
                                            "相対パスは ComfyUI の input フォルダからも探します。"}),
            "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "0 で全フレーム"}),
            "skip_first": ("INT", {"default": 0, "min": 0, "max": 100000}),
        }}

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("images", "fps", "name")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, paths, max_frames, skip_first):
        h = hashlib.md5(f"{max_frames}{skip_first}".encode())
        try:
            for p in vio.expand_paths(paths, [folder_paths.get_input_directory()]):
                h.update(f"{p}{os.path.getmtime(p)}".encode())
        except Exception:
            return float("nan")
        return h.hexdigest()

    def load(self, paths, max_frames, skip_first):
        files = vio.expand_paths(paths, [folder_paths.get_input_directory()])
        if not files:
            raise ValueError("動画パスを入力してください")
        imgs, fpss, names = [], [], []
        for f in files:
            frames, fps = vio.read_video(f, max_frames, skip_first)
            imgs.append(frames)
            fpss.append(fps)
            names.append(_sanitize(os.path.splitext(os.path.basename(f))[0]))
        return (imgs, fpss, names)


# --------------------------------------------------------------------------- #
# 2. ループ修正
# --------------------------------------------------------------------------- #
class AL_AutoLoopFix:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "fps": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 1000.0, "step": 0.001}),
            "max_search_n": ("INT", {"default": 30, "min": 0, "max": 300,
                                     "tooltip": "末尾から最大何フレームまで削ってループ点を探すか"}),
            "min_m": ("INT", {"default": 1, "min": 0, "max": 32, "tooltip": "継ぎ目に挟む補間フレーム数の下限"}),
            "max_m": ("INT", {"default": 8, "min": 0, "max": 32, "tooltip": "継ぎ目に挟む補間フレーム数の上限"}),
            "allow_start_cut": ("BOOLEAN", {"default": True, "tooltip": "先頭側も削って開始位置を探す"}),
            "top_candidates": ("INT", {"default": 3, "min": 1, "max": 5}),
            "candidate_rank": ("INT", {"default": 1, "min": 1, "max": 5, "tooltip": "採用する候補の順位(1=最良)"}),
            "motion_bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                                      "tooltip": "運動量重視度。大きいほど動きの大きいループ点を優先"}),
            "rife_scale": (SCALES, {"default": "1.0"}),
            "fp16": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("images", "fps", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, fps, max_search_n, min_m, max_m, allow_start_cut, top_candidates,
            candidate_rank, motion_bias, rife_scale, fp16):
        loop, info = fix_loop(images, fps, max_search_n, min_m, max_m, allow_start_cut,
                              candidate_rank, top_candidates, motion_bias, rife_scale, fp16)
        return (loop, fps, info)


# --------------------------------------------------------------------------- #
# 3. フレーム補完(ループ対応)
# --------------------------------------------------------------------------- #
class AL_LoopInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "fps": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 1000.0, "step": 0.001}),
            "multiplier": ("INT", {"default": 2, "min": 1, "max": 8, "tooltip": "fps を何倍にするか(1で何もしない)"}),
            "cyclic": ("BOOLEAN", {"default": True, "tooltip": "最終→先頭も補完してループ全体を等間隔にする"}),
            "rife_scale": (SCALES, {"default": "1.0"}),
            "fp16": ("BOOLEAN", {"default": True}),
            "dedupe": ("BOOLEAN", {"default": False, "tooltip": "2コマ撮り・3コマ撮りの重複フレームをまとめてから補完"}),
            "dedupe_thr": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 5.0, "step": 0.1}),
            "cut_detect": ("BOOLEAN", {"default": True, "tooltip": "シーンカットをまたぐ補完を避ける"}),
            "cut_thr": ("FLOAT", {"default": 30.0, "min": 10.0, "max": 100.0, "step": 1.0}),
        }}

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "fps")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, fps, multiplier, cyclic, rife_scale, fp16, dedupe, dedupe_thr, cut_detect, cut_thr):
        out, _ = interpolate(images, multiplier, rife_scale, fp16, cyclic, dedupe, dedupe_thr, cut_detect, cut_thr)
        return (out, float(fps) * max(int(multiplier), 1))


# --------------------------------------------------------------------------- #
# 4. 再生速度
# --------------------------------------------------------------------------- #
class AL_LoopSpeed:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "fps": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 1000.0, "step": 0.001}),
            "speed": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.05,
                                "tooltip": "0.25 なら 4倍スロー、2.0 なら 2倍速"}),
            "target_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001,
                                     "tooltip": "0: フレームはそのまま fps だけ変更 / >0: その fps になるよう再サンプリング"}),
            "blend": ("BOOLEAN", {"default": True, "tooltip": "再サンプリング時に隣接フレームをブレンド(オフで最近傍)"}),
        }}

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "fps")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, fps, speed, target_fps, blend):
        return core.retime(images, fps, speed, target_fps, blend)


# --------------------------------------------------------------------------- #
# 5. オールインワン
# --------------------------------------------------------------------------- #
class AL_LoopMaker:
    """ループ修正 → フレーム補完 → 速度変更 を1ノードで実行"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "fps": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 1000.0, "step": 0.001}),
            "fix_loop": ("BOOLEAN", {"default": True}),
            "max_search_n": ("INT", {"default": 30, "min": 0, "max": 300}),
            "min_m": ("INT", {"default": 1, "min": 0, "max": 32}),
            "max_m": ("INT", {"default": 8, "min": 0, "max": 32}),
            "allow_start_cut": ("BOOLEAN", {"default": True}),
            "top_candidates": ("INT", {"default": 3, "min": 1, "max": 5}),
            "candidate_rank": ("INT", {"default": 1, "min": 1, "max": 5}),
            "motion_bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                                      "tooltip": "運動量重視度。大きいほど動きの大きいループ点を優先"}),
            "multiplier": ("INT", {"default": 2, "min": 1, "max": 8}),
            "speed": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.05}),
            "target_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001,
                                     "tooltip": "0: fps×倍率×速度 をそのまま出力fpsに / >0: その fps に再サンプリング"}),
            "rife_scale": (SCALES, {"default": "1.0"}),
            "fp16": ("BOOLEAN", {"default": True}),
            "dedupe": ("BOOLEAN", {"default": False}),
            "dedupe_thr": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 5.0, "step": 0.1}),
            "cut_detect": ("BOOLEAN", {"default": True}),
            "cut_thr": ("FLOAT", {"default": 30.0, "min": 10.0, "max": 100.0, "step": 1.0}),
            "free_vram_after": ("BOOLEAN", {"default": False, "tooltip": "処理後に RIFE をVRAMから解放"}),
        }}

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("images", "fps", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, fps, fix_loop, max_search_n, min_m, max_m, allow_start_cut, top_candidates,
            candidate_rank, motion_bias, multiplier, speed, target_fps, rife_scale, fp16, dedupe, dedupe_thr,
            cut_detect, cut_thr, free_vram_after):
        info = [f"入力 {images.shape[0]}フレーム @ {fps:.3f}fps"]
        cur = images
        try:
            if fix_loop:
                cur, txt = globals()["fix_loop"](cur, fps, max_search_n, min_m, max_m, allow_start_cut,
                                                 candidate_rank, top_candidates, motion_bias, rife_scale, fp16)
                info.append("[ループ修正]\n" + txt.split("\n", 1)[1])
            fps_i = float(fps)
            if int(multiplier) >= 2:
                cur, st = interpolate(cur, multiplier, rife_scale, fp16, True, dedupe, dedupe_thr,
                                      cut_detect, cut_thr)
                fps_i *= int(multiplier)
                info.append(f"[補完] ×{int(multiplier)} → {cur.shape[0]}フレーム (重複扱い {st['dups']} / カット扱い {st['cuts']})")
            cur, fps_o = core.retime(cur, fps_i, speed, target_fps, True)
            info.append(f"[出力] {cur.shape[0]}フレーム @ {fps_o:.3f}fps / ループ1周 {cur.shape[0] / fps_o:.2f}秒 (速度 {speed:g}x)")
        finally:
            if free_vram_after:
                rb.release_engine()
        return (cur, fps_o, "\n".join(info))


# --------------------------------------------------------------------------- #
# 6. 保存
# --------------------------------------------------------------------------- #
class AL_SaveLoop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "AutoLoop/loop"}),
                "format": (["mp4", "webp", "gif"], {"default": "mp4"}),
                "loop_repeats": ("INT", {"default": 1, "min": 1, "max": 100,
                                         "tooltip": "ファイルに書き出すループの繰り返し回数(確認用)"}),
                "crf": ("INT", {"default": 16, "min": 0, "max": 51, "tooltip": "mp4 の画質(小さいほど高画質)"}),
            },
            "optional": {"name": ("STRING", {"forceInput": True, "tooltip": "ローダーの name を繋ぐとファイル名に使われます"})},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def save(self, images, fps, filename_prefix, format, loop_repeats, crf, name=None):
        out_root = folder_paths.get_output_directory()
        prefix = filename_prefix.replace("\\", "/").strip("/")
        sub, base = os.path.split(prefix)
        base = _sanitize(base) if base else "loop"
        if name:
            base = f"{base}_{_sanitize(name)}"
        folder = os.path.normpath(os.path.join(out_root, sub))
        if not folder.startswith(os.path.abspath(out_root)):
            raise ValueError("出力先が output フォルダの外です")
        os.makedirs(folder, exist_ok=True)

        path, i = os.path.join(folder, f"{base}.{format}"), 1
        while os.path.exists(path):
            path = os.path.join(folder, f"{base}_{i:04d}.{format}")
            i += 1

        if format == "mp4":
            vio.write_mp4(path, images, fps, loop_repeats, crf)
        else:
            vio.write_pil(path, images, fps, format, loop_repeats)

        rel_sub = os.path.relpath(folder, out_root)
        rel_sub = "" if rel_sub == "." else rel_sub
        item = {"filename": os.path.basename(path), "subfolder": rel_sub, "type": "output"}
        if format == "mp4":
            ui = {
                "text": [path],
                "videos": [item],
                "images": [{**item, "format": "video/mp4"}],
            }
        else:
            ui = {
                "text": [path],
                "images": [item],
                "animated": (True,),
            }
        return {"ui": ui}


# --------------------------------------------------------------------------- #
# 7. ComfyUI 標準の VIDEO 型への変換(プレビュー/保存に標準ノードを使いたい場合)
# --------------------------------------------------------------------------- #
class AL_ToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "fps": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 1000.0, "step": 0.001}),
        }}

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, fps):
        rate = Fraction(float(fps)).limit_denominator(1000)
        try:
            from comfy_api.latest import InputImpl, Types
            return (InputImpl.VideoFromComponents(
                Types.VideoComponents(images=images, audio=None, frame_rate=rate)),)
        except ImportError:
            from comfy_api.input_impl import VideoFromComponents
            from comfy_api.util import VideoComponents
            return (VideoFromComponents(VideoComponents(images=images, audio=None, frame_rate=rate)),)

class AL_FromVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO",),
        }}

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "fps")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, video):
        components = video.get_components() if hasattr(video, "get_components") else video
        images = getattr(components, "images", None)
        if images is None:
            raise ValueError("VIDEO 型から画像フレームを取得できませんでした")
        fps = float(getattr(components, "frame_rate", 24.0))
        return (images, fps)

class AL_EasyLoop:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO",),
            "target_fps": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 240.0, "step": 1.0,
                                     "tooltip": "目標FPS (0で倍率通りのFPS)"}),
            "multiplier": ("INT", {"default": 2, "min": 1, "max": 8, "tooltip": "RIFE補間倍率"}),
            "repeats": ("INT", {"default": 3, "min": 1, "max": 10, "tooltip": "確認用ループ繰り返し回数"}),
            "motion_bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                                      "tooltip": "運動量重視度"}),
        }}

    RETURN_TYPES = ("VIDEO", "VIDEO", "VIDEO", "VIDEO", "VIDEO", "VIDEO")
    RETURN_NAMES = ("rank1_3x", "rank1_single", "rank2_3x", "rank2_single", "rank3_3x", "rank3_single")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, video, target_fps, multiplier, repeats, motion_bias):
        components = video.get_components() if hasattr(video, "get_components") else video
        images = getattr(components, "images", None)
        if images is None:
            raise ValueError("VIDEO 型から画像フレームを取得できませんでした。動画ファイルが正しく選択されているか確認してください。")
        fps = float(getattr(components, "frame_rate", 24.0))

        interp = _interp_fn("1.0", True)
        prog = core.make_progress(3 * 2 + 1)
        cands = core.search_loops(
            images, interp, max_search_n=30, min_m=1, max_m=8,
            allow_start_cut=True, top_candidates=3, motion_bias=motion_bias, progress=prog,
        )
        if not cands:
            raise ValueError("ループ候補が見つかりませんでした")

        def to_video_obj(frames_tensor, out_fps):
            rate = Fraction(float(out_fps)).limit_denominator(1000)
            try:
                from comfy_api.latest import InputImpl, Types
                return InputImpl.VideoFromComponents(
                    Types.VideoComponents(images=frames_tensor, audio=None, frame_rate=rate))
            except ImportError:
                from comfy_api.input_impl import VideoFromComponents
                from comfy_api.util import VideoComponents
                return VideoFromComponents(VideoComponents(images=frames_tensor, audio=None, frame_rate=rate))

        def process_cand(cand):
            loop = core.assemble_loop(images, cand)
            fps_cur = fps
            if int(multiplier) >= 2:
                loop, _ = interpolate(loop, multiplier, "1.0", True, True, False, 1.0, True, 30.0)
                fps_cur *= int(multiplier)
            loop_1x, fps_out = core.retime(loop, fps_cur, 1.0, target_fps, True)
            rep = max(1, int(repeats))
            loop_3x = torch.cat([loop_1x] * rep, dim=0) if rep > 1 else loop_1x
            return to_video_obj(loop_3x, fps_out), to_video_obj(loop_1x, fps_out)

        cand_results = [process_cand(c) for c in cands]
        while len(cand_results) < 3:
            cand_results.append(cand_results[-1])

        return (
            cand_results[0][0], cand_results[0][1],
            cand_results[1][0], cand_results[1][1],
            cand_results[2][0], cand_results[2][1],
        )

class AL_AuditExport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "fps": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 1000.0, "step": 0.001}),
            "name": ("STRING", {"default": "video"}),
            "target_fps": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 240.0, "step": 1.0}),
            "multiplier": ("INT", {"default": 2, "min": 1, "max": 8}),
            "motion_bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
            "crf": ("INT", {"default": 16, "min": 0, "max": 51}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "export"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def export(self, images, fps, name, target_fps, multiplier, motion_bias, crf):
        safe_name = _sanitize(name)
        out_root = folder_paths.get_output_directory()
        folder = os.path.join(out_root, "AutoLoop", safe_name)
        os.makedirs(folder, exist_ok=True)

        interp = _interp_fn("1.0", True)
        prog = core.make_progress(3 * 2 + 1)
        cands = core.search_loops(
            images, interp, max_search_n=30, min_m=1, max_m=8,
            allow_start_cut=True, top_candidates=3, motion_bias=motion_bias, progress=prog,
        )
        if not cands:
            raise ValueError("ループ候補が見つかりませんでした")

        saved_files = []
        for i, cand in enumerate(cands, 1):
            loop = core.assemble_loop(images, cand)
            fps_cur = float(fps)
            if int(multiplier) >= 2:
                loop, _ = interpolate(loop, multiplier, "1.0", True, True, False, 1.0, True, 30.0)
                fps_cur *= int(multiplier)
            loop_1x, fps_out = core.retime(loop, fps_cur, 1.0, target_fps, True)
            loop_3x = torch.cat([loop_1x] * 3, dim=0)

            p_1x = os.path.join(folder, f"rank{i}_1x.mp4")
            p_3x = os.path.join(folder, f"rank{i}_3x.mp4")
            vio.write_mp4(p_1x, loop_1x, fps_out, 1, crf)
            vio.write_mp4(p_3x, loop_3x, fps_out, 1, crf)
            saved_files.extend([p_1x, p_3x])

        return {"ui": {"text": saved_files}}

class AL_FrameInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
                                         "tooltip": "表示を開始するコマ番号"}),
                "view_count": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1,
                                       "tooltip": "表示するコマ数 (1なら単一コマ、複数ならグリッド一覧)"}),
                "grid_columns": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1,
                                         "tooltip": "複数コマ表示時のタイル列数"}),
            },
            "optional": {
                "video": ("VIDEO",),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "total_frames")
    OUTPUT_NODE = True
    FUNCTION = "inspect"
    CATEGORY = CATEGORY

    def inspect(self, frame_index, view_count, grid_columns, video=None, images=None):
        from PIL import Image

        frames = None
        if video is not None:
            components = video.get_components() if hasattr(video, "get_components") else video
            frames = getattr(components, "images", None)
        elif images is not None:
            frames = images

        if frames is None or frames.shape[0] == 0:
            raise ValueError("video または images を接続してください")

        total = int(frames.shape[0])
        start = min(max(int(frame_index), 0), total - 1)
        count = max(int(view_count), 1)
        end = min(start + count, total)
        selected = frames[start:end]

        if selected.shape[0] == 1:
            out_img = selected
        else:
            n, h, w, c = selected.shape
            cols = min(int(grid_columns), n)
            rows = math.ceil(n / cols)
            grid = torch.zeros((rows * h, cols * w, c), dtype=selected.dtype)
            for idx in range(n):
                r, col = idx // cols, idx % cols
                grid[r * h:(r + 1) * h, col * w:(col + 1) * w] = selected[idx]
            out_img = grid.unsqueeze(0)

        temp_dir = folder_paths.get_temp_directory()
        tag = hashlib.md5(f"{start}_{count}_{cols if selected.shape[0] > 1 else 1}".encode()).hexdigest()[:8]
        filename = f"inspect_{tag}.png"
        filepath = os.path.join(temp_dir, filename)

        arr = (out_img[0].clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
        Image.fromarray(arr).save(filepath)

        ui = {"images": [{"filename": filename, "subfolder": "", "type": "temp"}]}
        return {"ui": ui, "result": (out_img, total)}

from . import cugan_backend as cb

class AL_CUGANUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "scale": ([2, 3, 4], {"default": 2}),
            "denoise_level": (["no_denoise", "conservative", "denoise3x"], {"default": "conservative"}),
            "tile": ("INT", {"default": 256, "min": 0, "max": 2048, "step": 64}),
            "fp16": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = CATEGORY

    def upscale(self, images, scale, denoise_level, tile, fp16):
        engine = cb.get_cugan_engine(scale=scale, denoise_level=denoise_level, tile=tile, fp16=fp16)
        n = images.shape[0]
        prog = core.make_progress(n)
        outs = []
        for i in range(n):
            core.check_interrupt()
            t = images[i:i + 1].permute(0, 3, 1, 2).to(engine.device)
            up = engine.enhance_tensor(t)
            outs.append(up.permute(0, 2, 3, 1).cpu())
            prog(1)
        return (torch.cat(outs, dim=0).float(),)

class AL_VideoInterpolateToFps:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "fps": ("FLOAT", {"forceInput": True}),
            "target_fps": ("FLOAT", {"default": 120.0, "min": 1.0, "max": 1000.0, "step": 1.0}),
            "rife_scale": (SCALES, {"default": "1.0"}),
            "fp16": ("BOOLEAN", {"default": True}),
            "dedupe": ("BOOLEAN", {"default": False}),
            "dedupe_thr": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 5.0, "step": 0.1}),
            "cut_detect": ("BOOLEAN", {"default": True}),
            "cut_thr": ("FLOAT", {"default": 30.0, "min": 10.0, "max": 100.0, "step": 1.0}),
        }}

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "fps")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, fps, target_fps, rife_scale, fp16, dedupe, dedupe_thr, cut_detect, cut_thr):
        mult = max(2, int(math.ceil(float(target_fps) / float(fps))))
        interp = _interp_fn(rife_scale, fp16)
        prog = core.make_progress(images.shape[0] + 1)
        interp_frames, _ = core.interpolate_sequence(
            images, mult, interp, cyclic=False, dedupe=dedupe, thr=dedupe_thr,
            cut_on=cut_detect, cut_thr=cut_thr, progress=prog,
        )
        fps_interp = float(fps) * mult
        out_frames, out_fps = core.retime(interp_frames, fps_interp, 1.0, target_fps=target_fps, blend=True)
        return (out_frames, out_fps)

class AL_AnimeEnhance120fps:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "fps": ("FLOAT", {"forceInput": True}),
            "target_fps": ("FLOAT", {"default": 120.0, "min": 1.0, "max": 1000.0, "step": 1.0}),
            "process_order": (["RIFE_then_CUGAN", "CUGAN_then_RIFE"], {"default": "RIFE_then_CUGAN"}),
            "cugan_scale": ([2, 3, 4], {"default": 2}),
            "denoise_level": (["no_denoise", "conservative", "denoise3x"], {"default": "conservative"}),
            "cugan_tile": ("INT", {"default": 256, "min": 0, "max": 2048, "step": 64}),
            "rife_scale": (SCALES, {"default": "1.0"}),
            "fp16": ("BOOLEAN", {"default": True}),
            "dedupe": ("BOOLEAN", {"default": False}),
            "dedupe_thr": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 5.0, "step": 0.1}),
            "cut_detect": ("BOOLEAN", {"default": True}),
            "cut_thr": ("FLOAT", {"default": 30.0, "min": 10.0, "max": 100.0, "step": 1.0}),
        }}

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "fps", "info", "interp_images", "prefix_enhanced", "prefix_interp")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, fps, target_fps, process_order, cugan_scale, denoise_level, cugan_tile,
            rife_scale, fp16, dedupe, dedupe_thr, cut_detect, cut_thr):
        import time
        cugan = AL_CUGANUpscale()
        vfi = AL_VideoInterpolateToFps()
        n_in = images.shape[0]

        def sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        t_total_start = time.perf_counter()
        if process_order == "RIFE_then_CUGAN":
            sync()
            t_rife_start = time.perf_counter()
            interp_cur, out_fps = vfi.run(images, fps, target_fps, rife_scale, fp16, dedupe, dedupe_thr, cut_detect, cut_thr)
            sync()
            t_rife = time.perf_counter() - t_rife_start

            sync()
            t_cugan_start = time.perf_counter()
            cur, = cugan.upscale(interp_cur, cugan_scale, denoise_level, cugan_tile, fp16)
            sync()
            t_cugan = time.perf_counter() - t_cugan_start
            raw_interp = interp_cur
        else:
            sync()
            t_rife_start = time.perf_counter()
            raw_interp, out_fps = vfi.run(images, fps, target_fps, rife_scale, fp16, dedupe, dedupe_thr, cut_detect, cut_thr)
            sync()
            t_rife = time.perf_counter() - t_rife_start

            sync()
            t_cugan_start = time.perf_counter()
            cugan_base, = cugan.upscale(images, cugan_scale, denoise_level, cugan_tile, fp16)
            sync()
            t_cugan = time.perf_counter() - t_cugan_start

            cur, _ = vfi.run(cugan_base, fps, target_fps, rife_scale, fp16, dedupe, dedupe_thr, cut_detect, cut_thr)

        sync()
        t_total = time.perf_counter() - t_total_start
        n_out = cur.shape[0]
        throughput = n_out / t_total if t_total > 0 else 0.0

        info = (
            f"順序: {process_order}\n"
            f"入力: {n_in}フレーム @ {fps:.2f}fps ({images.shape[2]}x{images.shape[1]})\n"
            f"出力: {n_out}フレーム @ {out_fps:.2f}fps ({cur.shape[2]}x{cur.shape[1]})\n"
            f"RIFE所要時間: {t_rife:.2f}秒\n"
            f"CUGAN所要時間: {t_cugan:.2f}秒\n"
            f"合計処理時間: {t_total:.2f}秒 (スループット: {throughput:.2f} fps)"
        )
        fps_label = f"{int(round(out_fps))}fps"
        prefix_enhanced = f"Enhanced/{fps_label}_upscaled"
        prefix_interp = f"Enhanced/{fps_label}_no_upscale"
        return (cur, out_fps, info, raw_interp, prefix_enhanced, prefix_interp)

class AL_UploadVideo:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.exists(input_dir):
            files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in vio.VIDEO_EXT]
        files = sorted(files)
        return {
            "required": {
                "video_file": (files, {"video_upload": True}),
            },
            "optional": {
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "skip_first": ("INT", {"default": 0, "min": 0, "max": 100000}),
            }
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("video", "images", "fps", "name")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, video_file, max_frames=0, skip_first=0):
        p = folder_paths.get_annotated_filepath(video_file)
        if not os.path.isfile(p):
            return float("nan")
        return f"{p}_{os.path.getmtime(p)}_{max_frames}_{skip_first}"

    def load(self, video_file, max_frames=0, skip_first=0):
        if not video_file:
            raise ValueError("動画ファイルを選択してください")
        path = folder_paths.get_annotated_filepath(video_file)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"動画ファイルが見つかりません: {path}")

        images, fps = vio.read_video(path, max_frames, skip_first)
        name = _sanitize(os.path.splitext(os.path.basename(path))[0])

        rate = Fraction(float(fps)).limit_denominator(1000)
        try:
            from comfy_api.latest import InputImpl, Types
            v_out = InputImpl.VideoFromComponents(
                Types.VideoComponents(images=images, audio=None, frame_rate=rate))
        except ImportError:
            from comfy_api.input_impl import VideoFromComponents
            from comfy_api.util import VideoComponents
            v_out = VideoFromComponents(VideoComponents(images=images, audio=None, frame_rate=rate))

        return (v_out, images, fps, name)

class AL_EnhanceDebugger:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "fps": ("FLOAT", {"forceInput": True}),
            "target_fps": ("FLOAT", {"default": 120.0, "min": 1.0, "max": 1000.0, "step": 1.0}),
            "cugan_scale": ([2, 3, 4], {"default": 2}),
            "denoise_level": (["no_denoise", "conservative", "denoise3x"], {"default": "conservative"}),
            "cugan_tile": ("INT", {"default": 256, "min": 0, "max": 2048, "step": 64}),
            "rife_scale": (SCALES, {"default": "1.0"}),
            "fp16": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("input_frame", "rife_frame", "cugan_frame", "out_fps", "debug_report")
    FUNCTION = "diagnose"
    CATEGORY = CATEGORY

    def diagnose(self, images, fps, target_fps, cugan_scale, denoise_level, cugan_tile, rife_scale, fp16):
        report = []

        def analyze(t, name):
            t_f = t.float()
            has_nan = bool(torch.isnan(t_f).any())
            min_v = float(t_f.min())
            max_v = float(t_f.max())
            mean_v = float(t_f.mean())
            report.append(f"[{name}] 形状: {tuple(t.shape)} / 最小: {min_v:.4f} / 最大: {max_v:.4f} / 平均: {mean_v:.4f} / NaN混入: {has_nan}")
            if max_v == 0.0 and min_v == 0.0:
                report.append(f"  --> 警告: {name} の全ピクセルが真っ黒(0.0)です")
            if has_nan:
                report.append(f"  --> 警告: {name} にNaN(非数)が含まれています")

        analyze(images[:1], "1. 入力元画像")

        vfi = AL_VideoInterpolateToFps()
        rife_out, out_fps = vfi.run(images[:2], fps, target_fps, rife_scale, fp16, False, 1.0, False, 30.0)
        analyze(rife_out[:1], "2. RIFE補間後")

        cugan = AL_CUGANUpscale()
        cugan_out, = cugan.upscale(images[:1], cugan_scale, denoise_level, cugan_tile, fp16)
        analyze(cugan_out[:1], "3. CUGAN超解像後")

        rep_text = "\n".join(report)
        return (images[:1], rife_out[:1], cugan_out[:1], out_fps, rep_text)

NODE_CLASS_MAPPINGS = {
    "AL_LoadVideoBatch": AL_LoadVideoBatch,
    "AL_AutoLoopFix": AL_AutoLoopFix,
    "AL_LoopInterpolate": AL_LoopInterpolate,
    "AL_LoopSpeed": AL_LoopSpeed,
    "AL_LoopMaker": AL_LoopMaker,
    "AL_SaveLoop": AL_SaveLoop,
    "AL_ToVideo": AL_ToVideo,
    "AL_FromVideo": AL_FromVideo,
    "AL_EasyLoop": AL_EasyLoop,
    "AL_AuditExport": AL_AuditExport,
    "AL_FrameInspector": AL_FrameInspector,
    "AL_CUGANUpscale": AL_CUGANUpscale,
    "AL_VideoInterpolateToFps": AL_VideoInterpolateToFps,
    "AL_AnimeEnhance120fps": AL_AnimeEnhance120fps,
    "AL_UploadVideo": AL_UploadVideo,
    "AL_EnhanceDebugger": AL_EnhanceDebugger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AL_LoadVideoBatch": "動画を複数読み込み (リスト出力)",
    "AL_AutoLoopFix": "自動ループ修正",
    "AL_LoopInterpolate": "ループ対応フレーム補完 (RIFE)",
    "AL_LoopSpeed": "ループ再生速度 / リタイム",
    "AL_LoopMaker": "ループメーカー (オールインワン)",
    "AL_SaveLoop": "ループ動画を保存",
    "AL_ToVideo": "→ VIDEO 型に変換",
    "AL_FromVideo": "VIDEO 型から展開",
    "AL_EasyLoop": "かんたんループ＆FPS (VIDEO)",
    "AL_AuditExport": "監査用バッチループ出力 (1x/3x)",
    "AL_FrameInspector": "各コマ閲覧インスペクター",
    "AL_CUGANUpscale": "アニメ超解像 (Real-CUGAN)",
    "AL_VideoInterpolateToFps": "動画フレーム補完 / FPS指定 (RIFE)",
    "AL_AnimeEnhance120fps": "高FPS化 & CUGAN高画質化 (ループなし)",
    "AL_UploadVideo": "動画ファイル選択・アップロード",
    "AL_EnhanceDebugger": "高画質・高FPS診断デバッガー",
}
