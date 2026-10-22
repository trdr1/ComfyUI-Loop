# -*- coding: utf-8 -*-
"""
auto_loop.rs (Rust/Tauri 版) の自動ループ探索を torch に移植したコア。
ComfyUI の IMAGE (B,H,W,3 / float 0..1) をそのまま扱う。

ループの構造(Rust版と同じ):
    元動画の s 〜 k フレーム(1始まり)を使い、末尾 k → 先頭 s の継ぎ目に
    RIFE で作った m 枚の補間フレームを挟む。
        ループ = [s, s+1, ..., k] + [mid_1 ... mid_m]   (→ 先頭 s に戻る)
    s(開始位置), k(終了位置), m(補間枚数) を、視覚類似度・動きの連続性・実補間後の継ぎ目で評価して決める。
"""
import math

import torch
import torch.nn.functional as F

# ---- 定数 (Rust版と同じ) ----
THUMB_W = 256
BLOCK = 8
MARGIN = 10
MOTION_RANGE = 6
GAP_RANGE = 10
PAIR_LIMIT = 80

CUT_THR = 30.0
CUT_CORR = 0.8


def check_interrupt():
    try:
        import comfy.model_management as mm
        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def make_progress(total):
    try:
        from comfy.utils import ProgressBar
        pb = ProgressBar(max(int(total), 1))
        return lambda n=1: pb.update(n)
    except Exception:
        return lambda n=1: None


# --------------------------------------------------------------------------- #
# サムネイルと類似度
# --------------------------------------------------------------------------- #
class Thumb:
    __slots__ = ("w", "h", "rgb", "gray", "half")

    def __init__(self, frame):
        """frame: (H,W,3) float 0..1"""
        h0, w0 = frame.shape[:2]
        tw = THUMB_W
        th = max(16, int(round(h0 * tw / w0 / 2.0)) * 2)
        x = frame.detach().float().cpu().permute(2, 0, 1).unsqueeze(0)
        if w0 > tw:
            x = F.interpolate(x, size=(th, tw), mode="area")
        else:
            x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
        rgb = (x[0] * 255.0).permute(1, 2, 0).contiguous()  # (th,tw,3)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        hh, hw = th // 2, tw // 2
        g = gray[: hh * 2, : hw * 2]
        half = g.reshape(hh, 2, hw, 2).mean(dim=(1, 3))
        self.w, self.h, self.rgb, self.gray, self.half = tw, th, rgb, gray, half


def ssim_gray(a, b):
    """8x8 ブロックの SSIM 平均"""
    h, w = a.shape
    H, W = h // 8 * 8, w // 8 * 8
    if H < 8 or W < 8:
        return 0.5
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blocks(x):
        return x[:H, :W].double().reshape(H // 8, 8, W // 8, 8).permute(0, 2, 1, 3).reshape(-1, 64)

    A, B = blocks(a), blocks(b)
    ma, mb = A.mean(1), B.mean(1)
    va = (A * A).mean(1) - ma * ma
    vb = (B * B).mean(1) - mb * mb
    cov = (A * B).mean(1) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return float(s.mean().clamp(-1.0, 1.0))


def visual_similarity(a, b):
    ssim = max(ssim_gray(a.gray, b.gray), 0.0)
    mad = float((a.rgb - b.rgb).abs().mean())
    color = math.exp(-mad / 20.0)
    return min(max(0.6 * ssim + 0.4 * color, 0.0), 1.0)


def gray_diff(a, b):
    return float((a.gray - b.gray).abs().mean())


# --------------------------------------------------------------------------- #
# 動き推定(ブロックベクトル場)
# --------------------------------------------------------------------------- #
class Flow:
    __slots__ = ("v", "w")

    def __init__(self, v, w):
        self.v, self.w = v, w  # v: (n,2) float, w: (n,)


def estimate_flow(prev, nxt, rng):
    h, w = prev.shape
    if h < 2 * MARGIN + BLOCK or w < 2 * MARGIN + BLOCK or nxt.shape != prev.shape:
        return None
    rng = min(rng, MARGIN)
    ny = (h - 2 * MARGIN - BLOCK) // BLOCK + 1
    nx = (w - 2 * MARGIN - BLOCK) // BLOCK + 1
    ph, pw = ny * BLOCK, nx * BLOCK
    P = prev[MARGIN:MARGIN + ph, MARGIN:MARGIN + pw]

    blk = P.reshape(ny, BLOCK, nx, BLOCK)
    mean = blk.mean(dim=(1, 3))
    var = ((blk * blk).mean(dim=(1, 3)) - mean * mean).clamp(min=0)
    weight = var / (var + 60.0)

    shifts, sads = [], []
    for dy in range(-rng, rng + 1):
        for dx in range(-rng, rng + 1):
            N = nxt[MARGIN + dy:MARGIN + dy + ph, MARGIN + dx:MARGIN + dx + pw]
            sads.append((P - N).abs().reshape(ny, BLOCK, nx, BLOCK).sum(dim=(1, 3)))
            shifts.append((dx, dy))
    S = torch.stack(sads)  # (K,ny,nx)
    s0 = S[shifts.index((0, 0))]
    smin, best = S.min(dim=0)
    # ゼロ動きを基準に、2%以上良くならない限り動かさない(ノイズ抑制)
    use = (smin < 0.98 * s0) & (s0 > 0) & (weight > 0.02)
    dxs = torch.tensor([s[0] for s in shifts], dtype=torch.float32)
    dys = torch.tensor([s[1] for s in shifts], dtype=torch.float32)
    vx = torch.where(use, dxs[best], torch.zeros_like(smin))
    vy = torch.where(use, dys[best], torch.zeros_like(smin))
    v = torch.stack([vx.reshape(-1), vy.reshape(-1)], dim=1)
    return Flow(v, weight.reshape(-1))


def flow_activity(flow):
    if flow is None or flow.w.sum() < 1e-3:
        return 0.0
    w = flow.w.double()
    return float((w * torch.linalg.norm(flow.v.double(), dim=1)).sum() / w.sum())

def motion_scores(tail, head, gap, m, motion_bias=0.5):
    L = min(len(tail.v), len(head.v), len(gap.v))
    w = gap.w[:L].double()
    mask = w >= 1e-3
    if not bool(mask.any()):
        return 1.0, 1.0
    w = w[mask]
    g = gap.v[:L].double()[mask]
    t = tail.v[:L].double()[mask]
    hd = head.v[:L].double()[mask]
    u = g / (m + 1)
    nrm = lambda x: torch.linalg.norm(x, dim=1)
    sw = float(w.sum())
    et = float((w * nrm(u - t)).sum())
    eh = float((w * nrm(u - hd)).sum())
    mt = float((w * nrm(t)).sum())
    mh = float((w * nrm(hd)).sum())
    base = max((mt + mh) / (2.0 * sw), 0.5)
    err = (et + eh) / (2.0 * sw)
    pace = min(max(math.exp(-err / base), 0.0), 1.0)
    activity = (mt + mh) / (2.0 * sw)
    act_score = 1.0 + float(motion_bias) * math.tanh(activity / 3.0)
    return pace, act_score


def seam_real_score(tail_prev, tail, mids, head, head_next):
    """実際に補間したフレーム列での継ぎ目スコア。ジャンプ/停止を減点"""
    base = max((gray_diff(tail_prev, tail) + gray_diff(head, head_next)) / 2.0, 0.8)
    seq = [tail] + list(mids) + [head]
    diffs = [gray_diff(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
    if not diffs:
        return 0.5
    max_d = max(diffs)
    mean_d = sum(diffs) / len(diffs)
    r_max = max_d / base
    r_mean = max(mean_d / base, 0.05)
    jump = math.exp(-max(r_max - 1.25, 0.0) * 0.7)
    stall = math.exp(-abs(math.log(r_mean)) * 0.35)
    return min(max(jump * stall, 0.0), 1.0)


# --------------------------------------------------------------------------- #
# ループ探索
# --------------------------------------------------------------------------- #
def search_loops(frames, interp_fn, max_search_n=30, min_m=1, max_m=8,
                 allow_start_cut=True, top_candidates=3, motion_bias=0.5, progress=None):
    T = int(frames.shape[0])
    if T < 6:
        raise ValueError(f"総フレーム数({T})が少なすぎてループ探索できません(最低6フレーム必要)")
    min_m, max_m = min(min_m, max_m), max(min_m, max_m)
    eff_max_n = min(int(max_search_n), T - 6)

    head_count = min(max(T // 4, 2), 35) if allow_start_cut else 2
    tail_count = min(eff_max_n + 2, T)
    tail_start = T - tail_count + 1

    need = set(range(1, head_count + 1)) | set(range(tail_start, T + 1))
    thumbs = {}
    for i in sorted(need):
        thumbs[i] = Thumb(frames[i - 1])
    check_interrupt()

    s_max = head_count - 1 if allow_start_cut else 1
    head_flow = {}
    for s in range(1, s_max + 1):
        if s in thumbs and (s + 1) in thumbs:
            f = estimate_flow(thumbs[s].half, thumbs[s + 1].half, MOTION_RANGE)
            if f is not None:
                head_flow[s] = f
    tail_flow = {}
    for k in range(tail_start + 1, T + 1):
        if (k - 1) in thumbs and k in thumbs:
            f = estimate_flow(thumbs[k - 1].half, thumbs[k].half, MOTION_RANGE)
            if f is not None:
                tail_flow[k] = f

    tail_act = {k: flow_activity(f) for k, f in tail_flow.items()}
    head_act = {s: flow_activity(f) for s, f in head_flow.items()}

    pairs = []
    for s in range(1, s_max + 1):
        if s not in head_flow:
            continue
        for n in range(0, eff_max_n + 1):
            k = T - n
            if k < s + 4 or k not in tail_flow:
                continue
            vis = visual_similarity(thumbs[k], thumbs[s])
            act = (tail_act.get(k, 0.0) + head_act.get(s, 0.0)) * 0.5
            pair_score = vis * (1.0 + float(motion_bias) * math.tanh(act / 3.0))
            pairs.append((s, k, pair_score, vis))
    if not pairs:
        raise ValueError("ループ候補を探索できませんでした(フレーム数または設定を確認してください)")
    pairs.sort(key=lambda p: -p[2])
    pairs = pairs[:PAIR_LIMIT]
    check_interrupt()

    preds = []
    for s, k, _, vis in pairs:
        gap = estimate_flow(thumbs[k].half, thumbs[s].half, GAP_RANGE)
        if gap is None:
            continue
        n = T - k
        cut_pen = math.exp(-((s - 1) + n) * 0.005)
        best = None
        for m in range(min_m, max_m + 1):
            pace, act_score = motion_scores(tail_flow[k], head_flow[s], gap, m, motion_bias)
            total = vis * pace * act_score * math.exp(-m * 0.03) * cut_pen
            if best is None or total > best["total"]:
                best = dict(s=s, k=k, n=n, m=m, total=total, visual=vis, pace=pace)
        if best:
            preds.append(best)
    preds.sort(key=lambda p: -p["total"])

    # 近傍の重複を間引いて多様性を確保
    target = max(1, min(int(top_candidates), 5))
    pre_k = min(max(target * 2, target), 6)
    selected = []
    for p in preds:
        if not any(abs(q["s"] - p["s"]) <= 1 and abs(q["k"] - p["k"]) <= 1 for q in selected):
            selected.append(p)
        if len(selected) >= pre_k:
            break

    # 上位候補を実際にRIFE補間し、継ぎ目の実フレーム差で再ランキング
    results = []
    for p in selected:
        check_interrupt()
        s, k, m = p["s"], p["k"], p["m"]
        mids = interp_fn(frames[k - 1], frames[s - 1], m)
        mid_thumbs = [Thumb(x) for x in mids]
        real = seam_real_score(thumbs[k - 1], thumbs[k], mid_thumbs, thumbs[s], thumbs[s + 1])
        score = p["total"] * (0.4 + 0.6 * real)
        results.append(dict(
            s=s, k=k, n=p["n"], m=m, score=score, visual=p["visual"],
            motion=min(max(p["pace"] * 0.5 + real * 0.5, 0.0), 1.0), mids=mids,
        ))
        if progress:
            progress(1)
    results.sort(key=lambda r: -r["score"])
    return results[:target]


def assemble_loop(frames, cand):
    """候補から ループ1周分のフレーム列 (L,H,W,3) float32 を作る"""
    base = frames[cand["s"] - 1:cand["k"]].float()
    if cand["mids"]:
        mids = torch.stack([x.float() for x in cand["mids"]])
        return torch.cat([base, mids], dim=0)
    return base


def describe_candidates(cands, chosen, fps):
    lines = []
    for i, c in enumerate(cands, 1):
        L = (c["k"] - c["s"] + 1) + len(c["mids"])
        mark = "★" if i == chosen else "  "
        lines.append(
            f"{mark}#{i}: 開始S={c['s']} 末尾カットN={c['n']} 補間M={c['m']} / {L}フレーム({L / fps:.2f}秒) / "
            f"総合 {c['score']:.3f} (類似 {c['visual']:.3f}, 動き {c['motion']:.3f})"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# フレーム補完 (ループ対応: 末尾→先頭もまたいで補完)
# --------------------------------------------------------------------------- #
def small_gray(frame):
    x = frame.detach().float().cpu().permute(2, 0, 1).unsqueeze(0)
    g = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    return F.interpolate(g.unsqueeze(0), size=(36, 64), mode="area")[0, 0] * 255.0


def hist_corr(a, b):
    ha = torch.histc(a, bins=32, min=0, max=256).double()
    hb = torch.histc(b, bins=32, min=0, max=256).double()
    da, db = ha - ha.mean(), hb - hb.mean()
    den = float(torch.sqrt((da * da).sum() * (db * db).sum()))
    if den <= 0:
        return 1.0
    return float((da * db).sum() / den)


def interpolate_sequence(frames, mult, interp_fn, cyclic=True, dedupe=False, thr=1.0,
                         cut_on=True, cut_thr=CUT_THR, progress=None):
    """
    frames : (N,H,W,3)。出力は常に N×mult 枚。
    cyclic : True なら 最終フレーム→先頭 も補完してループ全体で等間隔にする
    dedupe : 同じ絵が連続するフレーム(2コマ撮り等)を1枚にまとめ、その区間全体をまたいで補完
    cut_on : シーンカットをまたぐ補完を避ける(補完せずホールド)
    戻り値 : (frames_out, stats)
    """
    N = int(frames.shape[0])
    if mult < 2 or N < 2:
        return frames, {"cuts": 0, "dups": 0}

    def is_cut(d, sa, sb):
        return cut_on and d > cut_thr and hist_corr(sa, sb) < CUT_CORR

    out = []
    cur, cur_s, cur_idx = frames[0], small_gray(frames[0]), 0
    cuts = dups = 0
    for idx in range(1, N):
        check_interrupt()
        s = small_gray(frames[idx])
        d = float((s - cur_s).abs().mean())
        if dedupe and d < thr:
            dups += 1
            continue
        n = (idx - cur_idx) * mult - 1
        nxt = frames[idx]
        out.append(cur)
        if is_cut(d, cur_s, s):
            cuts += 1
            out.extend([cur] * n)
        else:
            out.extend(interp_fn(cur, nxt, n))
        cur, cur_s, cur_idx = nxt, s, idx
        if progress:
            progress(1)

    # 末尾の処理
    n = (N - cur_idx) * mult - 1
    out.append(cur)
    if cyclic:
        s0 = small_gray(frames[0])
        d = float((s0 - cur_s).abs().mean())
        if (dedupe and d < thr) or cur_idx == 0:
            out.extend([cur] * n)  # 末尾が先頭と同じ絵 → ホールド
        elif is_cut(d, cur_s, s0):
            cuts += 1
            out.extend([cur] * n)
        else:
            out.extend(interp_fn(cur, frames[0], n))
    else:
        out.extend([cur] * n)
    if progress:
        progress(1)

    res = torch.stack([x.float() for x in out])
    return res, {"cuts": cuts, "dups": dups}


# --------------------------------------------------------------------------- #
# 再生速度の変更
# --------------------------------------------------------------------------- #
def retime(frames, fps, speed, target_fps=0.0, blend=True):
    """
    speed      : 再生速度の倍率 (0.5 = 2倍スロー, 2.0 = 2倍速)
    target_fps : 0 なら フレームはそのままで fps だけ fps×speed にする(実質メタデータの変更)。
                 >0 ならその fps になるよう ループ周期を保ったままフレームを再サンプリングする。
    blend      : 再サンプリング時に隣接2フレームを線形ブレンドするか(False なら最近傍)
    戻り値     : (frames, out_fps)
    """
    N = int(frames.shape[0])
    if target_fps is None or target_fps <= 0:
        return frames, float(fps) * float(speed)
    n_out = max(1, int(round(N * float(target_fps) / (float(fps) * float(speed)))))
    if n_out == N:
        return frames, float(target_fps)
    pos = torch.arange(n_out, dtype=torch.float64) * (N / n_out)  # ループ周期を厳密に保つ
    if not blend:
        idx = torch.round(pos).long() % N
        return frames[idx], float(target_fps)
    i0 = torch.floor(pos).long()
    frac = (pos - i0).float().view(-1, 1, 1, 1)
    a = frames[i0 % N].float()
    b = frames[(i0 + 1) % N].float()
    return a * (1 - frac) + b * frac, float(target_fps)
