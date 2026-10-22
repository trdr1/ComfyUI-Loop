# -*- coding: utf-8 -*-
"""
Practical-RIFE を ComfyUI 内から使うためのバックエンド。
 - コード/学習済みモデルが無ければ初回に自動取得 (rife_data/ 以下)
 - Practical-RIFE の `model` / `train_log` パッケージ名が他のカスタムノードと衝突しないよう、
   import の間だけ sys.modules を退避する
 - 入出力は torch テンソル (H,W,3 / float 0..1)
環境変数 AUTOLOOP_RIFE_DIR で Practical-RIFE の置き場所を変更できます。
"""
import glob
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile

import torch
import torch.nn.functional as F

RIFE_REPO_ZIP = "https://github.com/hzwer/Practical-RIFE/archive/refs/heads/main.zip"
# 別バージョンにしたい場合は環境変数 AUTOLOOP_RIFE_MODEL_URL で差し替え(任意時刻補完対応の v4 系のみ)
RIFE_MODEL_URL = os.environ.get(
    "AUTOLOOP_RIFE_MODEL_URL",
    "https://huggingface.co/hzwer/RIFE/resolve/main/RIFEv4.26_0921.zip",
)
DEFAULT_DIR = os.environ.get("AUTOLOOP_RIFE_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "rife_data", "Practical-RIFE"
)

_lock = threading.Lock()
_engine = None


def _download(url, dst):
    print(f"[AutoLoop] ダウンロード中: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        last = 0.0
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 3:
                last = time.time()
                print(f"[AutoLoop]   {done / 1e6:.0f} MB")


def ensure_rife(rife_dir=DEFAULT_DIR, model_url=RIFE_MODEL_URL):
    rife_dir = os.path.abspath(rife_dir)
    train_log = os.path.join(rife_dir, "train_log")
    need_code = not os.path.exists(os.path.join(rife_dir, "model", "warplayer.py"))
    need_model = not (
        os.path.exists(os.path.join(train_log, "flownet.pkl"))
        and os.path.exists(os.path.join(train_log, "RIFE_HDv3.py"))
    )
    if not (need_code or need_model):
        return rife_dir
    os.makedirs(rife_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        if need_code:
            zp = os.path.join(tmp, "repo.zip")
            _download(RIFE_REPO_ZIP, zp)
            with zipfile.ZipFile(zp) as z:
                z.extractall(os.path.join(tmp, "repo"))
            top = glob.glob(os.path.join(tmp, "repo", "*"))[0]
            shutil.copytree(top, rife_dir, dirs_exist_ok=True)
        if need_model:
            zp = os.path.join(tmp, "model.zip")
            _download(model_url, zp)
            with zipfile.ZipFile(zp) as z:
                z.extractall(os.path.join(tmp, "model"))
            hits = glob.glob(os.path.join(tmp, "model", "**", "flownet.pkl"), recursive=True)
            if not hits:
                raise RuntimeError("ダウンロードしたzipに flownet.pkl が見つかりませんでした")
            src = os.path.dirname(hits[0])
            os.makedirs(train_log, exist_ok=True)
            for name in os.listdir(src):
                if name == "flownet.pkl" or name.endswith(".py"):
                    shutil.copy2(os.path.join(src, name), os.path.join(train_log, name))
    print(f"[AutoLoop] RIFE の準備完了: {rife_dir}")
    return rife_dir


class _IsolatedImport:
    """import 中だけ `model` / `train_log` を Practical-RIFE のものにする"""

    NAMES = ("model", "train_log")

    def __init__(self, path):
        self.path = path
        self.saved = {}

    def __enter__(self):
        self.saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] in self.NAMES}
        for k in self.saved:
            del sys.modules[k]
        sys.path.insert(0, self.path)
        return self

    def __exit__(self, *exc):
        for k in [k for k in sys.modules if k.split(".")[0] in self.NAMES]:
            del sys.modules[k]
        sys.modules.update(self.saved)
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


class RifeEngine:
    def __init__(self, rife_dir=DEFAULT_DIR):
        rife_dir = ensure_rife(rife_dir)
        with _IsolatedImport(rife_dir):
            from train_log.RIFE_HDv3 import Model  # noqa

        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        model = Model()
        if not hasattr(model, "version"):
            model.version = 0
        model.load_model(os.path.join(rife_dir, "train_log"), -1)
        model.eval()
        model.device()
        if model.version < 3.9:
            raise RuntimeError("任意時刻補完に対応した RIFE v4 系のモデルを使ってください")
        self.model = model

    @torch.inference_mode()
    def interpolate(self, a, b, n, scale=1.0, fp16=True):
        """a,b: (H,W,3) float 0..1 のテンソル。a→b の間を n 枚 (時刻 i/(n+1)) 生成。CPU float16 のリストを返す"""
        if n <= 0:
            return []
        h, w = a.shape[:2]
        tmp = max(128, int(128 / scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        pad = (0, pw - w, 0, ph - h)

        def prep(x):
            t = x.to(self.device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            return F.pad(t, pad)

        out = []
        use16 = bool(fp16 and self.cuda)
        with torch.autocast(device_type="cuda" if self.cuda else "cpu", dtype=torch.float16, enabled=use16):
            t0, t1 = prep(a), prep(b)
            for i in range(1, n + 1):
                mid = self.model.inference(t0, t1, i / (n + 1), scale)
                img = mid[0].float().clamp(0, 1).permute(1, 2, 0)[:h, :w]
                out.append(img.to("cpu", dtype=torch.float16))
        return out


def get_engine(rife_dir=DEFAULT_DIR):
    global _engine
    with _lock:
        if _engine is None:
            try:
                _engine = RifeEngine(rife_dir)
            except Exception as e:
                raise RuntimeError(f"RIFE の読み込みに失敗しました: {e}") from e
        return _engine


def release_engine():
    global _engine
    with _lock:
        _engine = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
