import glob
import os
import shutil
import sys
import threading
import urllib.request
import zipfile
import numpy as np
import torch

_lock = threading.Lock()
_cached_engines = {}

def _download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_dest = dest_path + ".tmp"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(temp_dest, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    shutil.move(temp_dest, dest_path)

def setup_realcugan_resources(scale, noise_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cugan_dir = os.path.join(base_dir, "Real-CUGAN-PyTorch")
    weights_dir = os.path.join(cugan_dir, "weights_v3")
    os.makedirs(weights_dir, exist_ok=True)

    script_path = os.path.join(cugan_dir, "upcunet_v3.py")
    if not os.path.exists(script_path):
        candidate_paths = [
            "/notebooks/0_codes/Real-CUGAN-PyTorch/upcunet_v3.py",
            "/notebooks/Real-CUGAN-PyTorch/upcunet_v3.py",
            "/freespace/ComfyUI/custom_nodes/temp/upcunet_v3.py",
        ]
        found_code = None
        for cp in candidate_paths:
            if os.path.isfile(cp):
                found_code = cp
                break
        if not found_code:
            matches = glob.glob("/notebooks/**/upcunet_v3.py", recursive=True)
            if matches and os.path.isfile(matches[0]):
                found_code = matches[0]

        if found_code:
            shutil.copy2(found_code, script_path)
        else:
            code_url = "https://raw.githubusercontent.com/bilibili/ailab/main/Real-CUGAN/upcunet_v3.py"
            _download_file(code_url, script_path)

    if scale == 2:
        if noise_name in ["denoise3x", "3"]:
            weight_filename = "up2x-latest-denoise3x.pth"
        elif noise_name in ["denoise2x", "2"]:
            weight_filename = "up2x-latest-denoise2x.pth"
        elif noise_name in ["denoise1x", "1"]:
            weight_filename = "up2x-latest-denoise1x.pth"
        elif noise_name in ["no_denoise", "none", "-1"]:
            weight_filename = "up2x-latest-no-denoise.pth"
        else:
            weight_filename = "up2x-latest-conservative.pth"
    elif scale == 3:
        if noise_name in ["denoise3x", "3"]:
            weight_filename = "up3x-latest-denoise3x.pth"
        elif noise_name in ["no_denoise", "none", "-1"]:
            weight_filename = "up3x-latest-no-denoise.pth"
        else:
            weight_filename = "up3x-latest-conservative.pth"
    elif scale == 4:
        if noise_name in ["denoise3x", "3"]:
            weight_filename = "up4x-latest-denoise3x.pth"
        elif noise_name in ["no_denoise", "none", "-1"]:
            weight_filename = "up4x-latest-no-denoise.pth"
        else:
            weight_filename = "up4x-latest-conservative.pth"
    else:
        raise ValueError(f"対応していないスケールです: {scale}")

    model_path = os.path.join(weights_dir, weight_filename)

    if not os.path.exists(model_path):
        search_roots = ["/notebooks", "/freespace", cugan_dir]
        for sr in search_roots:
            if not os.path.exists(sr):
                continue
            hits = glob.glob(os.path.join(sr, "**", weight_filename), recursive=True)
            if hits and os.path.isfile(hits[0]):
                shutil.copy2(hits[0], model_path)
                break

    if not os.path.exists(model_path):
        zip_url = "https://github.com/ZHCSOFT/Real-CUGAN/releases/download/Real-CUGAN/weights_v3.zip"
        zip_path = os.path.join(cugan_dir, "weights_v3.zip")
        try:
            _download_file(zip_url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zr:
                for member in zr.namelist():
                    if member.endswith(".pth"):
                        fname = os.path.basename(member)
                        if fname:
                            tgt = os.path.join(weights_dir, fname)
                            with zr.open(member) as src, open(tgt, "wb") as dst:
                                shutil.copyfileobj(src, dst)
        finally:
            if os.path.exists(zip_path):
                os.remove(zip_path)

    if not os.path.exists(model_path):
        fallback_url = f"https://huggingface.co/JacksonYan/Real-CUGAN/resolve/main/{weight_filename}"
        _download_file(fallback_url, model_path)

    if cugan_dir not in sys.path:
        sys.path.insert(0, cugan_dir)

    return model_path

class TorchWaifuUpScaler:
    def __init__(self, scale, weight_path, half=True, device="cuda:0"):
        self.scale = int(scale)
        self.half = bool(half)
        self.device = torch.device(device)
        import upcunet_v3

        if self.scale == 2:
            model = upcunet_v3.UpCunet2x(in_channels=3, out_channels=3)
        elif self.scale == 3:
            model = upcunet_v3.UpCunet3x(in_channels=3, out_channels=3)
        elif self.scale == 4:
            model = upcunet_v3.UpCunet4x(in_channels=3, out_channels=3)
        else:
            raise ValueError(f"対応していないスケールです: {self.scale}")

        sd = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(sd)
        if self.half:
            model = model.half()
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, frame_np, tile_mode=0, alpha=1.0, cache_mode=0):
        t = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255.0
        if self.half:
            t = t.half()

        _, _, h, w = t.shape
        tile_size = 256 if tile_mode == 2 else (0 if tile_mode == 0 else 512)

        if tile_size <= 0 or (h <= tile_size and w <= tile_size):
            pad_h = (4 - (h % 4)) % 4
            pad_w = (4 - (w % 4)) % 4
            if pad_h > 0 or pad_w > 0:
                t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
            out = self.model(t)[:, :, :h * self.scale, :w * self.scale]
        else:
            pad = 16
            out = torch.zeros((1, 3, h * self.scale, w * self.scale), dtype=t.dtype, device=self.device)
            y = 0
            while y < h:
                x = 0
                while x < w:
                    y1 = min(y + tile_size, h)
                    x1 = min(x + tile_size, w)
                    ty0 = max(0, y - pad)
                    tx0 = max(0, x - pad)
                    ty1 = min(h, y1 + pad)
                    tx1 = min(w, x1 + pad)

                    crop = t[:, :, ty0:ty1, tx0:tx1]
                    ch, cw = crop.shape[2], crop.shape[3]
                    c_pad_h = (4 - (ch % 4)) % 4
                    c_pad_w = (4 - (cw % 4)) % 4
                    if c_pad_h > 0 or c_pad_w > 0:
                        crop = torch.nn.functional.pad(crop, (0, c_pad_w, 0, c_pad_h), mode="reflect")

                    crop_out = self.model(crop)
                    oy0 = (y - ty0) * self.scale
                    ox0 = (x - tx0) * self.scale
                    oy1 = oy0 + (y1 - y) * self.scale
                    ox1 = ox0 + (x1 - x) * self.scale

                    out[:, :, y * self.scale:y1 * self.scale, x * self.scale:x1 * self.scale] = crop_out[:, :, oy0:oy1, ox0:ox1]
                    x += tile_size
                y += tile_size

        res = (out[0].permute(1, 2, 0).float() * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        return res

class CuganEngine:
    def __init__(self, scale=2, denoise_level="conservative", tile=256, fp16=True):
        self.scale = int(scale)
        self.tile = int(tile)
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.cuda else "cpu")
        self.device_str = "cuda:0" if self.cuda else "cpu"
        self.fp16 = bool(fp16 and self.cuda)

        weight_path = setup_realcugan_resources(self.scale, denoise_level)

        try:
            from upcunet_v3 import RealWaifuUpScaler
            self.upscaler = RealWaifuUpScaler(
                scale=self.scale,
                weight_path=weight_path,
                half=self.fp16,
                device=self.device_str
            )
        except (ImportError, AttributeError):
            self.upscaler = TorchWaifuUpScaler(
                scale=self.scale,
                weight_path=weight_path,
                half=self.fp16,
                device=self.device_str
            )

    @torch.inference_mode()
    def enhance_tensor(self, x):
        b, c, h, w = x.shape
        tile_mode = 0 if self.tile <= 0 else (2 if self.tile <= 256 else 1)
        outs = []
        for i in range(b):
            frame_np = (x[i].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            res_np = self.upscaler(frame_np, tile_mode=tile_mode, alpha=1.0, cache_mode=0)
            res_t = torch.from_numpy(res_np).float().div_(255.0).permute(2, 0, 1)
            outs.append(res_t)
        return torch.stack(outs).to(self.device)

def get_cugan_engine(scale=2, denoise_level="conservative", tile=256, fp16=True):
    key = (scale, denoise_level, tile, fp16)
    with _lock:
        if key not in _cached_engines:
            _cached_engines[key] = CuganEngine(scale=scale, denoise_level=denoise_level, tile=tile, fp16=fp16)
        return _cached_engines[key]