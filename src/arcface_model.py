# ArcFace face recognition model (InsightFace w600k_r50)
# Uses ONNX Runtime; downloads weights on first use from InsightFace GitHub releases.

import io
import os
import shutil
import tempfile
import zipfile
from urllib.request import Request, urlopen

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

# buffalo_l.zip on InsightFace GitHub releases (public, no auth)
ZIP_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
ZIP_MEMBER = "w600k_r50.onnx"
MODEL_FILENAME = "w600k_r50.onnx"


def _cache_dir():
    torch_home = os.path.expanduser(
        os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
    )
    return os.path.join(torch_home, "arcface")


def _get_model_path():
    cache = _cache_dir()
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, MODEL_FILENAME)
    if not os.path.exists(path):
        print(f"Downloading ArcFace weights to {path} ...")
        req = Request(ZIP_URL, headers={"User-Agent": "arcface-model-downloader"})
        u = urlopen(req)
        file_size = int(u.info().get("Content-Length", 0)) or None

        buf = io.BytesIO()
        with tqdm(total=file_size, unit="B", unit_scale=True,
                  unit_divisor=1024, desc="Downloading buffalo_l.zip") as pbar:
            while True:
                chunk = u.read(65536)
                if not chunk:
                    break
                buf.write(chunk)
                pbar.update(len(chunk))

        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            # member may be at root or inside a subdirectory
            members = [m for m in zf.namelist() if m.endswith(ZIP_MEMBER)]
            if not members:
                raise RuntimeError(f"{ZIP_MEMBER} not found in {ZIP_URL}")
            with zf.open(members[0]) as src:
                tmp = tempfile.NamedTemporaryFile(delete=False, dir=cache)
                try:
                    shutil.copyfileobj(src, tmp)
                    tmp.close()
                    shutil.move(tmp.name, path)
                finally:
                    if os.path.exists(tmp.name):
                        os.remove(tmp.name)
    return path


class ArcFaceModel:
    """ArcFace face recognition using InsightFace's w600k_r50 ONNX model.

    Preprocessing: RGB PIL image → resize to 112×112 → (x - 127.5) / 128.
    Output: L2-normalised 512-d embeddings as a torch.Tensor on `device`.
    """

    def __init__(self, device):
        import onnxruntime as ort

        self.device = device
        model_path = _get_model_path()

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "cuda" in str(device)
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _preprocess(img):
        img = img.resize((112, 112))
        arr = np.array(img, dtype=np.float32)   # (112, 112, 3) RGB
        arr = (arr - 127.5) / 128.0
        return arr.transpose(2, 0, 1)            # (3, 112, 112)

    def embed_imgs(self, imgs):
        """
        Args:
            imgs: list of PIL Images (any size, RGB)
        Returns:
            torch.Tensor of shape (N, 512), L2-normalised, on self.device
        """
        # Model was exported with static batch=1; run one at a time to avoid shape warnings
        embs = np.stack([
            self.session.run(None, {self.input_name: self._preprocess(img)[None]})[0][0]
            for img in imgs
        ])  # (N, 512)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / (norms + 1e-8)
        return torch.tensor(embs, dtype=torch.float32).to(self.device)
