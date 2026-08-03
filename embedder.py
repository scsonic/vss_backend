"""CLIP embedder：把影格(image)與查詢(text)投影到同一個語意空間。

對應 NVIDIA VSS 的 TensorRT visual encoder，這裡改用 OpenCLIP 跑在 Apple Silicon (MPS)。
"""
from concurrent.futures import ThreadPoolExecutor

import torch
import open_clip
from PIL import Image
from tqdm import tqdm

import config


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ClipEmbedder:
    def __init__(self):
        self.device = _pick_device()
        print(f"[embedder] loading {config.CLIP_MODEL} ({config.CLIP_PRETRAINED}) on {self.device} ...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL)

    @torch.no_grad()
    def embed_image(self, path: str) -> list[float]:
        img = Image.open(path).convert("RGB")
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].cpu().float().tolist()

    @torch.no_grad()
    def embed_images(self, paths: list[str], batch_size: int = None,
                     num_workers: int = 8) -> list[list[float]]:
        """批次嵌入多張影格：CPU 端用多執行緒並行做 JPEG 解碼+前處理，
        再一次丟一批到 GPU(MPS) forward，大幅提高吞吐量。"""
        batch_size = batch_size or config.EMBED_BATCH_SIZE
        out: list[list[float]] = []
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for i in tqdm(range(0, len(paths), batch_size), desc="embedding", unit="batch"):
                chunk = paths[i:i + batch_size]
                tensors = list(pool.map(
                    lambda p: self.preprocess(Image.open(p).convert("RGB")), chunk
                ))
                batch = torch.stack(tensors).to(self.device)
                feat = self.model.encode_image(batch)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                out.extend(feat.cpu().float().tolist())
        return out

    @torch.no_grad()
    def embed_text(self, text: str) -> list[float]:
        toks = self.tokenizer([text]).to(self.device)
        feat = self.model.encode_text(toks)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0].cpu().float().tolist()
