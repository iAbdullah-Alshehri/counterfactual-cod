"""Appearance, learned-feature, and structural residual maps."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


class ResNet18FeatureDistance:
    """Frozen ImageNet ResNet-18 layer2 cosine distance, expanded to image size."""

    def __init__(self, weights_path: str | None = None) -> None:
        try:
            import torch
            import torch.nn.functional as functional
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise RuntimeError("Learned features require PyTorch and Torchvision; install with `pip install -e .[model]`.") from exc

        self.torch = torch
        self.functional = functional
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        checkpoint_path = Path(weights_path).resolve() if weights_path else (
            Path(__file__).resolve().parents[2] / "data" / "downloads" / "resnet18-f37072fd.pth")
        checkpoint = str(checkpoint_path) if checkpoint_path.is_file() else None
        if checkpoint is None:
            project_data = Path(__file__).resolve().parents[2] / "data" / "downloads" / "torch_hub"
            torch.hub.set_dir(str(project_data))
            weights = ResNet18_Weights.IMAGENET1K_V1
            state = weights.get_state_dict(progress=True, check_hash=True)
            checkpoint = str(project_data / "checkpoints" / "resnet18-f37072fd.pth")
        else:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = resnet18(weights=None)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        stamp = ""
        if checkpoint:
            stat = Path(checkpoint).stat()
            stamp = f":{stat.st_size}:{stat.st_mtime_ns}"
        self.cache_signature = f"resnet18-imagenet1k-v1-layer2-cosine:{checkpoint}{stamp}"

    def __call__(self, image_rgb: np.ndarray, background_rgb: np.ndarray) -> np.ndarray:
        torch, functional = self.torch, self.functional
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)

        def tensor(image: np.ndarray):
            x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            x = functional.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            return (x - mean) / std

        with torch.inference_mode():
            a, b = tensor(image_rgb), tensor(background_rgb)
            def layer2_features(x):
                x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
                x = self.model.layer1(x)
                return self.model.layer2(x)
            fa, fb = layer2_features(a), layer2_features(b)
            distance = (1.0 - functional.cosine_similarity(fa, fb, dim=1, eps=1e-8)) * 0.5
            distance = functional.interpolate(distance.unsqueeze(1), size=image_rgb.shape[:2], mode="bilinear", align_corners=False)
        return distance[0, 0].cpu().numpy().clip(0.0, 1.0).astype(np.float32)


def _bounded(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Keep residual magnitude on its defined [0,1] scale across images."""
    out = np.zeros(values.shape, dtype=np.float32)
    out[valid] = np.clip(values[valid], 0, 1)
    return out


def residual_maps(image_rgb: np.ndarray, background_rgb: np.ndarray,
                  candidate: np.ndarray,
                  feature_extractor: ResNet18FeatureDistance) -> dict[str, np.ndarray]:
    """Return spatial residual maps normalized over the candidate support."""
    support = candidate >= 0.5
    if not support.any():
        zeros = np.zeros(candidate.shape, dtype=np.float32)
        return {"appearance": zeros.copy(), "feature": zeros.copy(), "structural": zeros.copy()}

    original = image_rgb.astype(np.float32) / 255.0
    reconstructed = background_rgb.astype(np.float32) / 255.0
    appearance_raw = np.abs(original - reconstructed).mean(axis=2)

    feature_raw = feature_extractor(image_rgb, background_rgb)

    gray_a = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(background_rgb, cv2.COLOR_RGB2GRAY)
    edges_a = cv2.Canny(gray_a, 80, 160).astype(np.float32) / 255.0
    edges_b = cv2.Canny(gray_b, 80, 160).astype(np.float32) / 255.0
    structural_raw = np.abs(edges_a - edges_b)

    return {
        "appearance": _bounded(appearance_raw, support),
        "feature": _bounded(feature_raw, support),
        "structural": _bounded(structural_raw, support),
    }


def combine_residuals(maps: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = sum(weights.values())
    if total <= 0 or any(value < 0 for value in weights.values()):
        raise ValueError("Residual weights must be nonnegative with positive total")
    return sum(maps[name] * weight for name, weight in weights.items()) / total
