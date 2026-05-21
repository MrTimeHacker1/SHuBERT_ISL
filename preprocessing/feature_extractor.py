"""Feature extraction using DINOv2 and pose landmarks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torchvision import transforms


@dataclass
class FeatureConfig:
    dino_model: str = "dinov2_vits14"
    dino_dim: int = 384
    projection_dim: int = 256
    normalize: bool = True
    cache_dir: Path = Path("cache/features")


class FeatureExtractor:
    """Extracts DINOv2 and pose features and projects them to 256 dims."""

    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_dino_model(config.dino_model)
        self.model.eval()
        self.model.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.face_proj, self.pose_proj = self._load_or_create_projectors()

    def extract(
        self, crops: Dict[str, np.ndarray], body_landmarks: np.ndarray, batch_size: int = 64
    ) -> Dict[str, np.ndarray]:
        """Extract and project features for each stream.

        Args:
            crops: Dict with face, left_hand, right_hand crops (RGB).
            body_landmarks: [T, 7, 3] array.
            batch_size: Batch size for DINO inference.

        Returns:
            Dict with projected features per stream and concatenated features.
        """
        face_features = self._extract_dino(crops["face"], batch_size)
        left_features = self._extract_dino(crops["left_hand"], batch_size)
        right_features = self._extract_dino(crops["right_hand"], batch_size)
        body_features = self._extract_body(body_landmarks)

        if self.config.normalize:
            face_features = self._normalize(face_features)
            left_features = self._normalize(left_features)
            right_features = self._normalize(right_features)
            body_features = self._normalize(body_features)

        face_proj = self._project(face_features, self.face_proj)
        left_proj = self._project(left_features, self.face_proj)
        right_proj = self._project(right_features, self.face_proj)
        body_proj = self._project(body_features, self.pose_proj)

        combined = np.concatenate([face_proj, left_proj, right_proj, body_proj], axis=-1)
        return {
            "face": face_proj,
            "left_hand": left_proj,
            "right_hand": right_proj,
            "body": body_proj,
            "combined": combined,
        }

    def _extract_dino(self, crops: np.ndarray, batch_size: int) -> np.ndarray:
        features = []
        for start in range(0, crops.shape[0], batch_size):
            batch = crops[start : start + batch_size]
            tensor = torch.stack([self.transform(frame) for frame in batch]).to(self.device)
            with torch.no_grad():
                feats = self._forward_dino(tensor)
            features.append(feats.cpu().numpy())
        return np.concatenate(features, axis=0)

    def _extract_body(self, landmarks: np.ndarray) -> np.ndarray:
        coords = landmarks[:, :, :2].reshape(landmarks.shape[0], -1)
        return coords.astype(np.float32)

    def _forward_dino(self, tensor: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_features"):
            out = self.model.forward_features(tensor)
            if isinstance(out, dict) and "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]
        output = self.model(tensor)
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def _project(self, features: np.ndarray, projector: nn.Linear) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(features).to(self.device)
            projected = projector(tensor).cpu().numpy()
        return projected

    def _load_dino_model(self, model_name: str) -> nn.Module:
        try:
            return torch.hub.load("facebookresearch/dinov2", model_name)
        except Exception as exc:
            raise RuntimeError("Failed to load DINOv2 model. Ensure internet access or cache.") from exc

    def _load_or_create_projectors(self) -> Tuple[nn.Linear, nn.Linear]:
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        proj_path = self.config.cache_dir / "projectors.pt"
        # DINOv2 ViT-S outputs 384-dim features by default.
        face_proj = nn.Linear(self.config.dino_dim, self.config.projection_dim, bias=True)
        pose_proj = nn.Linear(14, self.config.projection_dim, bias=True)
        if proj_path.exists():
            state = torch.load(proj_path, map_location="cpu")
            face_proj.load_state_dict(state["face"])
            pose_proj.load_state_dict(state["pose"])
        else:
            torch.save({"face": face_proj.state_dict(), "pose": pose_proj.state_dict()}, proj_path)
        face_proj.to(self.device).eval()
        pose_proj.to(self.device).eval()
        return face_proj, pose_proj

    @staticmethod
    def _normalize(features: np.ndarray) -> np.ndarray:
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-6
        return (features - mean) / std
