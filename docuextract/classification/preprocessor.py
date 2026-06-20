from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image

from docuextract.config import ClassificationConfig


def _get_torch_modules():
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms
        return torch, Dataset, DataLoader, transforms
    except ImportError as e:
        raise ImportError(
            "PyTorch and torchvision are required for document classification. "
            "Install them with: pip install torch torchvision"
        ) from e


class DocumentImageDataset:

    def __init__(
        self,
        image_paths: List[Path],
        config: Optional[ClassificationConfig] = None,
    ):
        torch, Dataset, _DataLoader, transforms = _get_torch_modules()
        self.image_paths = image_paths
        self.config = config or ClassificationConfig()
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.config.input_size, self.config.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.mean,
                    std=self.config.std,
                ),
            ]
        )
        self._Dataset = Dataset
        self._torch = torch

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, str(path)


def preprocess_single(
    image_path: Path,
    config: Optional[ClassificationConfig] = None,
) -> Any:
    torch, _Dataset, _DataLoader, transforms = _get_torch_modules()
    cfg = config or ClassificationConfig()
    transform = transforms.Compose(
        [
            transforms.Resize((cfg.input_size, cfg.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


def create_dataloader(
    image_paths: List[Path],
    config: Optional[ClassificationConfig] = None,
    batch_size: int = 8,
    num_workers: int = 0,
) -> Any:
    _, _, DataLoader, _ = _get_torch_modules()
    dataset = DocumentImageDataset(image_paths, config)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def load_image_as_array(
    image_path: Path, target_size: Optional[int] = None
) -> np.ndarray:
    size = target_size or 224
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size))
    return np.array(img)
