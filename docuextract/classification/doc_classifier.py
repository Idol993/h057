from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docuextract.config import ClassificationConfig


class ClassifierModelNotFoundError(FileNotFoundError):
    """Classification model weight file not found."""
    pass


class ClassifierModelLoadError(RuntimeError):
    """Failed to load classification model weights (format mismatch or corrupted)."""
    pass


class DocumentClassifier:

    def __init__(self, config: Optional[ClassificationConfig] = None):
        self.config = config or ClassificationConfig()
        self.model: Any = None
        self.device: Any = None
        self._loaded = False
        self._torch = None
        self._nn = None
        self._random_initialized = False

    def _ensure_torch(self) -> None:
        if self._torch is not None:
            return
        try:
            import torch
            import torch.nn as nn
            from torchvision.models import mobilenetv3

            self._torch = torch
            self._nn = nn
            self._mobilenetv3 = mobilenetv3
        except ImportError as e:
            raise ImportError(
                "PyTorch and torchvision are required for document classification. "
                "Install them with: pip install torch torchvision"
            ) from e

    def _build_model(self) -> Any:
        self._ensure_torch()
        model = self._mobilenetv3.mobilenet_v3_small(num_classes=len(self.config.class_names))
        return model

    def load_model(self, model_path: Optional[Path] = None) -> None:
        self._ensure_torch()
        torch = self._torch

        self.model = self._build_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        path = model_path or self.config.model_path
        path = Path(path)

        if not path.exists():
            raise ClassifierModelNotFoundError(
                f"Classification model weight file not found: {path}\n"
                f"Please download the pre-trained MobileNetV3-small classification model "
                f"and place it at the above path, or use --model-path to specify its location."
            )

        try:
            try:
                state_dict = torch.load(str(path), map_location=self.device, weights_only=True)
            except TypeError:
                state_dict = torch.load(str(path), map_location=self.device)
        except Exception as e:
            raise ClassifierModelLoadError(
                f"Failed to load classification model from {path}: {e}\n"
                f"This may be caused by:\n"
                f"  1. Corrupted or incomplete model file\n"
                f"  2. Weight format incompatible with current PyTorch version "
                f"(torch {torch.__version__})\n"
                f"  3. Model saved with different architecture\n"
                f"Please verify the model file integrity and version compatibility."
            ) from e

        try:
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif isinstance(state_dict, dict) and "model" in state_dict:
                state_dict = state_dict["model"]

            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            raise ClassifierModelLoadError(
                f"Model architecture mismatch: cannot apply weights to MobileNetV3-small. "
                f"Error: {e}"
            ) from e

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True

    def predict(
        self, image_path: Path
    ) -> Tuple[str, float, Dict[str, float]]:
        if not self._loaded:
            self.load_model()

        torch = self._torch

        from docuextract.classification.preprocessor import preprocess_single

        with torch.no_grad():
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

            tensor = preprocess_single(image_path, self.config).to(self.device)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = probs.max(dim=1)

        class_probs = {
            name: float(probs[0, i].item())
            for i, name in enumerate(self.config.class_names)
        }

        predicted_class = self.config.class_names[int(pred_idx.item())]
        confidence = float(conf.item())

        if confidence < self.config.confidence_threshold:
            predicted_class = "other"

        return predicted_class, confidence, class_probs

    def predict_from_array(
        self, image_array: Any
    ) -> Tuple[str, float, Dict[str, float]]:
        if not self._loaded:
            self.load_model()

        torch = self._torch

        from docuextract.classification.preprocessor import preprocess_array

        with torch.no_grad():
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

            tensor = preprocess_array(image_array, self.config).to(self.device)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = probs.max(dim=1)

        class_probs = {
            name: float(probs[0, i].item())
            for i, name in enumerate(self.config.class_names)
        }

        predicted_class = self.config.class_names[int(pred_idx.item())]
        confidence = float(conf.item())

        if confidence < self.config.confidence_threshold:
            predicted_class = "other"

        return predicted_class, confidence, class_probs

    def predict_batch(
        self, image_paths: List[Path], batch_size: int = 8, num_workers: int = 0
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        if not self._loaded:
            self.load_model()

        torch = self._torch

        from docuextract.classification.preprocessor import create_dataloader

        dataloader = create_dataloader(
            image_paths, self.config, batch_size=batch_size, num_workers=num_workers
        )
        results: List[Tuple[str, float, Dict[str, float]]] = []

        with torch.no_grad():
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

            for batch_tensors, _batch_paths in dataloader:
                batch_tensors = batch_tensors.to(self.device)
                logits = self.model(batch_tensors)
                probs = torch.softmax(logits, dim=1)
                confs, pred_indices = probs.max(dim=1)

                for i in range(len(batch_tensors)):
                    class_probs = {
                        name: float(probs[i, j].item())
                        for j, name in enumerate(self.config.class_names)
                    }
                    predicted_class = self.config.class_names[int(pred_indices[i].item())]
                    confidence = float(confs[i].item())
                    if confidence < self.config.confidence_threshold:
                        predicted_class = "other"
                    results.append((predicted_class, confidence, class_probs))

        return results
