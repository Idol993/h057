from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docuextract.config import ClassificationConfig


class DocumentClassifier:

    def __init__(self, config: Optional[ClassificationConfig] = None):
        self.config = config or ClassificationConfig()
        self.model: Any = None
        self.device: Any = None
        self._loaded = False
        self._torch = None
        self._nn = None

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
        self.model = self._build_model()
        self.device = self._torch.device("cuda" if self._torch.cuda.is_available() else "cpu")
        path = model_path or self.config.model_path
        if not path.exists():
            self._init_random_weights()
            self._loaded = True
            return
        state_dict = self._torch.load(str(path), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True

    def _init_random_weights(self) -> None:
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self, image_path: Path
    ) -> Tuple[str, float, Dict[str, float]]:
        if not self._loaded:
            self.load_model()

        from docuextract.classification.preprocessor import preprocess_single

        tensor = preprocess_single(image_path, self.config).to(self.device)

        with self._torch.no_grad():
            logits = self.model(tensor)
            probs = self._torch.softmax(logits, dim=1)
            conf, pred_idx = probs.max(dim=1)

        class_probs = {
            name: probs[0, i].item()
            for i, name in enumerate(self.config.class_names)
        }

        predicted_class = self.config.class_names[pred_idx.item()]
        confidence = conf.item()

        if confidence < self.config.confidence_threshold:
            predicted_class = "other"

        return predicted_class, confidence, class_probs

    def predict_batch(
        self, image_paths: List[Path], batch_size: int = 8, num_workers: int = 0
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        if not self._loaded:
            self.load_model()

        from docuextract.classification.preprocessor import create_dataloader

        dataloader = create_dataloader(
            image_paths, self.config, batch_size=batch_size, num_workers=num_workers
        )
        results: List[Tuple[str, float, Dict[str, float]]] = []

        with self._torch.no_grad():
            for batch_tensors, batch_paths in dataloader:
                batch_tensors = batch_tensors.to(self.device)
                logits = self.model(batch_tensors)
                probs = self._torch.softmax(logits, dim=1)
                confs, pred_indices = probs.max(dim=1)

                for i in range(len(batch_paths)):
                    class_probs = {
                        name: probs[i, j].item()
                        for j, name in enumerate(self.config.class_names)
                    }
                    predicted_class = self.config.class_names[pred_indices[i].item()]
                    confidence = confs[i].item()
                    if confidence < self.config.confidence_threshold:
                        predicted_class = "other"
                    results.append((predicted_class, confidence, class_probs))

        return results
