from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from docuextract.config import OCRConfig
from docuextract.ocr_pipeline.detector import OCREngineNotFoundError


class TextRecognizer:

    def __init__(self, config: Optional[OCRConfig] = None, strict: bool = True):
        self.config = config or OCRConfig()
        self._ocr_engine = None
        self._init_failed = False
        self._init_error: Optional[str] = None
        self.strict = strict

    def _get_engine(self, raise_on_missing: bool = False) -> Any:
        if self._ocr_engine is not None:
            return self._ocr_engine

        if self._init_failed:
            if raise_on_missing or self.strict:
                raise OCREngineNotFoundError(
                    f"PaddleOCR engine initialization failed: {self._init_error}\n"
                    f"Please install PaddleOCR with: pip install paddleocr paddlepaddle"
                )
            return None

        try:
            from paddleocr import PaddleOCR

            self._ocr_engine = PaddleOCR(
                det_model_dir=str(self.config.det_model_dir) if self.config.det_model_dir.exists() else None,
                rec_model_dir=str(self.config.rec_model_dir) if self.config.rec_model_dir.exists() else None,
                cls_model_dir=str(self.config.cls_model_dir) if self.config.cls_model_dir.exists() else None,
                use_angle_cls=self.config.use_angle_cls,
                lang=self.config.lang,
                rec_batch_num=self.config.rec_batch_num,
                drop_score=self.config.drop_score,
                show_log=False,
            )
            return self._ocr_engine
        except ImportError as e:
            self._init_failed = True
            self._init_error = f"PaddleOCR not installed: {e}"
        except Exception as e:
            self._init_failed = True
            self._init_error = str(e)

        if raise_on_missing or self.strict:
            raise OCREngineNotFoundError(
                f"PaddleOCR engine is not available: {self._init_error}\n"
                f"Please install PaddleOCR with: pip install paddleocr paddlepaddle"
            )
        return None

    def recognize(
        self, image: np.ndarray
    ) -> List[Dict[str, Any]]:
        engine = self._get_engine(raise_on_missing=True)
        if engine is None:
            return []

        result = engine.ocr(image, det=True, rec=True)
        if result is None or len(result) == 0:
            return []

        texts = []
        for line in result[0]:
            bbox_pts = np.array(line[0]).astype(np.int32)
            text = line[1][0]
            confidence = line[1][1]
            x_min = int(bbox_pts[:, 0].min())
            y_min = int(bbox_pts[:, 1].min())
            x_max = int(bbox_pts[:, 0].max())
            y_max = int(bbox_pts[:, 1].max())
            texts.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "bbox": [x_min, y_min, x_max, y_max],
                }
            )
        return texts

    def recognize_crop(
        self, image: np.ndarray, bbox: List[int]
    ) -> Tuple[str, float]:
        engine = self._get_engine(raise_on_missing=True)
        if engine is None:
            return "", 0.0

        x_min, y_min, x_max, y_max = bbox
        h, w = image.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        if x_max <= x_min or y_max <= y_min:
            return "", 0.0

        crop = image[y_min:y_max, x_min:x_max]

        result = engine.ocr(crop, det=False, rec=True)
        if result is None or len(result) == 0 or len(result[0]) == 0:
            return "", 0.0

        text = result[0][0][0]
        confidence = result[0][0][1]
        return text, confidence

    def is_available(self) -> bool:
        try:
            self._get_engine(raise_on_missing=True)
            return True
        except OCREngineNotFoundError:
            return False

    def get_error_message(self) -> Optional[str]:
        if not self._init_failed:
            return None
        return self._init_error
