from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from docuextract.config import OCRConfig


class TextRecognizer:

    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self._ocr_engine = None

    def _get_engine(self) -> Any:
        if self._ocr_engine is not None:
            return self._ocr_engine

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
        except ImportError:
            self._ocr_engine = None

        return self._ocr_engine

    def recognize(
        self, image: np.ndarray
    ) -> List[Dict[str, Any]]:
        engine = self._get_engine()
        if engine is None:
            return self._fallback_recognize(image)

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
        x_min, y_min, x_max, y_max = bbox
        h, w = image.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        if x_max <= x_min or y_max <= y_min:
            return "", 0.0

        crop = image[y_min:y_max, x_min:x_max]

        engine = self._get_engine()
        if engine is None:
            return self._fallback_recognize_crop(crop)

        result = engine.ocr(crop, det=False, rec=True)
        if result is None or len(result) == 0 or len(result[0]) == 0:
            return "", 0.0

        text = result[0][0][0]
        confidence = result[0][0][1]
        return text, confidence

    def _fallback_recognize(self, image: np.ndarray) -> List[Dict[str, Any]]:
        h, w = image.shape[:2]
        return [
            {
                "text": "",
                "confidence": 0.0,
                "bbox": [0, 0, w, h],
            }
        ]

    def _fallback_recognize_crop(self, crop: np.ndarray) -> Tuple[str, float]:
        return "", 0.0
