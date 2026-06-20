from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from docuextract.config import OCRConfig


class TextDetector:

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
                det_limit_side=self.config.det_limit_side,
                det_db_thresh=self.config.det_db_thresh,
                det_db_box_thresh=self.config.det_db_box_thresh,
                det_db_unclip_ratio=self.config.det_db_unclip_ratio,
                rec_batch_num=self.config.rec_batch_num,
                drop_score=self.config.drop_score,
                show_log=False,
            )
        except ImportError:
            self._ocr_engine = None

        return self._ocr_engine

    def detect(
        self, image: np.ndarray
    ) -> List[Dict[str, Any]]:
        engine = self._get_engine()
        if engine is None:
            return self._fallback_detect(image)

        result = engine.ocr(image, det=True, rec=False)
        if result is None or len(result) == 0:
            return []

        boxes = []
        for line in result[0]:
            bbox = np.array(line).astype(np.int32)
            x_min = int(bbox[:, 0].min())
            y_min = int(bbox[:, 1].min())
            x_max = int(bbox[:, 0].max())
            y_max = int(bbox[:, 1].max())
            boxes.append(
                {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "polygon": bbox.tolist(),
                }
            )
        return boxes

    def detect_with_text(
        self, image: np.ndarray
    ) -> List[Dict[str, Any]]:
        engine = self._get_engine()
        if engine is None:
            return self._fallback_detect_with_text(image)

        result = engine.ocr(image, det=True, rec=True)
        if result is None or len(result) == 0:
            return []

        items = []
        for line in result[0]:
            bbox_pts = np.array(line[0]).astype(np.int32)
            text = line[1][0]
            confidence = line[1][1]
            x_min = int(bbox_pts[:, 0].min())
            y_min = int(bbox_pts[:, 1].min())
            x_max = int(bbox_pts[:, 0].max())
            y_max = int(bbox_pts[:, 1].max())
            items.append(
                {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "polygon": bbox_pts.tolist(),
                    "text": text,
                    "confidence": confidence,
                }
            )
        return items

    def _fallback_detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        h, w = image.shape[:2]
        return [
            {
                "bbox": [0, 0, w, h],
                "polygon": [[0, 0], [w, 0], [w, h], [0, h]],
            }
        ]

    def _fallback_detect_with_text(
        self, image: np.ndarray
    ) -> List[Dict[str, Any]]:
        h, w = image.shape[:2]
        return [
            {
                "bbox": [0, 0, w, h],
                "polygon": [[0, 0], [w, 0], [w, h], [0, h]],
                "text": "",
                "confidence": 0.0,
            }
        ]
