from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from docuextract.config import OCRConfig


class OCREngineNotFoundError(RuntimeError):
    """PaddleOCR engine is not available (not installed or initialization failed)."""
    pass


class TextDetector:

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
                det_limit_side=self.config.det_limit_side,
                det_db_thresh=self.config.det_db_thresh,
                det_db_box_thresh=self.config.det_db_box_thresh,
                det_db_unclip_ratio=self.config.det_db_unclip_ratio,
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

    def detect(
        self, image: np.ndarray
    ) -> List[Dict[str, Any]]:
        engine = self._get_engine()
        if engine is None:
            return []

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
        engine = self._get_engine(raise_on_missing=True)
        if engine is None:
            return []

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
