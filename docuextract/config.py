import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

_BASE_DIR = Path(__file__).resolve().parent
_DATA_DIR = _BASE_DIR / "data"
_TEMPLATES_DIR = _DATA_DIR / "templates"
_MODELS_DIR = _DATA_DIR / "models"


@dataclass
class ClassificationConfig:
    model_path: Path = _MODELS_DIR / "doc_classifier_mobilenetv3.pth"
    input_size: int = 224
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    class_names: List[str] = field(
        default_factory=lambda: ["invoice", "id_card", "bank_card", "receipt", "other"]
    )
    confidence_threshold: float = 0.6


@dataclass
class OCRConfig:
    det_model_dir: Path = _MODELS_DIR / "ch_PP-OCRv3_det_infer"
    rec_model_dir: Path = _MODELS_DIR / "ch_PP-OCRv3_rec_infer"
    cls_model_dir: Path = _MODELS_DIR / "ch_ppocr_mobile_v2.0_cls_infer"
    use_angle_cls: bool = True
    lang: str = "ch"
    det_limit_side: int = 960
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.5
    det_db_unclip_ratio: float = 1.6
    rec_batch_num: int = 6
    drop_score: float = 0.5


@dataclass
class ExtractionConfig:
    templates_dir: Path = _TEMPLATES_DIR
    edit_distance_threshold: int = 2
    position_tolerance: float = 0.15


@dataclass
class ValidationConfig:
    id_card_check_digit: bool = True
    invoice_amount_tolerance: float = 0.01
    bank_card_luhn: bool = True


@dataclass
class OutputConfig:
    default_format: str = "json"
    timestamp_format: str = "%Y%m%d_%H%M%S"
    json_indent: int = 2
    ensure_ascii: bool = False


@dataclass
class AppConfig:
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    models_dir: Path = _MODELS_DIR
    templates_dir: Path = _TEMPLATES_DIR


def get_config() -> AppConfig:
    return AppConfig()
