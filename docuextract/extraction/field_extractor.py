import re
from typing import Any, Dict, List, Optional, Tuple

from docuextract.config import ExtractionConfig
from docuextract.extraction.template_matcher import TemplateMatcher
from docuextract.ocr_pipeline.layout_parser import LayoutParser


def _edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


class FieldExtractor:

    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        template_matcher: Optional[TemplateMatcher] = None,
        layout_parser: Optional[LayoutParser] = None,
    ):
        self.config = config or ExtractionConfig()
        self.template_matcher = template_matcher or TemplateMatcher(self.config)
        self.layout_parser = layout_parser or LayoutParser()

    def extract(
        self,
        doc_type: str,
        ocr_items: List[Dict[str, Any]],
        image_shape: Tuple[int, int],
    ) -> Dict[str, Dict[str, Any]]:
        fields = self.template_matcher.get_fields(doc_type)
        regions = self.layout_parser.group_by_region(ocr_items, image_shape)
        rows = self.layout_parser.group_by_row(ocr_items)

        extracted: Dict[str, Dict[str, Any]] = {}

        for field_def in fields:
            field_name = field_def["name"]
            keywords = field_def.get("keywords", [])
            position = field_def.get("position", "")
            value_pattern = field_def.get("value_pattern", "")
            required = field_def.get("required", False)

            match_result = self._find_field(
                keywords=keywords,
                position=position,
                value_pattern=value_pattern,
                ocr_items=ocr_items,
                regions=regions,
                rows=rows,
                image_shape=image_shape,
            )

            if match_result:
                extracted[field_name] = {
                    "value": match_result["value"],
                    "confidence": match_result["confidence"],
                    "bbox": match_result["bbox"],
                    "matched_keyword": match_result.get("matched_keyword", ""),
                    "required": required,
                }
            else:
                extracted[field_name] = {
                    "value": None,
                    "confidence": 0.0,
                    "bbox": None,
                    "matched_keyword": "",
                    "required": required,
                }

        return extracted

    def _find_field(
        self,
        keywords: List[str],
        position: str,
        value_pattern: str,
        ocr_items: List[Dict[str, Any]],
        regions: Dict[str, List[Dict[str, Any]]],
        rows: List[List[Dict[str, Any]]],
        image_shape: Tuple[int, int],
    ) -> Optional[Dict[str, Any]]:
        threshold = self.config.edit_distance_threshold
        tolerance = self.config.position_tolerance

        candidates: List[Dict[str, Any]] = []

        for item in ocr_items:
            text = item.get("text", "")
            confidence = item.get("confidence", 0.0)
            bbox = item.get("bbox", [0, 0, 0, 0])

            for keyword in keywords:
                dist = _edit_distance(text, keyword)
                max_dist = min(threshold, len(keyword) // 3 + 1)

                if dist <= max_dist or keyword in text:
                    value = self._extract_value_near_keyword(
                        keyword=keyword,
                        keyword_item=item,
                        ocr_items=ocr_items,
                        rows=rows,
                        value_pattern=value_pattern,
                    )

                    if position:
                        pos_label = self.layout_parser.get_position_label(bbox, image_shape)
                        if not self._position_matches(pos_label, position, tolerance):
                            if value:
                                candidates.append(
                                    {
                                        "value": value,
                                        "confidence": confidence * 0.7,
                                        "bbox": bbox,
                                        "matched_keyword": keyword,
                                    }
                                )
                            continue

                    if value:
                        candidates.append(
                            {
                                "value": value,
                                "confidence": confidence,
                                "bbox": bbox,
                                "matched_keyword": keyword,
                            }
                        )
                    break

        if not candidates:
            return None

        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        return candidates[0]

    def _extract_value_near_keyword(
        self,
        keyword: str,
        keyword_item: Dict[str, Any],
        ocr_items: List[Dict[str, Any]],
        rows: List[List[Dict[str, Any]]],
        value_pattern: str,
    ) -> Optional[str]:
        keyword_bbox = keyword_item.get("bbox", [0, 0, 0, 0])
        keyword_text = keyword_item.get("text", "")

        value_in_same_text = self._try_extract_from_text(keyword_text, keyword, value_pattern)
        if value_in_same_text:
            return value_in_same_text

        for row in rows:
            if any(item is keyword_item for item in row):
                for item in row:
                    if item is keyword_item:
                        continue
                    text = item.get("text", "")
                    val = self._try_match_pattern(text, value_pattern)
                    if val:
                        return val

                for item in row:
                    if item is keyword_item:
                        continue
                    text = item.get("text", "")
                    clean = text.strip()
                    if clean and clean != keyword:
                        return clean
                break

        kw_cx = (keyword_bbox[0] + keyword_bbox[2]) / 2
        kw_cy = (keyword_bbox[1] + keyword_bbox[3]) / 2

        nearby = []
        for item in ocr_items:
            if item is keyword_item:
                continue
            item_bbox = item.get("bbox", [0, 0, 0, 0])
            item_cx = (item_bbox[0] + item_bbox[2]) / 2
            item_cy = (item_bbox[1] + item_bbox[3]) / 2

            dist = ((item_cx - kw_cx) ** 2 + (item_cy - kw_cy) ** 2) ** 0.5
            text = item.get("text", "")
            val = self._try_match_pattern(text, value_pattern)
            if val:
                nearby.append((dist, val))

        if nearby:
            nearby.sort(key=lambda x: x[0])
            return nearby[0][1]

        return None

    def _try_extract_from_text(
        self, text: str, keyword: str, value_pattern: str
    ) -> Optional[str]:
        idx = text.find(keyword)
        if idx < 0:
            kw_end = 0
            for kw_char in keyword:
                pos = text.find(kw_char, kw_end)
                if pos < 0:
                    return None
                kw_end = pos + 1
            remaining = text[kw_end:].strip()
        else:
            remaining = text[idx + len(keyword) :].strip()

        if not remaining:
            return None

        return self._try_match_pattern(remaining, value_pattern)

    @staticmethod
    def _try_match_pattern(text: str, value_pattern: str) -> Optional[str]:
        if not value_pattern or not text:
            return None
        try:
            match = re.search(value_pattern, text)
            if match:
                return match.group()
        except re.error:
            pass
        return None

    @staticmethod
    def _position_matches(actual: str, expected: str, tolerance: float) -> bool:
        if not expected or not actual:
            return True

        actual_parts = set(actual.split("_"))
        expected_parts = set(expected.split("_"))

        overlap = actual_parts & expected_parts
        total = expected_parts

        if not total:
            return True

        return len(overlap) / len(total) >= (1 - tolerance)
