from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class LayoutParser:

    def __init__(self, row_threshold: float = 0.5, column_gap_threshold: float = 0.3):
        self.row_threshold = row_threshold
        self.column_gap_threshold = column_gap_threshold

    def group_by_row(
        self, items: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        if not items:
            return []

        sorted_items = sorted(items, key=lambda x: x["bbox"][1])

        rows: List[List[Dict[str, Any]]] = []
        current_row: List[Dict[str, Any]] = [sorted_items[0]]

        for item in sorted_items[1:]:
            prev_y_center = self._y_center(current_row[-1]["bbox"])
            curr_y_center = self._y_center(item["bbox"])

            prev_height = self._height(current_row[-1]["bbox"])
            curr_height = self._height(item["bbox"])
            avg_height = (prev_height + curr_height) / 2

            if avg_height > 0 and abs(curr_y_center - prev_y_center) < avg_height * self.row_threshold:
                current_row.append(item)
            else:
                rows.append(sorted(current_row, key=lambda x: x["bbox"][0]))
                current_row = [item]

        if current_row:
            rows.append(sorted(current_row, key=lambda x: x["bbox"][0]))

        return rows

    def group_by_region(
        self, items: List[Dict[str, Any]], image_shape: Tuple[int, int]
    ) -> Dict[str, List[Dict[str, Any]]]:
        h, w = image_shape[:2]
        regions: Dict[str, List[Dict[str, Any]]] = {
            "top_left": [],
            "top_center": [],
            "top_right": [],
            "center_left": [],
            "center": [],
            "center_right": [],
            "bottom_left": [],
            "bottom_center": [],
            "bottom_right": [],
        }

        for item in items:
            region = self._classify_region(item["bbox"], w, h)
            regions[region].append(item)

        return regions

    def get_position_label(
        self, bbox: List[int], image_shape: Tuple[int, int]
    ) -> str:
        h, w = image_shape[:2]
        return self._classify_region(bbox, w, h)

    def _classify_region(self, bbox: List[int], img_w: int, img_h: int) -> str:
        x_min, y_min, x_max, y_max = bbox
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        col = "left" if cx < img_w / 3 else ("right" if cx > 2 * img_w / 3 else "center")
        row = "top" if cy < img_h / 3 else ("bottom" if cy > 2 * img_h / 3 else "center")

        if row == "center" and col == "center":
            return "center"
        return f"{row}_{col}"

    @staticmethod
    def _y_center(bbox: List[int]) -> float:
        return (bbox[1] + bbox[3]) / 2

    @staticmethod
    def _height(bbox: List[int]) -> float:
        return bbox[3] - bbox[1]
