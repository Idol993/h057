import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from docuextract.config import OutputConfig


class JSONReporter:

    def __init__(self, config: Optional[OutputConfig] = None):
        self.config = config or OutputConfig()

    def generate_report(
        self,
        file_path: str,
        doc_type: str,
        classification_confidence: float,
        fields: Dict[str, Dict[str, Any]],
        validation: Dict[str, Dict[str, Any]],
        ocr_items: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "file": file_path,
            "type": doc_type,
            "classification_confidence": round(classification_confidence, 4),
            "fields": {},
            "validation": {},
        }

        for field_name, field_data in fields.items():
            report["fields"][field_name] = {
                "value": field_data.get("value"),
                "confidence": round(field_data.get("confidence", 0.0), 4),
            }

        for rule_name, rule_data in validation.items():
            report["validation"][rule_name] = {
                "pass": rule_data.get("pass", False),
                "reason": rule_data.get("reason", ""),
            }

        if ocr_items is not None:
            report["ocr_text_count"] = len(ocr_items)

        return report

    def generate_batch_report(
        self, reports: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        total = len(reports)
        passed = sum(
            1
            for r in reports
            if all(v.get("pass", False) for v in r.get("validation", {}).values())
        )

        type_counts: Dict[str, int] = {}
        for r in reports:
            t = r.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "summary": {
                "total_files": total,
                "validation_passed": passed,
                "validation_failed": total - passed,
                "type_distribution": type_counts,
            },
            "results": reports,
        }

    def save_report(
        self,
        report: Dict[str, Any],
        output_path: Optional[Path] = None,
    ) -> Path:
        if output_path is None:
            timestamp = datetime.now().strftime(self.config.timestamp_format)
            output_path = Path(f"extraction_{timestamp}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                report,
                f,
                indent=self.config.json_indent,
                ensure_ascii=self.config.ensure_ascii,
            )

        return output_path

    def format_to_console(self, report: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append(f"File: {report['file']}")
        lines.append(f"Type: {report['type']} (confidence: {report['classification_confidence']})")
        lines.append("")
        lines.append("Fields:")

        for field_name, field_data in report.get("fields", {}).items():
            value = field_data.get("value", "N/A")
            conf = field_data.get("confidence", 0.0)
            lines.append(f"  {field_name}: {value} (confidence: {conf})")

        lines.append("")
        lines.append("Validation:")

        for rule_name, rule_data in report.get("validation", {}).items():
            passed = rule_data.get("pass", False)
            reason = rule_data.get("reason", "")
            status = "PASS" if passed else "FAIL"
            if reason:
                lines.append(f"  {rule_name}: {status} - {reason}")
            else:
                lines.append(f"  {rule_name}: {status}")

        return "\n".join(lines)
