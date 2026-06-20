import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from docuextract.config import OutputConfig


class CSVExporter:

    INVOICE_COLUMNS = [
        "file",
        "invoice_code",
        "invoice_number",
        "amount",
        "amount_without_tax",
        "tax_amount",
        "invoice_date",
        "seller_name",
        "buyer_name",
    ]

    ID_CARD_COLUMNS = [
        "file",
        "name",
        "id_number",
        "gender",
        "ethnicity",
        "address",
    ]

    BANK_CARD_COLUMNS = [
        "file",
        "card_number",
        "cardholder_name",
        "expiry_date",
        "bank_name",
        "card_type",
    ]

    COLUMN_MAP = {
        "invoice": INVOICE_COLUMNS,
        "receipt": INVOICE_COLUMNS,
        "id_card": ID_CARD_COLUMNS,
        "bank_card": BANK_CARD_COLUMNS,
    }

    def __init__(self, config: Optional[OutputConfig] = None):
        self.config = config or OutputConfig()

    def export(
        self,
        reports: List[Dict[str, Any]],
        output_path: Optional[Path] = None,
        doc_type: Optional[str] = None,
    ) -> Path:
        if output_path is None:
            timestamp = datetime.now().strftime(self.config.timestamp_format)
            output_path = Path(f"extraction_{timestamp}.csv")

        resolved_type = doc_type
        if not resolved_type and reports:
            resolved_type = reports[0].get("type", "invoice")

        columns = self.COLUMN_MAP.get(resolved_type, self.INVOICE_COLUMNS)

        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(columns)

            for report in reports:
                row = self._build_row(report, columns)
                writer.writerow(row)

        return output_path

    def _build_row(self, report: Dict[str, Any], columns: List[str]) -> List[str]:
        row: List[str] = []
        fields = report.get("fields", {})

        for col in columns:
            if col == "file":
                row.append(report.get("file", ""))
            else:
                field = fields.get(col, {})
                value = field.get("value", "") if field else ""
                row.append(str(value) if value is not None else "")

        return row

    def export_all_types(
        self,
        reports: List[Dict[str, Any]],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        if output_dir is None:
            output_dir = Path(".")

        type_reports: Dict[str, List[Dict[str, Any]]] = {}
        for report in reports:
            doc_type = report.get("type", "other")
            if doc_type not in type_reports:
                type_reports[doc_type] = []
            type_reports[doc_type].append(report)

        exported: Dict[str, Path] = {}
        timestamp = datetime.now().strftime(self.config.timestamp_format)

        for doc_type, type_report_list in type_reports.items():
            if doc_type in self.COLUMN_MAP:
                output_path = output_dir / f"extraction_{doc_type}_{timestamp}.csv"
                self.export(type_report_list, output_path, doc_type)
                exported[doc_type] = output_path

        return exported
