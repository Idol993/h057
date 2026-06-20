import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from docuextract.config import ExtractionConfig


class TemplateMatcher:

    BUILTIN_TEMPLATES = {
        "invoice": "invoice.json",
        "id_card": "id_card.json",
        "bank_card": "bank_card.json",
        "receipt": "invoice.json",
    }

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self._cache: Dict[str, Dict[str, Any]] = {}

    def load_template(self, doc_type: str) -> Dict[str, Any]:
        if doc_type in self._cache:
            return self._cache[doc_type]

        template_filename = self.BUILTIN_TEMPLATES.get(doc_type, f"{doc_type}.json")
        template_path = self.config.templates_dir / template_filename

        if not template_path.exists():
            raise FileNotFoundError(
                f"Template not found for doc_type '{doc_type}': {template_path}"
            )

        with open(template_path, "r", encoding="utf-8") as f:
            template = json.load(f)

        self._cache[doc_type] = template
        return template

    def get_fields(self, doc_type: str) -> List[Dict[str, Any]]:
        template = self.load_template(doc_type)
        return template.get("fields", [])

    def get_validation_rules(self, doc_type: str) -> List[Dict[str, Any]]:
        template = self.load_template(doc_type)
        return template.get("validation_rules", [])

    def add_template(self, template_path: Path, doc_type: Optional[str] = None) -> None:
        with open(template_path, "r", encoding="utf-8") as f:
            template = json.load(f)

        resolved_type = doc_type or template.get("doc_type")
        if not resolved_type:
            raise ValueError("Template must have 'doc_type' field or doc_type must be provided")

        dest = self.config.templates_dir / f"{resolved_type}.json"
        shutil.copy2(str(template_path), str(dest))

        if resolved_type in self._cache:
            del self._cache[resolved_type]

        if resolved_type not in self.BUILTIN_TEMPLATES:
            self.BUILTIN_TEMPLATES[resolved_type] = f"{resolved_type}.json"

    def list_templates(self) -> List[str]:
        templates = []
        if self.config.templates_dir.exists():
            for f in self.config.templates_dir.glob("*.json"):
                templates.append(f.stem)
        return templates
