import re
from typing import Any, Dict, List, Optional

from docuextract.config import ValidationConfig


class Validator:

    WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    CHECK_CHARS = "10X98765432"

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()

    def validate(
        self,
        doc_type: str,
        fields: Dict[str, Dict[str, Any]],
        rules: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}

        if doc_type == "id_card":
            results.update(self._validate_id_card(fields))
        elif doc_type == "invoice":
            results.update(self._validate_invoice(fields, rules))
        elif doc_type == "bank_card":
            results.update(self._validate_bank_card(fields))
        elif doc_type == "receipt":
            results.update(self._validate_invoice(fields, rules))

        for rule in rules:
            rule_type = rule.get("type", "")
            if rule_type not in results:
                results[rule_type] = {
                    "pass": True,
                    "reason": "Rule not applicable or no data to validate",
                }

        return results

    def _validate_id_card(
        self, fields: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}

        id_number = self._get_field_value(fields, "id_number")

        if not id_number:
            results["id_number_checksum"] = {
                "pass": False,
                "reason": "ID number not found",
            }
            results["id_number_length"] = {
                "pass": False,
                "reason": "ID number not found",
            }
            return results

        clean_id = id_number.upper().strip()

        length_ok = len(clean_id) == 18
        if not length_ok:
            results["id_number_length"] = {
                "pass": False,
                "reason": f"ID number must be 18 characters, got {len(clean_id)}",
            }
        else:
            results["id_number_length"] = {
                "pass": True,
                "reason": "",
            }

        if self.config.id_card_check_digit and length_ok:
            checksum_ok = self._check_id_number(clean_id)
            if checksum_ok:
                results["id_number_checksum"] = {
                    "pass": True,
                    "reason": "",
                }
            else:
                results["id_number_checksum"] = {
                    "pass": False,
                    "reason": "Checksum digit mismatch (GB 11643-1999, ISO 7064:1983 MOD 11-2)",
                }
        elif not length_ok:
            results["id_number_checksum"] = {
                "pass": False,
                "reason": "Cannot verify checksum: invalid length",
            }
        else:
            results["id_number_checksum"] = {
                "pass": True,
                "reason": "Checksum validation skipped",
            }

        return results

    def _check_id_number(self, id_number: str) -> bool:
        if len(id_number) != 18:
            return False

        digits_part = id_number[:17]
        check_char = id_number[17]

        if not digits_part.isdigit():
            return False

        total = 0
        for i, d in enumerate(digits_part):
            total += int(d) * self.WEIGHTS[i]

        remainder = total % 11
        expected_check = self.CHECK_CHARS[remainder]

        return check_char == expected_check

    def _validate_invoice(
        self,
        fields: Dict[str, Dict[str, Any]],
        rules: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}

        amount_str = self._get_field_value(fields, "amount")
        amount_no_tax_str = self._get_field_value(fields, "amount_without_tax")
        tax_str = self._get_field_value(fields, "tax_amount")
        invoice_code = self._get_field_value(fields, "invoice_code")
        invoice_number = self._get_field_value(fields, "invoice_number")

        if amount_str and amount_no_tax_str and tax_str:
            try:
                amount = self._parse_amount(amount_str)
                amount_no_tax = self._parse_amount(amount_no_tax_str)
                tax = self._parse_amount(tax_str)

                if abs(amount - (amount_no_tax + tax)) < self.config.invoice_amount_tolerance:
                    results["amount_check"] = {
                        "pass": True,
                        "reason": "",
                    }
                else:
                    diff = abs(amount - (amount_no_tax + tax))
                    results["amount_check"] = {
                        "pass": False,
                        "reason": f"Amount mismatch: {amount} != {amount_no_tax} + {tax} (diff={diff:.2f})",
                    }
            except (ValueError, TypeError):
                results["amount_check"] = {
                    "pass": False,
                    "reason": "Cannot parse amount values for comparison",
                }
        else:
            results["amount_check"] = {
                "pass": True,
                "reason": "Insufficient data for amount validation",
            }

        if invoice_code:
            clean_code = re.sub(r"\s+", "", invoice_code)
            if re.match(r"^\d{10,12}$", clean_code):
                results["invoice_code_format"] = {
                    "pass": True,
                    "reason": "",
                }
            else:
                results["invoice_code_format"] = {
                    "pass": False,
                    "reason": f"Invoice code must be 10-12 digits, got '{invoice_code}'",
                }
        else:
            results["invoice_code_format"] = {
                "pass": False,
                "reason": "Invoice code not found",
            }

        if invoice_number:
            clean_num = re.sub(r"\s+", "", invoice_number)
            if re.match(r"^\d{8,12}$", clean_num):
                results["invoice_number_format"] = {
                    "pass": True,
                    "reason": "",
                }
            else:
                results["invoice_number_format"] = {
                    "pass": False,
                    "reason": f"Invoice number must be 8-12 digits, got '{invoice_number}'",
                }
        else:
            results["invoice_number_format"] = {
                "pass": False,
                "reason": "Invoice number not found",
            }

        return results

    def _validate_bank_card(
        self, fields: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}

        card_number = self._get_field_value(fields, "card_number")

        if not card_number:
            results["luhn_check"] = {
                "pass": False,
                "reason": "Card number not found",
            }
            results["card_number_length"] = {
                "pass": False,
                "reason": "Card number not found",
            }
            return results

        clean_number = re.sub(r"[\s-]", "", card_number)

        length_ok = 13 <= len(clean_number) <= 19 and clean_number.isdigit()
        if length_ok:
            results["card_number_length"] = {
                "pass": True,
                "reason": "",
            }
        else:
            results["card_number_length"] = {
                "pass": False,
                "reason": f"Card number must be 13-19 digits, got '{card_number}'",
            }

        if self.config.bank_card_luhn and clean_number.isdigit():
            luhn_ok = self._luhn_check(clean_number)
            if luhn_ok:
                results["luhn_check"] = {
                    "pass": True,
                    "reason": "",
                }
            else:
                results["luhn_check"] = {
                    "pass": False,
                    "reason": "Luhn checksum failed",
                }
        else:
            results["luhn_check"] = {
                "pass": False,
                "reason": "Cannot perform Luhn check on non-numeric card number",
            }

        return results

    @staticmethod
    def _luhn_check(number: str) -> bool:
        digits = [int(d) for d in number]
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]

        total = sum(odd_digits)

        for d in even_digits:
            doubled = d * 2
            total += doubled - 9 if doubled > 9 else doubled

        return total % 10 == 0

    @staticmethod
    def _parse_amount(value: str) -> float:
        cleaned = re.sub(r"[,，¥￥元]", "", value.strip())
        return float(cleaned)

    @staticmethod
    def _get_field_value(fields: Dict[str, Dict[str, Any]], field_name: str) -> Optional[str]:
        field = fields.get(field_name)
        if field and field.get("value"):
            return str(field["value"])
        return None
