import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from docuextract.config import AppConfig, get_config
from docuextract.classification.doc_classifier import (
    DocumentClassifier,
    ClassifierModelNotFoundError,
    ClassifierModelLoadError,
)
from docuextract.ocr_pipeline.detector import (
    TextDetector,
    OCREngineNotFoundError,
)
from docuextract.ocr_pipeline.layout_parser import LayoutParser
from docuextract.extraction.field_extractor import FieldExtractor
from docuextract.extraction.template_matcher import TemplateMatcher
from docuextract.extraction.validator import Validator
from docuextract.reporting.json_reporter import JSONReporter
from docuextract.reporting.csv_exporter import CSVExporter

console = Console()

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".pdf"}


class DocumentLoadError(RuntimeError):
    """Failed to load document image or PDF."""
    pass


def _collect_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [input_path]
        console.print(f"[red]Unsupported file type: {input_path.suffix}[/red]")
        return []

    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(input_path.glob(f"*{ext}"))
        files.extend(input_path.glob(f"*{ext.upper()}"))

    files = sorted(set(files))
    return files


def _load_image(image_path: Path) -> np.ndarray:
    """Load an image file (including PDF first page) as numpy array.

    Raises DocumentLoadError on failure.
    """
    try:
        if image_path.suffix.lower() == ".pdf":
            try:
                from pdf2image import convert_from_path

                pages = convert_from_path(str(image_path), first_page=1, last_page=1)
                if not pages:
                    raise DocumentLoadError(
                        f"PDF file has no pages: {image_path}"
                    )
                return np.array(pages[0])
            except ImportError as e:
                raise DocumentLoadError(
                    f"pdf2image is required for PDF support. "
                    f"Install with: pip install pdf2image. Error: {e}"
                ) from e
            except Exception as e:
                raise DocumentLoadError(
                    f"Failed to convert PDF to image: {e}"
                ) from e

        img = Image.open(image_path).convert("RGB")
        return np.array(img)
    except DocumentLoadError:
        raise
    except Exception as e:
        raise DocumentLoadError(f"Failed to load image {image_path}: {e}") from e


def _build_error_report(
    file_path: Path,
    error_type: str,
    error_message: str,
) -> Dict[str, Any]:
    return {
        "file": str(file_path),
        "type": "unknown",
        "classification_confidence": 0.0,
        "fields": {},
        "validation": {},
        "error": {
            "type": error_type,
            "message": error_message,
        },
        "success": False,
    }


def _is_error_report(report: Dict[str, Any]) -> bool:
    return report.get("success") is False or "error" in report


def _process_single_file(
    image_path: Path,
    config: AppConfig,
    classifier: Optional[DocumentClassifier] = None,
    detector: Optional[TextDetector] = None,
    extractor: Optional[FieldExtractor] = None,
    validator: Optional[Validator] = None,
) -> Dict[str, Any]:
    try:
        image = _load_image(image_path)
    except DocumentLoadError as e:
        return _build_error_report(image_path, "load_error", str(e))

    try:
        if classifier is None:
            classifier = DocumentClassifier(config.classification)
            classifier.load_model()
    except (ClassifierModelNotFoundError, ClassifierModelLoadError) as e:
        return _build_error_report(image_path, "classifier_error", str(e))

    try:
        doc_type, class_conf, _ = classifier.predict_from_array(image)
    except Exception as e:
        return _build_error_report(
            image_path, "classification_error",
            f"Classification failed: {e}"
        )

    try:
        if detector is None:
            detector = TextDetector(config.ocr, strict=True)
        ocr_items = detector.detect_with_text(image)
    except OCREngineNotFoundError as e:
        return _build_error_report(image_path, "ocr_unavailable", str(e))
    except Exception as e:
        return _build_error_report(
            image_path, "ocr_error",
            f"OCR processing failed: {e}"
        )

    try:
        if extractor is None:
            extractor = FieldExtractor(
                config.extraction,
                TemplateMatcher(config.extraction),
                LayoutParser(),
            )
        fields = extractor.extract(doc_type, ocr_items, image.shape)
    except Exception as e:
        return _build_error_report(
            image_path, "extraction_error",
            f"Field extraction failed: {e}"
        )

    try:
        if validator is None:
            validator = Validator(config.validation)
        rules = extractor.template_matcher.get_validation_rules(doc_type)
        validation = validator.validate(doc_type, fields, rules)
    except Exception as e:
        return _build_error_report(
            image_path, "validation_error",
            f"Validation failed: {e}"
        )

    return {
        "file": str(image_path),
        "type": doc_type,
        "classification_confidence": class_conf,
        "fields": fields,
        "validation": validation,
        "success": True,
    }


def _process_single_for_parallel(args: Tuple[Path, AppConfig]) -> Dict[str, Any]:
    image_path, config = args
    return _process_single_file(image_path, config)


def _confidence_style(confidence: float) -> str:
    if confidence > 0.9:
        return "green"
    elif confidence >= 0.7:
        return "yellow"
    return "red"


def _print_result_table(report: Dict[str, Any]) -> None:
    if _is_error_report(report):
        err = report.get("error", {})
        console.print(f"[bold red]Processing failed:[/bold red] {Path(report['file']).name}")
        console.print(f"  Error type: {err.get('type', 'unknown')}")
        console.print(f"  Message: {err.get('message', '')}")
        console.print()
        return

    table = Table(title=f"Extraction Result: {Path(report['file']).name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_column("Confidence")
    table.add_column("Validation", justify="center")

    fields = report.get("fields", {})
    validation = report.get("validation", {})

    for field_name, field_data in fields.items():
        value = field_data.get("value")
        conf = field_data.get("confidence", 0.0)
        value_str = str(value) if value is not None else "N/A"
        conf_text = f"{conf:.2f}"
        conf_style = _confidence_style(conf)

        val_status = ""
        for rule_name, rule_data in validation.items():
            if field_name in rule_name.lower() or rule_name in field_name:
                if rule_data.get("pass", False):
                    val_status = "[green]\u2713[/green]"
                else:
                    val_status = f"[red]\u2717[/red] {rule_data.get('reason', '')}"

        table.add_row(
            field_name,
            value_str,
            f"[{conf_style}]{conf_text}[/{conf_style}]",
            val_status,
        )

    console.print(table)
    console.print()

    val_table = Table(title="Validation Summary")
    val_table.add_column("Rule", style="cyan")
    val_table.add_column("Result", justify="center")
    val_table.add_column("Reason")

    for rule_name, rule_data in validation.items():
        if rule_data.get("pass", False):
            result = "[green]\u2713 PASS[/green]"
        else:
            result = "[red]\u2717 FAIL[/red]"
        reason = rule_data.get("reason", "")
        val_table.add_row(rule_name, result, reason)

    console.print(val_table)


@click.group()
@click.version_option(version="1.0.0", prog_name="docuextract")
def cli():
    """DocuExtract - Document Intelligent Information Extraction CLI"""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output JSON file path")
@click.option("--format", "fmt", type=click.Choice(["json", "csv", "both"]), default="json", help="Output format")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed OCR results")
def predict(input_file: Path, output: Optional[Path], fmt: str, verbose: bool):
    """Extract information from a single document image or PDF."""
    config = get_config()

    console.print(f"[bold blue]Processing:[/bold blue] {input_file}")

    try:
        image = _load_image(input_file)
    except DocumentLoadError as e:
        console.print(f"[bold red]Failed to load document:[/bold red] {e}")
        sys.exit(1)

    console.print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    with console.status("Loading classification model..."):
        try:
            classifier = DocumentClassifier(config.classification)
            classifier.load_model()
        except ClassifierModelNotFoundError as e:
            console.print(f"[bold red]Classification model not found:[/bold red]")
            console.print(str(e))
            sys.exit(1)
        except ClassifierModelLoadError as e:
            console.print(f"[bold red]Failed to load classification model:[/bold red]")
            console.print(str(e))
            sys.exit(1)

    with console.status("Classifying document type..."):
        try:
            doc_type, class_conf, class_probs = classifier.predict_from_array(image)
        except Exception as e:
            console.print(f"[bold red]Classification failed:[/bold red] {e}")
            sys.exit(1)

    conf_style = _confidence_style(class_conf)
    console.print(
        f"[bold]Document Type:[/bold] {doc_type} "
        f"(confidence: [{conf_style}]{class_conf:.4f}[/{conf_style}])"
    )

    if verbose:
        prob_table = Table(title="Classification Probabilities")
        prob_table.add_column("Class", style="cyan")
        prob_table.add_column("Probability", justify="right")
        for cls_name, prob in class_probs.items():
            pct = f"{prob * 100:.2f}%"
            pct_style = _confidence_style(prob)
            prob_table.add_row(cls_name, f"[{pct_style}]{pct}[/{pct_style}]")
        console.print(prob_table)

    with console.status("Initializing OCR engine..."):
        try:
            detector = TextDetector(config.ocr, strict=True)
            _ = detector.is_available()
        except OCREngineNotFoundError as e:
            console.print(f"[bold red]OCR engine is not available:[/bold red]")
            console.print(str(e))
            sys.exit(1)

    with console.status("Running OCR..."):
        try:
            ocr_items = detector.detect_with_text(image)
        except Exception as e:
            console.print(f"[bold red]OCR processing failed:[/bold red] {e}")
            sys.exit(1)

    console.print(f"[bold]OCR blocks detected:[/bold] {len(ocr_items)}")

    if verbose and ocr_items:
        ocr_table = Table(title="OCR Results")
        ocr_table.add_column("Text")
        ocr_table.add_column("Confidence")
        ocr_table.add_column("Position")

        for item in ocr_items:
            text = item.get("text", "")
            conf = item.get("confidence", 0.0)
            bbox = item.get("bbox", [0, 0, 0, 0])
            conf_style = _confidence_style(conf)
            ocr_table.add_row(
                text,
                f"[{conf_style}]{conf:.4f}[/{conf_style}]",
                str(bbox),
            )
        console.print(ocr_table)

    with console.status("Extracting fields..."):
        layout_parser = LayoutParser()
        template_matcher = TemplateMatcher(config.extraction)
        extractor = FieldExtractor(config.extraction, template_matcher, layout_parser)
        try:
            fields = extractor.extract(doc_type, ocr_items, image.shape)
        except Exception as e:
            console.print(f"[bold red]Field extraction failed:[/bold red] {e}")
            sys.exit(1)

    with console.status("Validating..."):
        rules = template_matcher.get_validation_rules(doc_type)
        val = Validator(config.validation)
        validation = val.validate(doc_type, fields, rules)

    reporter = JSONReporter(config.output)
    report = reporter.generate_report(
        file_path=str(input_file),
        doc_type=doc_type,
        classification_confidence=class_conf,
        fields=fields,
        validation=validation,
        ocr_items=ocr_items if verbose else None,
    )
    report["success"] = True

    _print_result_table(report)

    timestamp = datetime.now().strftime(config.output.timestamp_format)

    if fmt in ("json", "both"):
        json_path = output or Path(f"extraction_{timestamp}.json")
        saved = reporter.save_report(report, json_path)
        console.print(f"[green]JSON report saved to:[/green] {saved}")

    if fmt in ("csv", "both"):
        csv_exporter = CSVExporter(config.output)
        csv_path = Path(f"extraction_{timestamp}.csv")
        saved = csv_exporter.export([report], csv_path, doc_type)
        console.print(f"[green]CSV report saved to:[/green] {saved}")


@cli.command()
@click.option("--input", "-i", "input_dir", type=click.Path(exists=True, path_type=Path), required=True, help="Input directory with document images/PDFs")
@click.option("--output", "-o", "output_path", type=click.Path(path_type=Path), default=None, help="Output file path")
@click.option("--format", "fmt", type=click.Choice(["json", "csv", "both"]), default="json", help="Output format")
@click.option("--workers", "-w", type=int, default=1, help="Number of parallel workers")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed results")
def batch(input_dir: Path, output_path: Optional[Path], fmt: str, workers: int, verbose: bool):
    """Batch process a directory of document images and PDFs."""
    config = get_config()

    files = _collect_files(input_dir)
    if not files:
        console.print("[red]No supported image/PDF files found[/red]")
        sys.exit(1)

    console.print(f"[bold blue]Found {len(files)} files to process[/bold blue]")

    reports: List[Dict[str, Any]] = []

    if workers > 1:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing documents...", total=len(files))

            args_list = [(f, config) for f in files]

            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_process_single_for_parallel, args): args[0] for args in args_list}

                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        report = future.result()
                        reports.append(report)
                        status_icon = "\u2713" if not _is_error_report(report) else "\u2717"
                        progress.update(
                            task, advance=1,
                            description=f"{status_icon} {Path(str(file_path)).name}"
                        )
                    except Exception as e:
                        reports.append(_build_error_report(
                            file_path, "unexpected_error", str(e)
                        ))
                        progress.update(task, advance=1)
    else:
        try:
            classifier = DocumentClassifier(config.classification)
            classifier.load_model()
        except (ClassifierModelNotFoundError, ClassifierModelLoadError) as e:
            console.print(f"[bold red]Failed to load classification model:[/bold red]")
            console.print(str(e))
            console.print()
            console.print("[yellow]Tip: Place the pre-trained model at the path shown above,[/yellow]")
            console.print("[yellow]or install pytorch + torchvision if not present.[/yellow]")
            sys.exit(1)

        try:
            detector = TextDetector(config.ocr, strict=True)
            _ = detector.is_available()
        except OCREngineNotFoundError as e:
            console.print(f"[bold yellow]OCR engine not available:[/bold yellow]")
            console.print(str(e))
            console.print()
            console.print("[yellow]Batch processing will continue but all files will fail at OCR stage.[/yellow]")
            console.print("[yellow]Install PaddleOCR for full functionality: pip install paddleocr paddlepaddle[/yellow]")

        extractor = FieldExtractor(
            config.extraction,
            TemplateMatcher(config.extraction),
            LayoutParser(),
        )
        val = Validator(config.validation)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing documents...", total=len(files))

            for file_path in files:
                progress.update(task, description=f"Processing {file_path.name}")
                try:
                    report = _process_single_file(
                        file_path, config, classifier, detector, extractor, val
                    )
                    reports.append(report)
                except Exception as e:
                    reports.append(_build_error_report(
                        file_path, "unexpected_error", str(e)
                    ))
                progress.update(task, advance=1)

    reporter = JSONReporter(config.output)

    success_reports = [r for r in reports if not _is_error_report(r)]
    error_reports = [r for r in reports if _is_error_report(r)]

    console.print()
    console.print(f"[bold]Batch Processing Summary[/bold]")
    console.print(f"  Total files: {len(reports)}")
    console.print(f"  [green]Successfully processed: {len(success_reports)}[/green]")
    console.print(f"  [red]Failed: {len(error_reports)}[/red]")

    if error_reports:
        console.print()
        err_table = Table(title="Failed Files")
        err_table.add_column("File", style="cyan")
        err_table.add_column("Error Type")
        err_table.add_column("Message")
        for report in error_reports:
            err = report.get("error", {})
            err_table.add_row(
                Path(report["file"]).name,
                err.get("type", "unknown"),
                err.get("message", "")[:80],
            )
        console.print(err_table)

    if success_reports:
        type_counts: Dict[str, int] = {}
        for r in success_reports:
            t = r.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        console.print(f"  Type distribution: {type_counts}")

        passed = sum(
            1
            for r in success_reports
            if all(v.get("pass", False) for v in r.get("validation", {}).values())
        )
        console.print(f"  [green]Validation passed: {passed}[/green]")
        console.print(f"  [red]Validation failed: {len(success_reports) - passed}[/red]")

    if verbose:
        for report in reports:
            _print_result_table(report)

    all_reports = []
    for r in reports:
        if _is_error_report(r):
            all_reports.append(r)
        else:
            all_reports.append(r)

    batch_report: Dict[str, Any] = {
        "summary": {
            "total_files": len(reports),
            "success_count": len(success_reports),
            "failed_count": len(error_reports),
        },
        "results": all_reports,
    }

    timestamp = datetime.now().strftime(config.output.timestamp_format)

    if fmt in ("json", "both"):
        json_path = output_path or Path(f"extraction_{timestamp}.json")
        saved = reporter.save_report(batch_report, json_path)
        console.print(f"[green]JSON report saved to:[/green] {saved}")

    if fmt in ("csv", "both") and success_reports:
        csv_exporter = CSVExporter(config.output)
        if output_path:
            csv_path = output_path.with_suffix(".csv")
        else:
            csv_path = Path(f"extraction_{timestamp}.csv")
        saved = csv_exporter.export(success_reports, csv_path)
        console.print(f"[green]CSV report saved to:[/green] {saved} (success records only)")

    if error_reports and len(error_reports) > 0:
        console.print(f"[yellow]{len(error_reports)} file(s) failed. See JSON report for details.[/yellow]")


@cli.command()
@click.option("--input", "-i", "input_file", type=click.Path(exists=True, path_type=Path), required=True, help="Input document image/PDF to validate")
@click.option("--type", "doc_type", type=click.Choice(["invoice", "id_card", "bank_card", "receipt"]), default=None, help="Override document type (skip classification)")
def validate(input_file: Path, doc_type: Optional[str]):
    """Validate extracted fields from a document image or PDF."""
    config = get_config()

    console.print(f"[bold blue]Validating:[/bold blue] {input_file}")

    try:
        image = _load_image(input_file)
    except DocumentLoadError as e:
        console.print(f"[bold red]Failed to load document:[/bold red] {e}")
        sys.exit(1)

    if doc_type is None:
        with console.status("Loading classification model..."):
            try:
                classifier = DocumentClassifier(config.classification)
                classifier.load_model()
            except ClassifierModelNotFoundError as e:
                console.print(f"[bold red]Classification model not found:[/bold red]")
                console.print(str(e))
                sys.exit(1)
            except ClassifierModelLoadError as e:
                console.print(f"[bold red]Failed to load classification model:[/bold red]")
                console.print(str(e))
                sys.exit(1)

        with console.status("Classifying document type..."):
            doc_type, class_conf, _ = classifier.predict_from_array(image)
        console.print(f"[bold]Document Type:[/bold] {doc_type} (confidence: {class_conf:.4f})")
    else:
        console.print(f"[bold]Document Type:[/bold] {doc_type} (manually set)")

    with console.status("Initializing OCR engine..."):
        try:
            detector = TextDetector(config.ocr, strict=True)
            _ = detector.is_available()
        except OCREngineNotFoundError as e:
            console.print(f"[bold red]OCR engine is not available:[/bold red]")
            console.print(str(e))
            sys.exit(1)

    with console.status("Running OCR and extraction..."):
        ocr_items = detector.detect_with_text(image)

        layout_parser = LayoutParser()
        template_matcher = TemplateMatcher(config.extraction)
        extractor = FieldExtractor(config.extraction, template_matcher, layout_parser)
        fields = extractor.extract(doc_type, ocr_items, image.shape)

    with console.status("Running validation..."):
        rules = template_matcher.get_validation_rules(doc_type)
        val = Validator(config.validation)
        validation = val.validate(doc_type, fields, rules)

    _print_result_table({
        "file": str(input_file),
        "type": doc_type,
        "classification_confidence": 0.0 if doc_type else 0,
        "fields": fields,
        "validation": validation,
        "success": True,
    })

    all_pass = all(v.get("pass", False) for v in validation.values())
    if all_pass:
        console.print("\n[bold green]All validations passed! \u2713[/bold green]")
    else:
        console.print("\n[bold red]Some validations failed! \u2717[/bold red]")
        for rule_name, rule_data in validation.items():
            if not rule_data.get("pass", False):
                console.print(f"  [red]\u2717 {rule_name}: {rule_data.get('reason', '')}[/red]")


@cli.command(name="add-template")
@click.argument("template_file", type=click.Path(exists=True, path_type=Path))
@click.option("--type", "doc_type", type=str, default=None, help="Document type name (overrides template doc_type)")
def add_template(template_file: Path, doc_type: Optional[str]):
    """Add a custom field template for extraction."""
    config = get_config()
    matcher = TemplateMatcher(config.extraction)
    matcher.add_template(template_file, doc_type)
    console.print(f"[green]Template added successfully from {template_file}[/green]")


@cli.command(name="list-templates")
def list_templates():
    """List all available field templates."""
    config = get_config()
    matcher = TemplateMatcher(config.extraction)
    templates = matcher.list_templates()

    if not templates:
        console.print("[yellow]No templates found[/yellow]")
        return

    table = Table(title="Available Templates")
    table.add_column("Template Name", style="cyan")
    table.add_column("Fields", justify="right")

    for name in templates:
        try:
            fields = matcher.get_fields(name)
            table.add_row(name, str(len(fields)))
        except Exception:
            table.add_row(name, "?")

    console.print(table)


if __name__ == "__main__":
    cli()
