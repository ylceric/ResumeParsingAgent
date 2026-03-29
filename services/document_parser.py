"""Extract plain text from resume files (no LangChain)."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from docx import Document as DocxDocument
from PIL import Image
from pypdf import PdfReader

from chains.vision_resume import extract_text_from_image_vision
from utils.config import AppConfig


@dataclass
class ParseOutcome:
    text: str
    method: str
    error: str | None = None


MIN_OCR_CHARS = 80
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"}


def extract_text(path: str | Path, config: AppConfig | None = None) -> ParseOutcome:
    """Route by extension; images use OCR then optional vision fallback."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return ParseOutcome(
            text="",
            method="unsupported",
            error=f"Unsupported file type: {ext}",
        )
    try:
        if ext == ".pdf":
            return _pdf_text(p)
        if ext == ".docx":
            return _docx_text(p)
        if ext == ".txt":
            return _txt_text(p)
        return _image_text(p, config)
    except Exception as exc:  # noqa: BLE001 — surface to UI
        return ParseOutcome(
            text="",
            method="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def _pdf_text(path: Path) -> ParseOutcome:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            parts.append(t)
    text = "\n".join(parts).strip()
    if not text:
        return ParseOutcome(
            text="",
            method="pdf_empty",
            error="No extractable text in PDF (may be scanned). Try image/PDF OCR workflow.",
        )
    return ParseOutcome(text=text, method="pdf_text")


def _docx_text(path: Path) -> ParseOutcome:
    doc = DocxDocument(str(path))
    paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    text = "\n".join(paras).strip()
    if not text:
        return ParseOutcome(text="", method="docx_empty", error="Empty DOCX body")
    return ParseOutcome(text=text, method="docx")


def _txt_text(path: Path) -> ParseOutcome:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            text = raw.decode(enc).strip()
            return ParseOutcome(text=text, method=f"txt_{enc}")
        except UnicodeDecodeError:
            continue
    return ParseOutcome(text="", method="txt_decode", error="Could not decode text file")


def _image_text(path: Path, config: AppConfig | None) -> ParseOutcome:
    try:
        import pytesseract
    except ImportError:
        pytesseract = None  # type: ignore[assignment]

    ocr_text = ""
    if pytesseract is not None:
        try:
            img = Image.open(path)
            ocr_text = (pytesseract.image_to_string(img) or "").strip()
        except Exception as exc:  # noqa: BLE001
            ocr_text = ""
            ocr_err = f"{type(exc).__name__}: {exc}"
        else:
            ocr_err = None
    else:
        ocr_err = "pytesseract not installed"

    if len(ocr_text) >= MIN_OCR_CHARS:
        return ParseOutcome(text=ocr_text, method="ocr")

    if (
        config
        and config.openai_api_key
        and config.vision_fallback_enabled
    ):
        try:
            vision_text = extract_text_from_image_vision(str(path), config).strip()
            if vision_text:
                return ParseOutcome(
                    text=vision_text,
                    method="vision_llm",
                    error=None if not ocr_err else f"OCR weak ({ocr_err}); used vision model",
                )
        except Exception as exc:  # noqa: BLE001
            vision_err = f"{type(exc).__name__}: {exc}"
            err = " | ".join(filter(None, [ocr_err, vision_err]))
            return ParseOutcome(text=ocr_text, method="vision_failed", error=err)

    err_parts = []
    if ocr_err:
        err_parts.append(ocr_err)
    if len(ocr_text) < MIN_OCR_CHARS:
        err_parts.append(f"Extracted only {len(ocr_text)} characters")
    return ParseOutcome(
        text=ocr_text,
        method="ocr_insufficient",
        error="; ".join(err_parts) if err_parts else "Insufficient text from image",
    )


def is_supported_filename(name: str) -> bool:
    return Path(name).suffix.lower() in SUPPORTED_EXTENSIONS
