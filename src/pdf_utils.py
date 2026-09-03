import io
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*fitz.*")

import pymupdf
try:
    pymupdf.TOOLS.mupdf_display_errors(False)
except Exception:
    try:
        pymupdf.mupdf_display_errors(False)
    except Exception:
        pass

from PIL import Image
from .config import SUPPORTED_EXTENSIONS


def validate_file(input_path: str | Path) -> Path:
    """Validate file existence, non-zero size, and supported extension."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Document file is empty (0 bytes): {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: '{suffix}'. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    return path

def get_document_page_count(input_path: str | Path) -> int:
    """Return total number of pages in PDF or 1 for image."""
    path = validate_file(input_path)
    if path.suffix.lower() != ".pdf":
        return 1
    try:
        with pymupdf.open(str(path)) as doc:
            return len(doc)
    except Exception:
        raw_bytes = path.read_bytes()
        with pymupdf.open(stream=raw_bytes, filetype="pdf") as doc:
            return len(doc)


def load_single_page(input_path: str | Path, page_index: int = 0, pdf_dpi: int = 170) -> Image.Image:
    """Render and return ONLY a single page as a PIL RGB Image, conserving memory."""
    path = validate_file(input_path)
    if path.suffix.lower() != ".pdf":
        with Image.open(path) as img:
            return img.convert("RGB")

    try:
        with pymupdf.open(str(path)) as doc:
            if page_index >= len(doc):
                raise IndexError(f"Page index {page_index} out of range (document has {len(doc)} pages)")
            page = doc[page_index]
            pix = page.get_pixmap(dpi=pdf_dpi, alpha=False)
            return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception as exc:
        raw_bytes = path.read_bytes()
        with pymupdf.open(stream=raw_bytes, filetype="pdf") as doc:
            page = doc[page_index]
            pix = page.get_pixmap(dpi=pdf_dpi, alpha=False)
            return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def load_pages(input_path: str | Path, pdf_dpi: int = 170) -> list[Image.Image]:
    """Render PDF pages or load single image into a list of RGB PIL Images.
    
    Args:
        input_path: Path to the document.
        pdf_dpi: Rendering resolution for PDF pages (default 170 DPI).
        
    Returns:
        List of PIL Image objects (RGB).
    """
    path = validate_file(input_path)

    if path.suffix.lower() != ".pdf":
        try:
            with Image.open(path) as img:
                return [img.convert("RGB")]
        except Exception as exc:
            raise ValueError(f"Failed to open image {path}: {exc}") from exc

    # PDF Validation & Diagnostics
    try:
        header = path.read_bytes()[:8]
        if not header.startswith(b"%PDF"):
            # If not a standard PDF header, verify if it's an image disguised as PDF or malformed
            pass
    except Exception:
        pass

    pages = []
    doc = None
    try:
        # Open PDF file with pymupdf
        doc = pymupdf.open(str(path))
        if len(doc) == 0:
            raise ValueError(f"PDF document has 0 pages: {path}")

        for page in doc:
            pix = page.get_pixmap(dpi=pdf_dpi, alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(image)

    except Exception as exc:
        # Fallback: try opening from memory stream if direct file read had xref issues
        try:
            raw_bytes = path.read_bytes()
            doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(dpi=pdf_dpi, alpha=False)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pages.append(image)
        except Exception:
            raise RuntimeError(f"Error rendering PDF {path}: {exc}") from exc
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass

    return pages
