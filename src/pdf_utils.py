"""PDF and document rendering utilities using PyMuPDF (fitz) and Pillow."""

import io
from pathlib import Path
import fitz
from PIL import Image
from .config import SUPPORTED_EXTENSIONS


def validate_file(input_path: str | Path) -> Path:
    """Validate file existence, non-zero size, and supported extension."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Document file is empty (0 bytes): {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: '{path.suffix}'. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    return path


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

    pages = []
    doc = None
    try:
        doc = fitz.open(str(path))
        if len(doc) == 0:
            raise ValueError(f"PDF document has 0 pages: {path}")

        for page in doc:
            pix = page.get_pixmap(dpi=pdf_dpi, alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            pages.append(image)

    except Exception as exc:
        raise RuntimeError(f"Error rendering PDF {path}: {exc}") from exc
    finally:
        if doc is not None:
            doc.close()

    return pages
