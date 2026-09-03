"""LayoutLMv3 tokenization preprocessing and row clustering."""

import numpy as np
from .utils import clamp_box, y_center, row_height


def normalize_box_1000(box, width: int, height: int) -> list[int]:
    """Normalize [x1, y1, x2, y2] to [0, 1000] scale required by LayoutLMv3."""
    x1, y1, x2, y2 = clamp_box(box, width, height)
    return [
        int(round(1000.0 * x1 / max(width, 1))),
        int(round(1000.0 * y1 / max(height, 1))),
        int(round(1000.0 * x2 / max(width, 1))),
        int(round(1000.0 * y2 / max(height, 1))),
    ]


def make_chunks(n: int, chunk_size: int = 350, overlap: int = 25) -> list[tuple[int, int]]:
    """Partition n word tokens into overlapping chunks to fit within 512 subwords."""
    if n <= chunk_size:
        return [(0, n)]

    chunks = []
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append((start, end))
        if end == n:
            break
        start = end - overlap

    return chunks


def candidate_rows(words: list[str], boxes: list[list[float]]) -> list[dict]:
    """Cluster words into visual rows based on vertical proximity."""
    if len(words) != len(boxes):
        raise ValueError("Words and boxes length mismatch")

    indexed = sorted(
        range(len(words)),
        key=lambda i: (y_center(boxes[i]), boxes[i][0])
    )

    rows = []

    for idx in indexed:
        yc = y_center(boxes[idx])
        h = row_height(boxes[idx])

        best = None
        best_delta = None

        for row in rows:
            tolerance = max(8.0, 0.55 * max(h, row["median_height"]))
            delta = abs(yc - row["y_center"])

            if delta <= tolerance:
                if best_delta is None or delta < best_delta:
                    best = row
                    best_delta = delta

        if best is None:
            rows.append({
                "indices": [idx],
                "y_center": yc,
                "median_height": h,
            })
        else:
            best["indices"].append(idx)
            ys = [y_center(boxes[i]) for i in best["indices"]]
            hs = [row_height(boxes[i]) for i in best["indices"]]
            best["y_center"] = float(np.median(ys))
            best["median_height"] = float(np.median(hs))

    for row in rows:
        row["indices"].sort(key=lambda i: boxes[i][0])
        row["text"] = " ".join(words[i] for i in row["indices"])
        row["x1"] = min(boxes[i][0] for i in row["indices"])
        row["x2"] = max(boxes[i][2] for i in row["indices"])

    rows.sort(key=lambda r: r["y_center"])
    return rows
