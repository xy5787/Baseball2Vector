"""Render every manuscript preview page to PNG for visual inspection."""

from __future__ import annotations

import sys
from pathlib import Path

RUN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUN_ROOT / "tools" / "python"))
import fitz

pdf_path = RUN_ROOT / "manuscript_preview" / "baseball2vector_en_revision.pdf"
output_dir = RUN_ROOT / "manuscript_preview" / "rendered_pages"
output_dir.mkdir(exist_ok=True)
document = fitz.open(pdf_path)
for index, page in enumerate(document):
    pixmap = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
    pixmap.save(output_dir / f"page_{index + 1:02d}.png")
print(f"Rendered {len(document)} pages to {output_dir}")
