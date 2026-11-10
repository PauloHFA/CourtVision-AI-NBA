"""Extract text from the CourtVision technical specification PDF."""
from pathlib import Path

from pypdf import PdfReader

root = Path(__file__).resolve().parents[1]
pdf_path = root / "CourtVision_AI_Especificacao_Tecnica_Completa.pdf"
out_path = root / "docs" / "especificacao_extraida.txt"

reader = PdfReader(str(pdf_path))
parts = [f"PAGES: {len(reader.pages)}", ""]
for i, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    parts.append(f"\n\n===== PAGE {i} =====\n")
    parts.append(text)

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("".join(parts), encoding="utf-8")
print(f"pages={len(reader.pages)}")
print(f"chars={out_path.stat().st_size}")
print(f"wrote={out_path}")
