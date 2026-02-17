"""
Step 1: Run Mistral OCR on all extracted population census PDFs.
Saves raw markdown results to ocr_cache/ so parsing can be re-run without re-OCR.

Usage:
    python3 mistral/extraction/popolazione/run_ocr.py
"""

import json
import os
import base64
import time
from pathlib import Path
from mistralai import Mistral

PDF_DIR = "PDF_folders copia/Popolazione 1921/extracted"
CACHE_DIR = "mistral/extraction/popolazione/ocr_cache"
API_KEY = os.getenv("MISTRAL_API_KEY", "")


def ocr_pdf(client: Mistral, pdf_path: str) -> dict:
    """Send a PDF to Mistral OCR and return the response."""
    with open(pdf_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    resp = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{b64}",
        },
        include_image_base64=False,
    )
    return json.loads(resp.model_dump_json())


def main():
    if not API_KEY:
        print("ERROR: Set MISTRAL_API_KEY environment variable")
        return

    client = Mistral(api_key=API_KEY)
    os.makedirs(CACHE_DIR, exist_ok=True)

    pdfs = sorted(Path(PDF_DIR).glob("*_extracted.pdf"))
    print(f"Found {len(pdfs)} PDFs to process\n")

    total_pages = 0
    for pdf_path in pdfs:
        region = pdf_path.stem.replace("_popolazione_extracted", "")
        cache_path = os.path.join(CACHE_DIR, f"{region}.json")

        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            n = len(cached["pages"])
            total_pages += n
            print(f"  {region}: {n} pages (cached)")
            continue

        print(f"  {region}: OCR processing...", end="", flush=True)
        t0 = time.time()
        result = ocr_pdf(client, str(pdf_path))
        elapsed = time.time() - t0
        n = len(result["pages"])
        total_pages += n

        with open(cache_path, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f" {n} pages in {elapsed:.1f}s")

    cost = total_pages * 0.002
    print(f"\nDone. {total_pages} pages total. Estimated cost: ${cost:.2f}")


if __name__ == "__main__":
    main()
