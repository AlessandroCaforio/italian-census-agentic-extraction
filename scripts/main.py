"""
Main Pipeline Orchestrator for 1927 Italian Census OCR.

Coordinates the complete extraction workflow:
1. PDF → Images (convert_pdf.py)
2. OCR with Surya (run_surya.py) - optional intermediate step
3. LLM structuring with Claude Code's native vision (llm_structure.py)
4. Merge and validate (merge_validate.py)

The LLM extraction (Step 3) uses Claude Code's native multimodal capabilities.
Claude Code reads page images directly and extracts structured data interactively.

Usage:
    # Prepare images from PDFs
    python main.py prepare --pdf-dir ./pdfs --output-dir ./output

    # After Claude Code extracts data, merge results
    python main.py merge --output-dir ./output

    # Full pipeline info
    python main.py status --output-dir ./output
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir: Path, level: str = "INFO") -> logging.Logger:
    """Configure logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger("census_ocr")
    logger.info(f"Logging to: {log_file}")
    return logger


def cmd_prepare(args):
    """Prepare images from PDFs for extraction."""
    from convert_pdf import convert_pdf_to_images, convert_multiple_pdfs

    logger = logging.getLogger("census_ocr")

    pdf_source = args.pdf or args.pdf_dir
    output_base = args.output_dir
    images_dir = output_base / "images"

    logger.info("=" * 60)
    logger.info("STEP 1: Converting PDFs to images")
    logger.info("=" * 60)

    if args.use_existing_images and images_dir.exists() and list(images_dir.glob("**/*.png")):
        logger.info("Using existing images (skipping conversion)")
        all_images = sorted(images_dir.glob("**/*.png"))
        print(f"Found {len(all_images)} existing images")
    else:
        if pdf_source.is_file():
            # Single PDF - put images directly in images_dir
            image_paths = convert_pdf_to_images(pdf_source, images_dir)
            print(f"Converted {len(image_paths)} pages from {pdf_source.name}")
        else:
            # Multiple PDFs - create subdirectories
            pdf_results = convert_multiple_pdfs(pdf_source, images_dir)
            total_images = sum(len(imgs) for imgs in pdf_results.values())
            print(f"Converted {total_images} pages from {len(pdf_results)} PDFs")

    # Show spread information
    from llm_structure import prepare_spreads

    structured_dir = output_base / "structured"

    # Find all image directories
    subdirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    if not subdirs:
        subdirs = [images_dir]

    total_spreads = 0
    total_pending = 0

    for img_dir in subdirs:
        spreads = prepare_spreads(img_dir, structured_dir / img_dir.name if img_dir != images_dir else structured_dir)
        pending = sum(1 for s in spreads if not s.is_processed)
        total_spreads += len(spreads)
        total_pending += pending

    print(f"\n{'='*60}")
    print("PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total page spreads: {total_spreads}")
    print(f"Pending extraction: {total_pending}")
    print(f"\nNext step: Use Claude Code to read images and extract data.")
    print(f"Images location: {images_dir}")


def cmd_status(args):
    """Show pipeline status."""
    from llm_structure import prepare_spreads, get_pending_spreads

    output_base = args.output_dir
    images_dir = output_base / "images"
    structured_dir = output_base / "structured"
    final_output = output_base / "output"

    print(f"\n{'='*60}")
    print("PIPELINE STATUS")
    print(f"{'='*60}")

    # Check images
    if images_dir.exists():
        all_images = list(images_dir.glob("**/*.png"))
        print(f"\n[Images] {len(all_images)} PNG files in {images_dir}")
    else:
        print(f"\n[Images] Not found - run 'prepare' first")
        return

    # Check spreads
    subdirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    if not subdirs:
        subdirs = [images_dir]

    all_spreads = []
    for img_dir in subdirs:
        struct_out = structured_dir / img_dir.name if img_dir != images_dir else structured_dir
        spreads = prepare_spreads(img_dir, struct_out)
        all_spreads.extend(spreads)

    processed = sum(1 for s in all_spreads if s.is_processed)
    pending = len(all_spreads) - processed

    print(f"\n[Extraction] {processed}/{len(all_spreads)} spreads processed ({pending} pending)")

    if pending > 0:
        print("\nPending spreads:")
        for spread in get_pending_spreads(all_spreads)[:10]:
            print(f"  ○ {spread.spread_id}")
            print(f"    Left:  {spread.left_page}")
            print(f"    Right: {spread.right_page}")
        if pending > 10:
            print(f"  ... and {pending - 10} more")

    # Check final output
    final_csv = final_output / "census_1927.csv"
    if final_csv.exists():
        import pandas as pd
        df = pd.read_csv(final_csv)
        print(f"\n[Output] {final_csv}")
        print(f"  Records: {len(df)}")
        print(f"  Provinces: {df['provincia'].nunique()}")
    else:
        print(f"\n[Output] Not yet generated - run 'merge' after extraction")


def cmd_merge(args):
    """Merge extracted data into final CSV."""
    from merge_validate import merge_and_export

    output_base = args.output_dir
    structured_dir = output_base / "structured"
    final_output_dir = output_base / "output"

    print(f"\n{'='*60}")
    print("MERGING EXTRACTED DATA")
    print(f"{'='*60}")

    # Find all structured JSON directories
    if structured_dir.exists():
        output_files = merge_and_export(structured_dir, final_output_dir)

        print(f"\n{'='*60}")
        print("MERGE COMPLETE")
        print(f"{'='*60}")
        print("\nOutput files:")
        for name, path in output_files.items():
            if path:
                print(f"  {name}: {path}")
    else:
        print(f"No extracted data found in {structured_dir}")
        print("Run Claude Code extraction first.")


def cmd_extract_interactive(args):
    """Show instructions for interactive extraction with Claude Code."""
    from llm_structure import prepare_spreads, get_pending_spreads, EXTRACTION_PROMPT

    output_base = args.output_dir
    images_dir = output_base / "images"
    structured_dir = output_base / "structured"

    # Find pending spreads
    subdirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
    if not subdirs:
        subdirs = [images_dir]

    all_spreads = []
    for img_dir in subdirs:
        struct_out = structured_dir / img_dir.name if img_dir != images_dir else structured_dir
        spreads = prepare_spreads(img_dir, struct_out)
        all_spreads.extend(spreads)

    pending = get_pending_spreads(all_spreads)

    if not pending:
        print("All spreads have been processed!")
        print("Run 'merge' to create the final CSV.")
        return

    # Show next spread to process
    next_spread = pending[0]

    print(f"\n{'='*60}")
    print("INTERACTIVE EXTRACTION WITH CLAUDE CODE")
    print(f"{'='*60}")
    print(f"\nPending: {len(pending)} spreads")
    print(f"\nNext spread to process:")
    print(f"  ID: {next_spread.spread_id}")
    print(f"  Left page (comuni):  {next_spread.left_page}")
    print(f"  Right page (dati):   {next_spread.right_page}")
    print(f"  Output: {next_spread.output_path}")

    print(f"\n{'='*60}")
    print("EXTRACTION PROMPT")
    print(f"{'='*60}")
    print(EXTRACTION_PROMPT)

    print(f"\n{'='*60}")
    print("INSTRUCTIONS FOR CLAUDE CODE")
    print(f"{'='*60}")
    print("""
1. Read both images using the Read tool:
   - Read: {left_page}
   - Read: {right_page}

2. Analyze the images and extract data following the prompt above

3. Save the extracted JSON using this Python code:
   ```python
   import sys
   sys.path.insert(0, 'src')
   from llm_structure import PageSpread, save_extraction
   from pathlib import Path

   spread = PageSpread(
       left_page=Path("{left_page}"),
       right_page=Path("{right_page}"),
       spread_id="{spread_id}",
       output_path=Path("{output_path}")
   )

   data = {{
       "provincia": "...",
       "comuni": [...],
       "totale_provincia": {{...}}
   }}

   save_extraction(spread, data)
   ```

4. Repeat for remaining spreads, or run 'merge' when done.
""".format(
        left_page=next_spread.left_page,
        right_page=next_spread.right_page,
        spread_id=next_spread.spread_id,
        output_path=next_spread.output_path,
    ))


def main():
    parser = argparse.ArgumentParser(
        description="1927 Italian Census OCR Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Prepare command
    prepare_parser = subparsers.add_parser("prepare", help="Convert PDFs to images")
    prepare_parser.add_argument("--pdf", type=Path, help="Single PDF file")
    prepare_parser.add_argument("--pdf-dir", type=Path, help="Directory with PDFs")
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--use-existing-images", action="store_true")
    prepare_parser.add_argument("--log-level", default="INFO")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show pipeline status")
    status_parser.add_argument("--output-dir", type=Path, required=True)
    status_parser.add_argument("--log-level", default="WARNING")

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge extracted data")
    merge_parser.add_argument("--output-dir", type=Path, required=True)
    merge_parser.add_argument("--log-level", default="INFO")

    # Extract command (shows instructions)
    extract_parser = subparsers.add_parser("extract", help="Show extraction instructions")
    extract_parser.add_argument("--output-dir", type=Path, required=True)
    extract_parser.add_argument("--log-level", default="WARNING")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\nExamples:")
        print("  # Prepare images from PDFs")
        print('  python main.py prepare --pdf-dir "./output copia" --output-dir ./pipeline')
        print()
        print("  # Check status")
        print("  python main.py status --output-dir ./pipeline")
        print()
        print("  # Show extraction instructions")
        print("  python main.py extract --output-dir ./pipeline")
        print()
        print("  # Merge extracted data into CSV")
        print("  python main.py merge --output-dir ./pipeline")
        return

    # Setup logging
    log_dir = args.output_dir / "logs"
    setup_logging(log_dir, args.log_level)

    # Route to command
    if args.command == "prepare":
        if not args.pdf and not args.pdf_dir:
            prepare_parser.error("Either --pdf or --pdf-dir required")
        cmd_prepare(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "merge":
        cmd_merge(args)
    elif args.command == "extract":
        cmd_extract_interactive(args)


if __name__ == "__main__":
    main()
