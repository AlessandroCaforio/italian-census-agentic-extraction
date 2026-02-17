#!/usr/bin/env python3
"""Census extraction CLI helper for the opus46 workflow.

Manages per-PDF extraction status and validates output JSONs against
the reference CSV and the expected schema.

Usage:
    python3 opus46/census_extractor.py status
    python3 opus46/census_extractor.py next
    python3 opus46/census_extractor.py validate opus46/pair_XXXX.json
    python3 opus46/census_extractor.py compare  opus46/pair_XXXX.json
"""

import argparse
import json
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPUS46_DIR = PROJECT_ROOT / "opus46"
PDF_DIR = PROJECT_ROOT / "2page_pdfs"
CSV_PATH = PROJECT_ROOT / "pre-war-covariates-complete_updated.csv"
NOTES_PATH = OPUS46_DIR / "notes.md"

# Reuse helpers from the existing extraction code
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from extract_ind_workers import (
    build_province_lookup,
    compare_values,
    load_reference_csv,
    map_province,
    match_name,
    normalize_name,
)

# Total PDFs in the corpus (pair_0000 .. pair_0187)
TOTAL_PDFS = 188

# ── Schema definition ──────────────────────────────────────────────────────

REQUIRED_TOP_KEYS = {"provincia", "comuni"}
REQUIRED_COMUNE_KEYS = {
    "numero_ordine",
    "comune",
    "complesso_esercizi",
    "complesso_addetti",
    "commercio_esercizi",
    "commercio_addetti",
    "vendita_vino",
}
OPTIONAL_COMUNE_KEYS = {"vendita_liquori"}
NUMERIC_FIELDS = {
    "numero_ordine",
    "complesso_esercizi",
    "complesso_addetti",
    "commercio_esercizi",
    "commercio_addetti",
    "vendita_vino",
    "vendita_liquori",
}
TOTALE_KEYS = REQUIRED_COMUNE_KEYS - {"numero_ordine", "comune"} | OPTIONAL_COMUNE_KEYS


# ── Inventory helpers ──────────────────────────────────────────────────────


def all_pair_ids() -> list[str]:
    """Return sorted list of all pair IDs like 'pair_0000'."""
    return [f"pair_{i:04d}" for i in range(TOTAL_PDFS)]


def done_pair_ids() -> set[str]:
    """Return set of pair IDs that already have a JSON or lock file in opus46/."""
    done = set()
    for f in OPUS46_DIR.glob("pair_*.json"):
        done.add(f.stem)
    for f in OPUS46_DIR.glob("pair_*.lock"):
        done.add(f.stem)
    return done


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ── Commands ───────────────────────────────────────────────────────────────


def cmd_status() -> None:
    """Dashboard: done / total / remaining / next PDF."""
    all_ids = all_pair_ids()
    done = done_pair_ids()

    remaining = [pid for pid in all_ids if pid not in done]
    skipped = 0
    in_progress = 0
    for pid in done:
        lock_path = OPUS46_DIR / f"{pid}.lock"
        json_path = OPUS46_DIR / f"{pid}.json"
        if lock_path.exists() and not json_path.exists():
            in_progress += 1
        elif json_path.exists():
            try:
                data = load_json(json_path)
                if data.get("_skip"):
                    skipped += 1
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

    completed = len(done) - skipped - in_progress
    next_pdf = remaining[0] if remaining else None

    print(f"\n{'='*50}")
    print("  OPUS46 EXTRACTION STATUS")
    print(f"{'='*50}")
    print(f"  Total PDFs:     {TOTAL_PDFS}")
    print(f"  Completed:      {completed:3d}")
    if in_progress:
        print(f"  In progress:    {in_progress:3d}")
    if skipped:
        print(f"  Skipped:        {skipped:3d}")
    print(f"  Remaining:      {len(remaining):3d}")
    if next_pdf:
        print(f"  Next:           {next_pdf}.pdf")
    else:
        print("  Next:           ALL DONE!")
    print(f"{'='*50}\n")


def cmd_next() -> None:
    """Print the next unprocessed PDF path + any notes."""
    all_ids = all_pair_ids()
    done = done_pair_ids()
    remaining = [pid for pid in all_ids if pid not in done]

    if not remaining:
        print("All PDFs have been processed!")
        return

    next_id = remaining[0]
    pdf_path = PDF_DIR / f"{next_id}.pdf"

    print(f"\nNext PDF:  {pdf_path}")
    print(f"Output:    {OPUS46_DIR / f'{next_id}.json'}")

    # Check if notes mention this pair
    if NOTES_PATH.exists():
        notes = NOTES_PATH.read_text(encoding="utf-8")
        if next_id in notes:
            # Extract lines mentioning this pair
            relevant = [l for l in notes.splitlines() if next_id in l]
            if relevant:
                print(f"\nNotes mentioning {next_id}:")
                for line in relevant:
                    print(f"  {line.strip()}")
    print()


def cmd_validate(json_path: Path) -> None:
    """Schema check + provincial total verification."""
    try:
        data = load_json(json_path)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"ERROR: Cannot read {json_path}: {e}")
        sys.exit(1)

    # Handle skip markers
    if data.get("_skip"):
        print(f"SKIP: {json_path.name} — {data.get('_reason', 'no reason given')}")
        return

    errors: list[str] = []
    warnings: list[str] = []

    # Top-level keys
    for key in REQUIRED_TOP_KEYS:
        if key not in data:
            errors.append(f"Missing top-level key: '{key}'")

    provincia = data.get("provincia", "")
    comuni = data.get("comuni", [])
    totale = data.get("totale_provincia")

    if not isinstance(comuni, list):
        errors.append("'comuni' is not a list")
        comuni = []

    if not comuni:
        errors.append("'comuni' is empty")

    # Per-municipality checks
    prev_ordine = 0
    for i, c in enumerate(comuni):
        # Required keys
        for key in REQUIRED_COMUNE_KEYS:
            if key not in c:
                errors.append(f"comuni[{i}] ({c.get('comune', '?')}): missing '{key}'")

        # Type checks for numeric fields
        for field in NUMERIC_FIELDS:
            val = c.get(field)
            if val is not None and not isinstance(val, (int, float)):
                errors.append(
                    f"comuni[{i}] ({c.get('comune', '?')}): "
                    f"'{field}' should be int|null, got {type(val).__name__}"
                )

        # Sequential numero_ordine
        ordine = c.get("numero_ordine")
        if ordine is not None:
            if ordine != prev_ordine + 1:
                warnings.append(
                    f"comuni[{i}] ({c.get('comune', '?')}): "
                    f"numero_ordine={ordine}, expected {prev_ordine + 1}"
                )
            prev_ordine = ordine

    # Provincial total verification
    if totale and comuni:
        for field in TOTALE_KEYS:
            total_val = totale.get(field)
            if total_val is None:
                continue
            computed = sum(c.get(field) or 0 for c in comuni)
            if computed != total_val:
                diff = total_val - computed
                errors.append(
                    f"totale_provincia.{field}: stated={total_val}, "
                    f"sum={computed}, diff={diff:+d}"
                )

    # Print results
    print(f"\nValidation: {json_path.name}")
    print(f"  Province:       {provincia}")
    print(f"  Municipalities: {len(comuni)}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    [!] {e}")
    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    [~] {w}")
    if not errors and not warnings:
        print("  Result:         PASS")
    elif not errors:
        print(f"  Result:         PASS (with {len(warnings)} warnings)")
    else:
        print(f"  Result:         FAIL ({len(errors)} errors)")
    print()


def cmd_compare(json_path: Path) -> None:
    """Accuracy check: compare complesso_addetti vs ind_workers_1927 in CSV."""
    try:
        data = load_json(json_path)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"ERROR: Cannot read {json_path}: {e}")
        sys.exit(1)

    if data.get("_skip"):
        print(f"SKIP: {json_path.name} — {data.get('_reason', 'no reason given')}")
        return

    ref_df = load_reference_csv()
    province_lookup = build_province_lookup(ref_df)

    provincia = data.get("provincia", "")
    comuni = data.get("comuni", [])
    modern_prov = map_province(provincia)
    prov_df = province_lookup.get(modern_prov) if modern_prov else None

    total = len(comuni)
    matched = 0
    mismatched = 0
    csv_empty = 0
    no_ref = 0
    mismatch_details: list[str] = []

    for c in comuni:
        comune_name = c.get("comune", "")
        extracted = c.get("complesso_addetti")

        if extracted is not None:
            try:
                extracted = int(extracted)
            except (ValueError, TypeError):
                extracted = None

        norm = normalize_name(comune_name)
        match_status, csv_name, csv_val = match_name(norm, prov_df, ref_df)

        if match_status == "no_ref":
            no_ref += 1
            continue

        val_status = compare_values(extracted, csv_val)
        if val_status == "match":
            matched += 1
        elif val_status == "csv_empty":
            csv_empty += 1
        elif val_status == "mismatch":
            mismatched += 1
            csv_int = int(round(csv_val)) if csv_val is not None else "?"
            mismatch_details.append(
                f"    {comune_name}: extracted={extracted}, csv={csv_int}"
                f" (matched as '{csv_name}', {match_status})"
            )
        else:
            csv_empty += 1  # extracted_null treated as csv_empty for reporting

    comparable = matched + mismatched
    accuracy = matched / comparable * 100 if comparable else 0

    print(f"\nComparison: {json_path.name}")
    print(f"  Province:       {provincia} → {modern_prov or 'N/A'}")
    print(f"  Municipalities: {total}")
    print(f"  Matched:        {matched}")
    print(f"  Mismatched:     {mismatched}")
    print(f"  CSV empty:      {csv_empty}")
    print(f"  No reference:   {no_ref}")
    if comparable:
        print(f"  Accuracy:       {accuracy:.1f}% ({matched}/{comparable})")
    else:
        print("  Accuracy:       N/A (no comparable values)")

    if mismatch_details:
        print(f"\n  Mismatches:")
        for d in mismatch_details:
            print(d)
    print()


# ── CLI entry point ────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Census extraction helper for opus46 workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show extraction dashboard")
    sub.add_parser("next", help="Print next PDF to extract")

    p_val = sub.add_parser("validate", help="Schema-check a JSON file")
    p_val.add_argument("json_file", type=Path, help="Path to the JSON file")

    p_cmp = sub.add_parser("compare", help="Compare JSON against reference CSV")
    p_cmp.add_argument("json_file", type=Path, help="Path to the JSON file")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status()
    elif args.command == "next":
        cmd_next()
    elif args.command == "validate":
        cmd_validate(args.json_file)
    elif args.command == "compare":
        cmd_compare(args.json_file)


if __name__ == "__main__":
    main()
