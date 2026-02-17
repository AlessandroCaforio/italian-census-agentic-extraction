"""
Extract ind_workers_1927 from structured JSON files with AI-assisted review loop.

Reads structured census JSON files, detects which column contains industrial
worker counts based on null-field signatures (Phase A/B/C), cross-references
against the reference CSV, and supports an iterative correction workflow.

Usage:
    python src/extract_ind_workers.py --extract       # Phase 1: auto-extract + validate
    python src/extract_ind_workers.py --review        # Phase 2: print review tasks
    python src/extract_ind_workers.py --apply         # Phase 3: apply corrections
    python src/extract_ind_workers.py --update-csv    # Phase 4: fill missing CSV values
"""

import argparse
import difflib
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRUCTURED_DIR = PROJECT_ROOT / "structured"
CSV_PATH = PROJECT_ROOT / "pre-war-covariates-complete_updated.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "ind_workers"
PDF_DIR = PROJECT_ROOT / "2page_pdfs"

# ── Historical → modern province mapping ─────────────────────────────────────

PROVINCE_MAP: dict[str, str | None] = {
    # Direct uppercase → title case for most
    # Only non-trivial mappings listed; others auto-derived via title case
    "AQUILA": "L'Aquila",
    "BARI DELLE PUGLIE": "Bari",
    "SPEZIA": "La Spezia",
    "MASSA E CARRARA": "Massa-Carrara",
    "PESARO-URBINO": "Pesaro e Urbino",
    "AOSTA": "Valle d'Aosta",
    "BOLZANO": "Bolzano/Bozen",
    "REGGIO DI CALABRIA": "Reggio di Calabria",
    "REGGIO NELL'EMILIA": "Reggio nell'Emilia",
    "FORLI": "Forli",
    # Not in modern Italy
    "FIUME": None,
    "POLA": None,
}


def map_province(historical_name: str) -> str | None:
    """Map a historical (uppercase) province name to its modern CSV equivalent."""
    if historical_name in PROVINCE_MAP:
        return PROVINCE_MAP[historical_name]
    # Default: title case
    return historical_name.title()


# ── Name normalization ───────────────────────────────────────────────────────


def normalize_name(name: str) -> str:
    """Normalize municipality name for matching.

    - Uppercase
    - Replace accented vowels with VOWEL' (e.g. à → A')
    - Collapse multiple spaces
    - Strip leading/trailing whitespace
    """
    if not name or not isinstance(name, str):
        return ""
    s = name.upper().strip()
    # Replace accented characters: decompose, find combining acute/grave, replace
    # with apostrophe after the base letter
    result = []
    for ch in s:
        decomposed = unicodedata.normalize("NFD", ch)
        if len(decomposed) > 1:
            # Has combining character — keep base letter + apostrophe
            result.append(decomposed[0])
            result.append("'")
        else:
            result.append(ch)
    s = "".join(result)
    # Collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ── Field detection ──────────────────────────────────────────────────────────


def detect_ind_workers_field(comuni: list[dict]) -> tuple[str, str]:
    """Detect which JSON field contains ind_workers (= 'complesso_addetti').

    New-format files (from the corrected 5-column prompt) have the field
    named ``complesso_addetti`` directly.  Old-format files (from the
    legacy 6-column prompt) spread the value across ``industria_esercizi``
    or ``industria_addetti`` depending on which field the LLM left as null.

    Returns (format_label, field_name).
    """
    if not comuni:
        return "EMPTY", "complesso_addetti"

    # New format: direct field name
    if "complesso_addetti" in comuni[0]:
        return "NEW", "complesso_addetti"

    # Legacy format: detect by null-field signature
    vendita_liquori_null = all(c.get("vendita_liquori") is None for c in comuni)
    industria_addetti_null = all(c.get("industria_addetti") is None for c in comuni)
    commercio_esercizi_null = all(c.get("commercio_esercizi") is None for c in comuni)

    if commercio_esercizi_null:
        return "LEGACY_C", "industria_addetti"
    if industria_addetti_null:
        return "LEGACY_B", "industria_esercizi"
    if vendita_liquori_null:
        return "LEGACY_A", "industria_esercizi"
    return "LEGACY_UNK", "industria_esercizi"


# ── CSV reference loading ────────────────────────────────────────────────────


def load_reference_csv() -> pd.DataFrame:
    """Load the reference CSV and add a normalized name column."""
    df = pd.read_csv(CSV_PATH)
    df["_norm_name"] = df["comune_matchable_50"].apply(normalize_name)
    return df


def build_province_lookup(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build a dict mapping modern province name → subset DataFrame."""
    lookup: dict[str, pd.DataFrame] = {}
    for prov, group in df.groupby("province", dropna=True):
        lookup[str(prov)] = group
    return lookup


# ── Name matching ────────────────────────────────────────────────────────────


def match_name(
    norm_name: str,
    province_df: pd.DataFrame | None,
    full_df: pd.DataFrame,
) -> tuple[str, str | None, float | None]:
    """Match a normalized municipality name to the reference CSV.

    Returns (match_status, csv_name, csv_value) where:
    - match_status: 'exact', 'fuzzy', 'no_ref', 'no_province'
    - csv_name: the matched name from the CSV (or None)
    - csv_value: the ind_workers_1927 value from CSV (or None)
    """
    # Try within province first
    search_df = province_df if province_df is not None else full_df

    # 1. Exact match
    exact = search_df[search_df["_norm_name"] == norm_name]
    if len(exact) > 0:
        row = exact.iloc[0]
        val = row["ind_workers_1927"]
        csv_val = float(val) if pd.notna(val) else None
        return "exact", row["comune_matchable_50"], csv_val

    # 2. Fuzzy match
    candidates = search_df["_norm_name"].tolist()
    matches = difflib.get_close_matches(norm_name, candidates, n=1, cutoff=0.85)
    if matches:
        matched_norm = matches[0]
        row = search_df[search_df["_norm_name"] == matched_norm].iloc[0]
        val = row["ind_workers_1927"]
        csv_val = float(val) if pd.notna(val) else None
        return "fuzzy", row["comune_matchable_50"], csv_val

    # 3. If we searched within province, try the full dataset
    if province_df is not None:
        exact_full = full_df[full_df["_norm_name"] == norm_name]
        if len(exact_full) > 0:
            row = exact_full.iloc[0]
            val = row["ind_workers_1927"]
            csv_val = float(val) if pd.notna(val) else None
            return "exact", row["comune_matchable_50"], csv_val

        all_candidates = full_df["_norm_name"].tolist()
        matches_full = difflib.get_close_matches(norm_name, all_candidates, n=1, cutoff=0.85)
        if matches_full:
            matched_norm = matches_full[0]
            row = full_df[full_df["_norm_name"] == matched_norm].iloc[0]
            val = row["ind_workers_1927"]
            csv_val = float(val) if pd.notna(val) else None
            return "fuzzy", row["comune_matchable_50"], csv_val

    return "no_ref", None, None


# ── Value comparison ─────────────────────────────────────────────────────────


def compare_values(extracted: int | None, csv_val: float | None) -> str:
    """Compare extracted value with CSV reference value.

    Returns: 'match', 'mismatch', 'csv_empty', 'extracted_null'
    """
    if extracted is None:
        return "extracted_null"
    if csv_val is None:
        return "csv_empty"

    csv_int = int(round(csv_val))
    if csv_int == 0 and extracted == 0:
        return "match"
    if csv_int == 0 or extracted == 0:
        return "mismatch"

    # Allow 10% tolerance
    ratio = abs(extracted - csv_int) / max(abs(csv_int), abs(extracted))
    if ratio <= 0.10:
        return "match"
    return "mismatch"


# ── Phase 1: Extract ─────────────────────────────────────────────────────────


def cmd_extract() -> None:
    """Extract ind_workers from all structured JSONs and cross-reference with CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading reference CSV...")
    ref_df = load_reference_csv()
    province_lookup = build_province_lookup(ref_df)

    json_files = sorted(STRUCTURED_DIR.glob("pair_*.json"))
    logger.info(f"Found {len(json_files)} structured JSON files")

    rows: list[dict] = []
    file_stats: dict[str, dict] = {}

    for jf in tqdm(json_files, desc="Extracting"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Skipping {jf.name}: {e}")
            file_stats[jf.name] = {"error": str(e)}
            continue

        provincia_hist = data.get("provincia", "")
        comuni = data.get("comuni", [])

        if not comuni:
            logger.warning(f"Skipping {jf.name}: no comuni")
            file_stats[jf.name] = {"error": "no comuni"}
            continue

        phase, field = detect_ind_workers_field(comuni)
        modern_prov = map_province(provincia_hist)
        prov_df = province_lookup.get(modern_prov) if modern_prov else None

        file_match_count = 0
        file_comparable = 0  # rows where CSV has a value to compare against

        for c in comuni:
            comune_name = c.get("comune", "")
            numero_ordine = c.get("numero_ordine")
            extracted_val = c.get(field)

            # Convert to int if possible
            if extracted_val is not None:
                try:
                    extracted_val = int(extracted_val)
                except (ValueError, TypeError):
                    extracted_val = None

            norm = normalize_name(comune_name)
            match_status, csv_name, csv_val = match_name(norm, prov_df, ref_df)

            val_status = compare_values(extracted_val, csv_val)
            if val_status in ("match", "mismatch"):
                file_comparable += 1
                if val_status == "match":
                    file_match_count += 1

            rows.append({
                "source_file": jf.name,
                "provincia": provincia_hist,
                "numero_ordine": numero_ordine,
                "comune": comune_name,
                "ind_workers_1927": extracted_val,
                "phase": phase,
                "csv_match_name": csv_name,
                "csv_value": csv_val,
                "match_status": f"{match_status}:{val_status}",
            })

        match_rate = file_match_count / file_comparable if file_comparable > 0 else 1.0
        file_stats[jf.name] = {
            "provincia": provincia_hist,
            "phase": phase,
            "field": field,
            "total": len(comuni),
            "comparable": file_comparable,
            "matched": file_match_count,
            "match_rate": round(match_rate, 4),
        }

    # Write extraction CSV
    extraction_df = pd.DataFrame(rows)
    extraction_path = OUTPUT_DIR / "ind_workers_extraction.csv"
    extraction_df.to_csv(extraction_path, index=False, encoding="utf-8")
    logger.info(f"Wrote {len(rows)} rows to {extraction_path}")

    # Build review manifest (files with <90% match rate)
    review_files: dict[str, dict] = {}
    for fname, stats in file_stats.items():
        if "error" in stats:
            review_files[fname] = stats
            continue
        if stats["match_rate"] < 0.90:
            # Collect mismatched municipalities for this file
            file_rows = [r for r in rows if r["source_file"] == fname]
            mismatches = [
                {
                    "comune": r["comune"],
                    "extracted": r["ind_workers_1927"],
                    "csv_name": r["csv_match_name"],
                    "csv_value": r["csv_value"],
                    "status": r["match_status"],
                }
                for r in file_rows
                if "mismatch" in r["match_status"] or "no_ref" in r["match_status"]
            ]
            review_files[fname] = {
                **stats,
                "mismatches": mismatches,
            }

    manifest_path = OUTPUT_DIR / "review_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(review_files, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote review manifest with {len(review_files)} files to {manifest_path}")

    # Summary stats
    total_rows = len(rows)

    # Count compound statuses
    status_counts: dict[str, int] = {}
    for r in rows:
        s = r["match_status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    # Aggregate by name-match tier and value-match tier
    name_found = sum(1 for r in rows if not r["match_status"].startswith("no_ref"))
    val_match = sum(1 for r in rows if ":match" in r["match_status"])
    val_mismatch = sum(1 for r in rows if ":mismatch" in r["match_status"])
    val_csv_empty = sum(1 for r in rows if ":csv_empty" in r["match_status"])
    val_extracted_null = sum(1 for r in rows if ":extracted_null" in r["match_status"])
    no_ref = total_rows - name_found

    phase_counts: dict[str, int] = {}
    for r in rows:
        phase_counts[r["phase"]] = phase_counts.get(r["phase"], 0) + 1

    summary = {
        "total_municipalities": total_rows,
        "total_files": len(json_files),
        "files_with_errors": sum(1 for s in file_stats.values() if "error" in s),
        "files_needing_review": len(review_files),
        "status_breakdown": status_counts,
        "phase_counts": phase_counts,
    }

    stats_path = OUTPUT_DIR / "extraction_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    comparable = val_match + val_mismatch
    accuracy = val_match / comparable * 100 if comparable > 0 else 0

    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total municipalities extracted: {total_rows}")
    print(f"Files processed: {len(json_files)}")
    print(f"\nFormat breakdown:")
    for p, cnt in sorted(phase_counts.items()):
        print(f"  {p}: {cnt} municipalities")
    print(f"\nName matching:")
    print(f"  Found in CSV:   {name_found:5d} ({name_found/total_rows*100:.1f}%)")
    print(f"  No reference:   {no_ref:5d} ({no_ref/total_rows*100:.1f}%)")
    print(f"\nValue comparison (where both values exist):")
    print(f"  Match:          {val_match:5d}")
    print(f"  Mismatch:       {val_mismatch:5d}")
    print(f"  Accuracy:       {accuracy:.1f}%")
    print(f"\nOther:")
    print(f"  CSV empty:      {val_csv_empty:5d} (can be filled with --update-csv)")
    print(f"  Extracted null: {val_extracted_null:5d}")
    print(f"\nFiles needing review (<90% match): {len(review_files)}")
    print(f"\nOutputs:")
    print(f"  {extraction_path}")
    print(f"  {manifest_path}")
    print(f"  {stats_path}")


# ── Phase 2: Review ──────────────────────────────────────────────────────────


def cmd_review() -> None:
    """Print actionable review info for files needing attention."""
    manifest_path = OUTPUT_DIR / "review_manifest.json"
    if not manifest_path.exists():
        print("No review manifest found. Run --extract first.")
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    if not manifest:
        print("No files need review. All match rates ≥90%.")
        return

    # Sort by match rate (worst first), errors at top
    def sort_key(item: tuple[str, dict]) -> tuple[int, float]:
        fname, info = item
        if "error" in info:
            return (0, 0.0)
        return (1, info.get("match_rate", 0))

    sorted_files = sorted(manifest.items(), key=sort_key)

    print(f"\n{'='*60}")
    print(f"REVIEW MANIFEST: {len(sorted_files)} files need attention")
    print(f"{'='*60}")

    for fname, info in sorted_files:
        pdf_name = fname.replace(".json", ".pdf")
        pdf_path = PDF_DIR / pdf_name

        print(f"\n{'─'*60}")
        print(f"File: {fname}")

        if "error" in info:
            print(f"  ERROR: {info['error']}")
            print(f"  PDF: {pdf_path}")
            continue

        print(f"  Province: {info['provincia']}")
        print(f"  Format: {info['phase']} (field: {info['field']})")
        print(f"  Match rate: {info['match_rate']*100:.1f}% ({info['matched']}/{info['total']})")
        print(f"  PDF: {pdf_path}")

        mismatches = info.get("mismatches", [])
        if mismatches:
            print(f"  Mismatches ({len(mismatches)}):")
            for m in mismatches[:10]:
                ext = m["extracted"]
                csv = m["csv_value"]
                csv_name = m["csv_name"] or "—"
                print(f"    {m['comune']}: extracted={ext}, csv={csv} (matched: {csv_name}) [{m['status']}]")
            if len(mismatches) > 10:
                print(f"    ... and {len(mismatches) - 10} more")

    print(f"\n{'='*60}")
    print("REVIEW WORKFLOW")
    print(f"{'='*60}")
    print("""
1. For each file above, read the corresponding PDF:
     Read tool → <pdf_path>

2. Verify/correct the extracted values

3. Save corrections to: output/ind_workers/corrections.json
   Format:
   {
     "pair_XXXX.json": {
       "Comune Name": 1234,
       "Another": {"corrected_name": "Fixed Name", "ind_workers": 567}
     }
   }

4. Run: python src/extract_ind_workers.py --apply
""")


# ── Phase 3: Apply corrections ──────────────────────────────────────────────


def cmd_apply() -> None:
    """Apply manual corrections from corrections.json and re-validate."""
    corrections_path = OUTPUT_DIR / "corrections.json"
    extraction_path = OUTPUT_DIR / "ind_workers_extraction.csv"

    if not extraction_path.exists():
        print("No extraction CSV found. Run --extract first.")
        sys.exit(1)

    if not corrections_path.exists():
        print(f"No corrections file found at {corrections_path}")
        print("Create it after reviewing PDFs, then re-run --apply.")
        sys.exit(1)

    with open(corrections_path, encoding="utf-8") as f:
        corrections = json.load(f)

    extraction_df = pd.read_csv(extraction_path)
    ref_df = load_reference_csv()

    before_mismatches = sum(1 for _, r in extraction_df.iterrows() if "mismatch" in str(r["match_status"]))
    before_no_ref = sum(1 for _, r in extraction_df.iterrows() if "no_ref" in str(r["match_status"]))

    applied_count = 0
    name_fixes = 0

    for source_file, file_corrections in corrections.items():
        mask = extraction_df["source_file"] == source_file
        if not mask.any():
            logger.warning(f"Source file {source_file} not found in extraction")
            continue

        for comune_name, correction in file_corrections.items():
            comune_mask = mask & (extraction_df["comune"] == comune_name)
            if not comune_mask.any():
                # Try normalized match
                norm = normalize_name(comune_name)
                comune_mask = mask & (extraction_df["comune"].apply(normalize_name) == norm)

            if not comune_mask.any():
                logger.warning(f"  Municipality '{comune_name}' not found in {source_file}")
                continue

            if isinstance(correction, (int, float)):
                extraction_df.loc[comune_mask, "ind_workers_1927"] = int(correction)
                applied_count += 1
            elif isinstance(correction, dict):
                if "ind_workers" in correction:
                    extraction_df.loc[comune_mask, "ind_workers_1927"] = int(correction["ind_workers"])
                    applied_count += 1
                if "corrected_name" in correction:
                    new_name = correction["corrected_name"]
                    extraction_df.loc[comune_mask, "comune"] = new_name
                    name_fixes += 1

    # Re-validate corrected rows
    province_lookup = build_province_lookup(ref_df)
    updated_count = 0
    for idx, row in extraction_df.iterrows():
        norm = normalize_name(row["comune"])
        modern_prov = map_province(row["provincia"])
        prov_df = province_lookup.get(modern_prov) if modern_prov else None
        status, csv_name, csv_val = match_name(norm, prov_df, ref_df)
        val_status = compare_values(
            int(row["ind_workers_1927"]) if pd.notna(row["ind_workers_1927"]) else None,
            csv_val,
        )
        new_status = f"{status}:{val_status}"
        if new_status != row["match_status"]:
            extraction_df.at[idx, "match_status"] = new_status
            extraction_df.at[idx, "csv_match_name"] = csv_name
            extraction_df.at[idx, "csv_value"] = csv_val
            updated_count += 1

    # Save corrected extraction
    corrected_path = OUTPUT_DIR / "ind_workers_corrected.csv"
    extraction_df.to_csv(corrected_path, index=False, encoding="utf-8")

    after_mismatches = sum(1 for _, r in extraction_df.iterrows() if "mismatch" in str(r["match_status"]))
    after_no_ref = sum(1 for _, r in extraction_df.iterrows() if "no_ref" in str(r["match_status"]))

    correction_stats = {
        "corrections_applied": applied_count,
        "name_fixes": name_fixes,
        "rows_revalidated": updated_count,
        "before": {"mismatches": before_mismatches, "no_ref": before_no_ref},
        "after": {"mismatches": after_mismatches, "no_ref": after_no_ref},
    }

    stats_path = OUTPUT_DIR / "correction_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(correction_stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("CORRECTIONS APPLIED")
    print(f"{'='*60}")
    print(f"Values corrected: {applied_count}")
    print(f"Names fixed: {name_fixes}")
    print(f"Rows revalidated: {updated_count}")
    print(f"\nBefore: {before_mismatches} mismatches, {before_no_ref} no_ref")
    print(f"After:  {after_mismatches} mismatches, {after_no_ref} no_ref")
    print(f"\nOutputs:")
    print(f"  {corrected_path}")
    print(f"  {stats_path}")


# ── Phase 4: Update CSV ─────────────────────────────────────────────────────


def cmd_update_csv(dry_run: bool = False) -> None:
    """Fill empty ind_workers_1927 cells in the reference CSV."""
    # Use corrected CSV if available, otherwise use extraction
    corrected_path = OUTPUT_DIR / "ind_workers_corrected.csv"
    extraction_path = OUTPUT_DIR / "ind_workers_extraction.csv"

    source_path = corrected_path if corrected_path.exists() else extraction_path
    if not source_path.exists():
        print("No extraction data found. Run --extract first.")
        sys.exit(1)

    extraction_df = pd.read_csv(source_path)
    ref_df = pd.read_csv(CSV_PATH)
    ref_df["_norm_name"] = ref_df["comune_matchable_50"].apply(normalize_name)

    # Build lookup: normalized CSV name → extraction row(s)
    # Only use rows with good match status and non-null extracted values
    good_statuses = {"exact:match", "exact:csv_empty", "fuzzy:match", "fuzzy:csv_empty"}
    usable = extraction_df[
        extraction_df["match_status"].isin(good_statuses)
        & extraction_df["ind_workers_1927"].notna()
    ]

    # For csv_empty rows, build a lookup from csv_match_name → extracted value
    fill_lookup: dict[str, int] = {}
    for _, row in usable.iterrows():
        csv_name = row["csv_match_name"]
        if pd.notna(csv_name) and pd.notna(row["ind_workers_1927"]):
            norm = normalize_name(csv_name)
            val = int(row["ind_workers_1927"])
            # If multiple extractions for same municipality, keep the one with
            # a match status (prefer exact over fuzzy)
            if norm not in fill_lookup:
                fill_lookup[norm] = val

    # Find empty cells in ref CSV
    empty_mask = ref_df["ind_workers_1927"].isna()
    empty_count = empty_mask.sum()

    filled = 0
    for idx in ref_df[empty_mask].index:
        norm = ref_df.at[idx, "_norm_name"]
        if norm in fill_lookup:
            if not dry_run:
                ref_df.at[idx, "ind_workers_1927"] = float(fill_lookup[norm])
            filled += 1

    if not dry_run:
        # Drop helper column and save
        ref_df = ref_df.drop(columns=["_norm_name"])
        ref_df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"UPDATE CSV {'(DRY RUN)' if dry_run else ''}")
    print(f"{'='*60}")
    print(f"Source: {source_path.name}")
    print(f"Empty ind_workers_1927 cells: {empty_count}")
    print(f"Cells {'would be' if dry_run else ''} filled: {filled}")
    print(f"Remaining empty: {empty_count - filled}")
    if not dry_run:
        print(f"\nUpdated: {CSV_PATH}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ind_workers_1927 from structured census JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/extract_ind_workers.py --extract        # Auto-extract + validate
  python src/extract_ind_workers.py --review         # Show files needing review
  python src/extract_ind_workers.py --apply          # Apply manual corrections
  python src/extract_ind_workers.py --update-csv     # Fill missing CSV values
  python src/extract_ind_workers.py --update-csv --dry-run  # Preview CSV update
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--extract", action="store_true", help="Extract and validate")
    group.add_argument("--review", action="store_true", help="Print review tasks")
    group.add_argument("--apply", action="store_true", help="Apply corrections")
    group.add_argument("--update-csv", action="store_true", help="Fill missing CSV values")

    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--log-level", default="WARNING", help="Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.extract:
        cmd_extract()
    elif args.review:
        cmd_review()
    elif args.apply:
        cmd_apply()
    elif args.update_csv:
        cmd_update_csv(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
