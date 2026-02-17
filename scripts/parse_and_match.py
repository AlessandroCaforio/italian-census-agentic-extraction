"""
Step 2: Parse OCR markdown tables and match against master CSV.

Reads cached OCR results from ocr_cache/, extracts municipality-level data,
computes analfabeti_1921, and matches against pre-war-covariates CSV.

Usage:
    python3 mistral/extraction/popolazione/parse_and_match.py
"""

import json
import os
import re
import csv
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

CACHE_DIR = "mistral/extraction/popolazione/ocr_cache"
MASTER_CSV = "pre-war-covariates-complete_updated.csv"
OUTPUT_CSV = "mistral/extraction/popolazione/popolazione_1921_extracted.csv"
MATCH_REPORT = "mistral/extraction/popolazione/match_report.txt"

# Map region names (from PDF filenames) to province names used in the CSV
# The 1921 census is organized by region, not province, so we need this mapping
REGION_PROVINCES = {
    "Piemonte": ["Torino", "Alessandria", "Cuneo", "Novara"],
    "Lombardia": ["Milano", "Bergamo", "Brescia", "Como", "Cremona", "Mantova", "Pavia", "Sondrio", "Varese"],
    "Veneto": ["Venezia", "Belluno", "Padova", "Rovigo", "Treviso", "Udine", "Verona", "Vicenza"],
    "VeneziaGiulia": ["Trieste", "Gorizia", "Pola", "Zara", "Fiume"],
    "VeneziaTridentina": ["Trento", "Bolzano"],
    "Liguria": ["Genova", "Imperia", "La Spezia", "Savona"],
    "Emilia": ["Bologna", "Ferrara", "Forli", "Modena", "Parma", "Piacenza", "Ravenna", "Reggio Emilia"],
    "Toscana": ["Firenze", "Arezzo", "Grosseto", "Livorno", "Lucca", "Massa", "Pisa", "Pistoia", "Siena"],
    "Marche": ["Ancona", "Ascoli Piceno", "Macerata", "Pesaro"],
    "Umbria": ["Perugia", "Terni"],
    "Lazio": ["Roma", "Frosinone", "Latina", "Rieti", "Viterbo"],
    "AbruzzoMolise": ["L'Aquila", "Chieti", "Pescara", "Teramo", "Campobasso", "Isernia"],
    "Campania": ["Napoli", "Avellino", "Benevento", "Caserta", "Salerno"],
    "Puglie": ["Bari", "Brindisi", "Foggia", "Lecce", "Taranto"],
    "Basilicata": ["Potenza", "Matera"],
    "Calabria": ["Catanzaro", "Cosenza", "Reggio Calabria"],
    "Sicilia": ["Palermo", "Agrigento", "Caltanissetta", "Catania", "Enna", "Messina", "Ragusa", "Siracusa", "Trapani"],
    "Sardegna": ["Cagliari", "Sassari", "Nuoro"],
}


def normalize_name(name: str) -> str:
    """Normalize municipality name for matching."""
    # Remove accents
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # Uppercase, strip, collapse whitespace
    s = re.sub(r"\s+", " ", s.strip().upper())
    # Remove punctuation except apostrophe
    s = re.sub(r"[^\w\s']", "", s)
    return s


def parse_number(s: str) -> int | None:
    """Parse a number that may have space as thousands separator."""
    s = s.strip()
    if not s or s in ("..", "—", "-", "…", ".", ""):
        return None
    # Remove space thousands separators
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)
    # Remove any remaining non-digit chars except minus
    s = re.sub(r"[^\d-]", "", s)
    if not s:
        return None
    return int(s)


def is_subtotal_row(name: str) -> bool:
    """Check if a row is a circondario/province subtotal, not a municipality."""
    lower = name.lower().strip()
    return any(kw in lower for kw in [
        "circondario", "provincia", "totale", "regione",
        "zone agr", "zona agr", "collina", "pianura", "montagna",
    ])


def parse_table_row(row_text: str) -> dict | None:
    """Parse a markdown table row into structured data.

    Expected format: | Name | MF | M | F | MF_lit | M_lit | F_lit | MF% | M% | F% |
    """
    # Split by pipe
    cells = [c.strip() for c in row_text.split("|")]
    # Remove empty first/last from leading/trailing pipes
    cells = [c for c in cells if c]

    if len(cells) < 7:
        return None

    name = cells[0].strip()
    if not name or name == "COMUNI":
        return None

    # Skip header-like rows
    if any(kw in name.upper() for kw in ["ABITANTI", "COMPLESSO", "SAPEVANO", "ETÀ"]):
        return None

    # Skip subtotal rows
    if is_subtotal_row(name):
        return None

    try:
        mf_printed = parse_number(cells[1])   # in complesso MF (printed)
        m_total = parse_number(cells[2])       # in complesso M
        f_total = parse_number(cells[3])       # in complesso F
        mf_literate = parse_number(cells[4])   # sapevano leggere MF
    except (IndexError, ValueError):
        return None

    # Use M+F as popres_1921_tot (matches how the CSV was built)
    # The printed MF column occasionally differs from M+F by a few units
    if m_total is not None and f_total is not None:
        mf_total = m_total + f_total
    elif mf_printed is not None:
        mf_total = mf_printed
    else:
        return None

    # Compute illiterates
    analfabeti = None
    if mf_total is not None and mf_literate is not None:
        analfabeti = mf_total - mf_literate

    return {
        "comune_raw": name,
        "popres_1921_tot": mf_total,
        "popres_1921_f": f_total,
        "analfabeti_1921": analfabeti,
        "mf_literate": mf_literate,  # keep for verification
    }


def parse_region_ocr(region: str, cache_path: str) -> list[dict]:
    """Parse all pages of a region's OCR output."""
    with open(cache_path) as f:
        data = json.load(f)

    municipalities = []
    current_circondario = None

    for page in data["pages"]:
        markdown = page["markdown"]

        # Detect circondario headers
        circ_matches = re.findall(
            r"CIRCONDARIO\s+DI\s+(.+?)[\.\s]*$",
            markdown, re.MULTILINE | re.IGNORECASE
        )

        for line in markdown.split("\n"):
            line = line.strip()
            if not line or not line.startswith("|"):
                # Check for circondario header outside table
                m = re.match(r"CIRCONDARIO\s+DI\s+(.+?)[\.\s]*$", line, re.IGNORECASE)
                if m:
                    current_circondario = m.group(1).strip()
                continue

            # Skip separator rows
            if re.match(r"\|\s*-+", line):
                continue

            parsed = parse_table_row(line)
            if parsed:
                parsed["region"] = region
                parsed["circondario"] = current_circondario
                municipalities.append(parsed)

    return municipalities


def load_master_csv() -> dict:
    """Load master CSV and return dict of normalized_name -> row."""
    rows = {}
    with open(MASTER_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = normalize_name(row["comune_matchable_50"])
            rows[name] = row
    return rows


def fuzzy_match(name: str, candidates: list[str], threshold: float = 0.75) -> tuple[str | None, float]:
    """Find best fuzzy match for a name among candidates.
    Uses quick_ratio as a pre-filter to avoid expensive full comparisons."""
    best_match = None
    best_score = 0.0

    for candidate in candidates:
        sm = SequenceMatcher(None, name, candidate)
        # Quick pre-filter: skip if upper bound is below current best
        if sm.quick_ratio() < best_score:
            continue
        score = sm.ratio()
        if score > best_score:
            best_score = score
            best_match = candidate

    if best_score >= threshold:
        return best_match, best_score
    return None, best_score


def main():
    # Step 1: Parse all OCR results
    print("=== Parsing OCR results ===\n")
    all_municipalities = []

    cache_files = sorted(Path(CACHE_DIR).glob("*.json"))
    for cache_path in cache_files:
        region = cache_path.stem
        munis = parse_region_ocr(region, str(cache_path))
        print(f"  {region}: {len(munis)} municipalities")
        all_municipalities.extend(munis)

    print(f"\nTotal extracted: {len(all_municipalities)} municipalities\n")

    # Step 2: Load master CSV
    print("=== Loading master CSV ===\n")
    master = load_master_csv()
    # Build province-to-region mapping for filtering
    csv_by_region = {}
    with open(MASTER_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            region = row.get("region", "")
            if region not in csv_by_region:
                csv_by_region[region] = []
            csv_by_region[region].append(normalize_name(row["comune_matchable_50"]))

    print(f"  Master CSV: {len(master)} municipalities\n")

    # Build region-filtered candidate pools for faster fuzzy matching
    # Map each region to the set of CSV municipality names in its provinces
    region_candidates = {}
    with open(MASTER_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            norm_name = normalize_name(row["comune_matchable_50"])
            csv_region = row.get("region", "")
            if csv_region not in region_candidates:
                region_candidates[csv_region] = []
            region_candidates[csv_region].append(norm_name)

    def get_candidates_for_region(region: str) -> list[str]:
        """Get CSV municipality names for provinces in the given region."""
        candidates = []
        provinces = REGION_PROVINCES.get(region, [])
        # Match CSV region field (which is the same as our region names)
        if region in region_candidates:
            candidates.extend(region_candidates[region])
        # Also check by individual region name variations
        for csv_region, names in region_candidates.items():
            if csv_region in (region, region.replace("_", " ")):
                continue  # already added
            # Check if any of our province names match
            for prov in provinces:
                if prov.lower() in csv_region.lower():
                    candidates.extend(names)
                    break
        return list(set(candidates)) if candidates else list(master.keys())

    # Step 3: Match extracted municipalities against master CSV
    print("=== Matching municipalities ===\n")

    matched = []
    unmatched_ocr = []
    all_master_names = list(master.keys())

    for muni in all_municipalities:
        norm = normalize_name(muni["comune_raw"])

        # Try exact match first
        if norm in master:
            muni["csv_match"] = norm
            muni["match_score"] = 1.0
            muni["match_type"] = "exact"
            matched.append(muni)
            continue

        # Try fuzzy match — first within region, then globally
        region = muni.get("region", "")
        candidates = get_candidates_for_region(region)
        match_name, score = fuzzy_match(norm, candidates)

        # If regional match is poor, try global
        if not match_name or score < 0.85:
            global_match, global_score = fuzzy_match(norm, all_master_names)
            if global_score > score:
                match_name, score = global_match, global_score

        if match_name and score >= 0.85:
            muni["csv_match"] = match_name
            muni["match_score"] = score
            muni["match_type"] = "fuzzy"
            matched.append(muni)
        else:
            muni["csv_match"] = None
            muni["match_score"] = score
            muni["match_type"] = "none"
            unmatched_ocr.append(muni)

    # Step 4: Find CSV municipalities that we didn't match
    matched_csv_names = {m["csv_match"] for m in matched if m["csv_match"]}
    unmatched_csv = [name for name in all_master_names if name not in matched_csv_names]

    # Step 5: Cross-check values where both OCR and CSV have data
    value_checks = []
    for muni in matched:
        csv_name = muni["csv_match"]
        if not csv_name or csv_name not in master:
            continue
        csv_row = master[csv_name]

        checks = {}
        for field in ["popres_1921_tot", "popres_1921_f", "analfabeti_1921"]:
            ocr_val = muni.get(field)
            csv_val_str = csv_row.get(field, "")
            csv_val = None
            if csv_val_str and csv_val_str.strip():
                try:
                    csv_val = int(float(csv_val_str))
                except (ValueError, TypeError):
                    pass

            if ocr_val is not None and csv_val is not None:
                match = ocr_val == csv_val
                checks[field] = {"ocr": ocr_val, "csv": csv_val, "match": match}

        if checks:
            value_checks.append({"name": muni["comune_raw"], "csv_name": csv_name, "checks": checks})

    # Step 6: Print summary
    n_exact = sum(1 for m in matched if m["match_type"] == "exact")
    n_fuzzy = sum(1 for m in matched if m["match_type"] == "fuzzy")

    print(f"  Exact matches:     {n_exact}")
    print(f"  Fuzzy matches:     {n_fuzzy}")
    print(f"  Unmatched (OCR):   {len(unmatched_ocr)}")
    print(f"  Unmatched (CSV):   {len(unmatched_csv)}")

    # Value accuracy
    total_checks = 0
    total_correct = 0
    for vc in value_checks:
        for field, info in vc["checks"].items():
            total_checks += 1
            if info["match"]:
                total_correct += 1

    if total_checks > 0:
        accuracy = total_correct / total_checks * 100
        print(f"\n  Value accuracy: {total_correct}/{total_checks} ({accuracy:.1f}%)")

    # Step 7: Write output CSV
    print(f"\n=== Writing output ===\n")

    fieldnames = [
        "comune_raw", "comune_matched", "match_type", "match_score",
        "region", "circondario",
        "popres_1921_tot", "popres_1921_f", "analfabeti_1921",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for muni in all_municipalities:
            writer.writerow({
                "comune_raw": muni["comune_raw"],
                "comune_matched": muni.get("csv_match", ""),
                "match_type": muni.get("match_type", "none"),
                "match_score": f"{muni.get('match_score', 0):.3f}",
                "region": muni.get("region", ""),
                "circondario": muni.get("circondario", ""),
                "popres_1921_tot": muni.get("popres_1921_tot", ""),
                "popres_1921_f": muni.get("popres_1921_f", ""),
                "analfabeti_1921": muni.get("analfabeti_1921", ""),
            })

    print(f"  Extracted data: {OUTPUT_CSV}")

    # Step 8: Write detailed match report
    with open(MATCH_REPORT, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("POPULATION 1921 — EXTRACTION & MATCHING REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total OCR municipalities:  {len(all_municipalities)}\n")
        f.write(f"Exact matches:             {n_exact}\n")
        f.write(f"Fuzzy matches:             {n_fuzzy}\n")
        f.write(f"Unmatched (OCR):           {len(unmatched_ocr)}\n")
        f.write(f"Unmatched (CSV):           {len(unmatched_csv)}\n")
        if total_checks > 0:
            f.write(f"Value accuracy:            {total_correct}/{total_checks} ({accuracy:.1f}%)\n")
        f.write("\n")

        # Fuzzy matches (for manual review)
        f.write("-" * 80 + "\n")
        f.write("FUZZY MATCHES (review these)\n")
        f.write("-" * 80 + "\n\n")
        fuzzy_matches = [m for m in matched if m["match_type"] == "fuzzy"]
        fuzzy_matches.sort(key=lambda x: x["match_score"])
        for m in fuzzy_matches:
            f.write(f"  OCR: {m['comune_raw']:<35} -> CSV: {m['csv_match']:<35} ({m['match_score']:.1%})\n")

        # Unmatched OCR municipalities
        f.write("\n" + "-" * 80 + "\n")
        f.write("UNMATCHED OCR MUNICIPALITIES\n")
        f.write("-" * 80 + "\n\n")
        for m in unmatched_ocr:
            f.write(f"  {m['comune_raw']:<40} (region: {m['region']})\n")

        # Value mismatches
        mismatches = [vc for vc in value_checks if any(not v["match"] for v in vc["checks"].values())]
        if mismatches:
            f.write("\n" + "-" * 80 + "\n")
            f.write("VALUE MISMATCHES (OCR vs CSV)\n")
            f.write("-" * 80 + "\n\n")
            for vc in mismatches:
                f.write(f"  {vc['name']} (matched: {vc['csv_name']})\n")
                for field, info in vc["checks"].items():
                    if not info["match"]:
                        diff = info["ocr"] - info["csv"]
                        f.write(f"    {field}: OCR={info['ocr']}, CSV={info['csv']} (diff={diff:+d})\n")
                f.write("\n")

    print(f"  Match report:   {MATCH_REPORT}")
    print("\nDone!")


if __name__ == "__main__":
    main()
