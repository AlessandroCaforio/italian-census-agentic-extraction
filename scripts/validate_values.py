"""
Validation helper: tracks progress and manages the validation output.

Usage:
    # Check progress
    python3 mistral/extraction/popolazione/validate_values.py status

    # Get next batch of pages to validate
    python3 mistral/extraction/popolazione/validate_values.py next [region]

    # After visual validation, record results
    python3 mistral/extraction/popolazione/validate_values.py record <json_results>

    # Final report
    python3 mistral/extraction/popolazione/validate_values.py report
"""

import csv
import json
import os
import sys

LOOKUP = "mistral/extraction/popolazione/validation_lookup.json"
OUTPUT = "mistral/extraction/popolazione/validated_fill_1921.csv"
PROGRESS = "mistral/extraction/popolazione/validation_progress.json"

OUTPUT_FIELDS = ["csv_name", "popres_1921_tot", "popres_1921_f", "analfabeti_1921", "status", "note"]


def load_lookup():
    with open(LOOKUP) as f:
        return json.load(f)


def load_progress():
    if os.path.exists(PROGRESS):
        with open(PROGRESS) as f:
            return json.load(f)
    return {"completed_pages": [], "results": []}


def save_progress(progress):
    with open(PROGRESS, "w") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def cmd_status():
    lookup = load_lookup()
    progress = load_progress()

    total_pages = len(lookup)
    done_pages = len(progress["completed_pages"])
    total_munis = sum(len(v) for v in lookup.values())
    done_munis = len(progress["results"])

    print(f"Pages: {done_pages}/{total_pages} ({done_pages/total_pages*100:.0f}%)")
    print(f"Municipalities validated: {done_munis}/{total_munis}")

    if done_munis > 0:
        confirmed = sum(1 for r in progress["results"] if r["status"] == "CONFIRMED")
        corrected = sum(1 for r in progress["results"] if r["status"] == "CORRECTED")
        not_found = sum(1 for r in progress["results"] if r["status"] == "NOT_FOUND")
        print(f"  CONFIRMED: {confirmed}  CORRECTED: {corrected}  NOT_FOUND: {not_found}")

    # Show remaining by region
    done_set = set(progress["completed_pages"])
    remaining = {}
    for key in lookup:
        if key not in done_set:
            region = key.rsplit("_p", 1)[0]
            if region not in remaining:
                remaining[region] = 0
            remaining[region] += 1

    if remaining:
        print(f"\nRemaining pages by region:")
        for region in sorted(remaining):
            print(f"  {region}: {remaining[region]} pages")


def cmd_next(region_filter=None):
    lookup = load_lookup()
    progress = load_progress()
    done_set = set(progress["completed_pages"])

    # Find next pages to validate
    batch = []
    for key in sorted(lookup.keys()):
        if key in done_set:
            continue
        region = key.rsplit("_p", 1)[0]
        if region_filter and region != region_filter:
            continue
        batch.append(key)
        if len(batch) >= 3:  # batch of 3 pages
            break

    if not batch:
        print("All pages validated!" if not region_filter else f"All {region_filter} pages done!")
        return

    for key in batch:
        region = key.rsplit("_p", 1)[0]
        page = int(key.rsplit("_p", 1)[1])
        munis = lookup[key]

        pdf_path = f"PDF_folders copia/Popolazione 1921/extracted/{region}_popolazione_extracted.pdf"
        print(f"\n{'='*60}")
        print(f"READ: {pdf_path} page {page + 1}")
        print(f"Key: {key}")
        print(f"Municipalities to check ({len(munis)}):")
        print(f"{'='*60}")
        for m in munis:
            print(f"  {m['search_name']:<35} tot={m['popres_1921_tot']:>6}  f={m['popres_1921_f']:>6}  analf={m['analfabeti_1921']:>6}")


def cmd_record(results_json):
    progress = load_progress()
    results = json.loads(results_json)

    for r in results.get("validated", []):
        progress["results"].append(r)

    if "page_key" in results:
        progress["completed_pages"].append(results["page_key"])

    save_progress(progress)
    print(f"Recorded {len(results.get('validated', []))} results. Page {results.get('page_key', '?')} marked complete.")


def cmd_report():
    progress = load_progress()
    results = progress["results"]

    if not results:
        print("No results yet.")
        return

    confirmed = [r for r in results if r["status"] == "CONFIRMED"]
    corrected = [r for r in results if r["status"] == "CORRECTED"]
    not_found = [r for r in results if r["status"] == "NOT_FOUND"]

    print(f"Total validated: {len(results)}")
    print(f"  CONFIRMED: {len(confirmed)} ({len(confirmed)/len(results)*100:.1f}%)")
    print(f"  CORRECTED: {len(corrected)} ({len(corrected)/len(results)*100:.1f}%)")
    print(f"  NOT_FOUND: {len(not_found)} ({len(not_found)/len(results)*100:.1f}%)")

    # Write final CSV
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for r in sorted(results, key=lambda x: x["csv_name"]):
            writer.writerow({k: r.get(k, "") for k in OUTPUT_FIELDS})

    print(f"\nWritten: {OUTPUT}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        cmd_status()
    elif cmd == "next":
        region = sys.argv[2] if len(sys.argv) > 2 else None
        cmd_next(region)
    elif cmd == "record":
        cmd_record(sys.argv[2])
    elif cmd == "report":
        cmd_report()
    else:
        print(f"Unknown command: {cmd}")
