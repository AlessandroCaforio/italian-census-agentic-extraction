"""
LLM-based Data Structuring for 1927 Italian Census Pipeline.

This module supports Claude Code's native vision capabilities for data extraction.
Claude Code reads page images directly and extracts structured census data.

Workflow:
1. prepare_spreads() - identifies page pairs to process
2. Claude Code reads images using its native vision (Read tool)
3. save_extraction() - saves the extracted JSON
4. validate_extraction() - validates against provincial totals
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Census data schema for reference
#
# The right-hand page of each spread has exactly 5 data columns, visible
# under these headers (the leftmost "Commercio totale" group is partially
# cropped in the scans, showing only its "Addetti" sub-column):
#
#   [Commercio totale]                              → Addetti
#   [Esercizi industriali e commerciali in complesso] → Esercizi, Addetti
#   [Esercizi con vendita di]                        → Vino, Liquori
#
# NOTE: The previous schema incorrectly listed 6 columns (including a
# "Commercio totale - Esercizi" that is never visible in the scans).
# Old JSON files use the legacy field names; see OLD_FIELD_NAMES below.
CENSUS_SCHEMA = {
    "provincia": "string - Nome della provincia",
    "comuni": [
        {
            "numero_ordine": "int",
            "comune": "string",
            "commercio_addetti": "int | null",
            "complesso_esercizi": "int | null",
            "complesso_addetti": "int | null",  # ← ind_workers_1927
            "vendita_vino": "int | null",
            "vendita_liquori": "int | null",
        }
    ],
    "totale_provincia": {
        "commercio_addetti": "int",
        "complesso_esercizi": "int",
        "complesso_addetti": "int",
        "vendita_vino": "int",
        "vendita_liquori": "int",
    },
}

# Legacy field names from the old 6-column prompt. Used to read existing
# structured JSON files that were extracted before the prompt was fixed.
OLD_FIELD_NAMES = [
    "commercio_addetti",
    "commercio_esercizi",
    "industria_esercizi",
    "industria_addetti",
    "vendita_vino",
    "vendita_liquori",
]

NEW_FIELD_NAMES = [
    "commercio_addetti",
    "complesso_esercizi",
    "complesso_addetti",
    "vendita_vino",
    "vendita_liquori",
]

EXTRACTION_PROMPT = """Analizza queste due pagine di un censimento italiano del 1927.
- Pagina sinistra: nomi comuni con numero d'ordine
- Pagina destra: dati numerici corrispondenti

La pagina destra ha esattamente 5 colonne di dati (da sinistra a destra):
1. Commercio totale - Addetti (la colonna più a sinistra, spesso parzialmente tagliata)
2. Esercizi industriali e commerciali in complesso - Esercizi
3. Esercizi industriali e commerciali in complesso - Addetti
4. Esercizi con vendita di Vino
5. Esercizi con vendita di Liquori

L'ultima colonna a destra è il Numero d'ordine (uguale alla pagina sinistra).

IMPORTANTE:
- Ci sono SOLO 5 colonne di dati. NON inventare una sesta colonna.
- Ogni riga nella pagina sinistra corrisponde alla riga con lo stesso numero
  d'ordine nella pagina destra.
- ".." significa dato mancante → usa null
- I numeri possono avere spazi come separatori migliaia (es. "1 216" = 1216)

Estrai i dati in formato JSON con questo schema:
{
  "provincia": "Nome della provincia (es. 'AGRIGENTO')",
  "comuni": [
    {
      "numero_ordine": 1,
      "comune": "Nome Comune",
      "commercio_addetti": 123,
      "complesso_esercizi": 456,
      "complesso_addetti": 789,
      "vendita_vino": 12,
      "vendita_liquori": 3
    }
  ],
  "totale_provincia": {
    "commercio_addetti": 1234,
    "complesso_esercizi": 5678,
    "complesso_addetti": 9012,
    "vendita_vino": 345,
    "vendita_liquori": 67
  }
}

Note critiche:
- Preserva esattamente i nomi dei comuni come scritti (inclusi accenti)
- Se vedi "Totale" o "TOTALE", quelli sono i totali provinciali
- Includi TUTTI i comuni visibili nelle pagine"""


@dataclass
class PageSpread:
    """Represents a pair of census pages (odd + even)."""
    left_page: Path  # Odd page with comune names
    right_page: Path  # Even page with numerical data
    spread_id: str
    output_path: Path

    @property
    def is_processed(self) -> bool:
        return self.output_path.exists()


def prepare_spreads(
    image_dir: Path,
    output_dir: Path,
) -> list[PageSpread]:
    """
    Identify page spreads to process.

    Args:
        image_dir: Directory containing page images
        output_dir: Directory for JSON output

    Returns:
        List of PageSpread objects ready for processing
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images and pair them
    images = sorted(image_dir.glob("*.png"))
    logger.info(f"Found {len(images)} images in {image_dir}")

    if len(images) == 0:
        logger.warning(f"No PNG images found in {image_dir}")
        return []

    # Pair odd and even pages
    spreads = []
    for i in range(0, len(images) - 1, 2):
        left_page = images[i]
        right_page = images[i + 1]
        spread_id = f"{left_page.stem}_{right_page.stem}"
        output_path = output_dir / f"{spread_id}.json"

        spreads.append(PageSpread(
            left_page=left_page,
            right_page=right_page,
            spread_id=spread_id,
            output_path=output_path,
        ))

    logger.info(f"Created {len(spreads)} page spreads")

    # Report status
    processed = sum(1 for s in spreads if s.is_processed)
    pending = len(spreads) - processed
    logger.info(f"Status: {processed} processed, {pending} pending")

    return spreads


def get_pending_spreads(spreads: list[PageSpread]) -> list[PageSpread]:
    """Get spreads that haven't been processed yet."""
    return [s for s in spreads if not s.is_processed]


def save_extraction(
    spread: PageSpread,
    data: dict[str, Any],
) -> Path:
    """
    Save extracted census data to JSON.

    Args:
        spread: The PageSpread being processed
        data: Extracted census data dictionary

    Returns:
        Path to saved JSON file
    """
    # Add metadata
    data["_metadata"] = {
        "left_page": str(spread.left_page),
        "right_page": str(spread.right_page),
        "spread_id": spread.spread_id,
        "status": "success",
    }

    # Save
    spread.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(spread.output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Saved extraction: {spread.output_path.name} "
        f"({len(data.get('comuni', []))} comuni)"
    )

    return spread.output_path


def load_extraction(spread: PageSpread) -> Optional[dict[str, Any]]:
    """Load previously extracted data for a spread."""
    if not spread.is_processed:
        return None

    with open(spread.output_path, encoding="utf-8") as f:
        return json.load(f)


def validate_extraction(result: dict[str, Any]) -> dict[str, Any]:
    """
    Validate extracted census data against provincial totals.

    Args:
        result: Extraction result with comuni and totale_provincia

    Returns:
        Validation report with any discrepancies
    """
    if "error" in result:
        return {"status": "skipped", "reason": "extraction_error"}

    comuni = result.get("comuni", [])
    totals = result.get("totale_provincia", {})

    if not comuni:
        return {"status": "skipped", "reason": "no_comuni"}

    # Detect whether this is old-format (6 fields) or new-format (5 fields)
    sample = comuni[0] if comuni else {}
    if "complesso_esercizi" in sample:
        fields = NEW_FIELD_NAMES
    else:
        fields = OLD_FIELD_NAMES

    calculated_totals = {}
    discrepancies = {}

    for field in fields:
        calculated = sum(
            c.get(field) or 0 for c in comuni if c.get(field) is not None
        )
        calculated_totals[field] = calculated

        reported = totals.get(field) if totals else None

        if reported is not None and calculated != reported:
            discrepancies[field] = {
                "calculated": calculated,
                "reported": reported,
                "difference": calculated - reported,
            }

    return {
        "status": "validated",
        "provincia": result.get("provincia"),
        "comuni_count": len(comuni),
        "calculated_totals": calculated_totals,
        "reported_totals": totals,
        "discrepancies": discrepancies,
        "is_valid": len(discrepancies) == 0,
    }


def load_all_extractions(output_dir: Path) -> list[dict[str, Any]]:
    """Load all extraction JSONs from output directory."""
    results = []
    for json_path in sorted(output_dir.glob("*.json")):
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
                data["_source_file"] = str(json_path)
                results.append(data)
        except Exception as e:
            logger.error(f"Failed to load {json_path}: {e}")
    return results


def print_extraction_prompt():
    """Print the extraction prompt for reference."""
    print("=" * 60)
    print("EXTRACTION PROMPT")
    print("=" * 60)
    print(EXTRACTION_PROMPT)
    print("=" * 60)


def print_spread_info(spread: PageSpread):
    """Print information about a spread to process."""
    print(f"\n{'='*60}")
    print(f"SPREAD: {spread.spread_id}")
    print(f"{'='*60}")
    print(f"Left page (comuni):  {spread.left_page}")
    print(f"Right page (dati):   {spread.right_page}")
    print(f"Output:              {spread.output_path}")
    print(f"Status:              {'✓ Processed' if spread.is_processed else '○ Pending'}")


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if len(sys.argv) < 3:
        print("Usage: python llm_structure.py <image_dir> <output_dir>")
        print("\nThis module prepares page spreads for Claude Code's native vision.")
        print("Run prepare_spreads() to identify pages, then use Claude Code")
        print("to read images and extract data interactively.")
        sys.exit(1)

    image_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    # Prepare spreads
    spreads = prepare_spreads(image_dir, output_dir)

    print(f"\nTotal spreads: {len(spreads)}")
    print(f"Processed: {sum(1 for s in spreads if s.is_processed)}")
    print(f"Pending: {len(get_pending_spreads(spreads))}")

    # Show pending spreads
    pending = get_pending_spreads(spreads)
    if pending:
        print("\nPending spreads:")
        for spread in pending[:5]:  # Show first 5
            print(f"  - {spread.spread_id}")
        if len(pending) > 5:
            print(f"  ... and {len(pending) - 5} more")

    # Print extraction prompt
    print("\n")
    print_extraction_prompt()
