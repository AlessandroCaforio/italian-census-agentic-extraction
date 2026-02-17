# Census Extraction Workflow (opus46)

## Trigger

When the user says **"extract"**, **"next"**, or **"estrai"**, start the extraction workflow below.

## Census Page Layout (5 columns)

Each PDF page shows a table for one province. The 5 data columns are:

```
┌──────────┬──────────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
│ N.       │ COMUNI               │ Col 1     │ Col 2     │ Col 3     │ Col 4     │ Col 5     │
│ d'ordine │                      │ Complesso │Complesso  │Commercio  │Commercio  │ Vendita   │
│          │                      │ Esercizi  │ Addetti   │ Esercizi  │ Addetti   │ vino      │
├──────────┼──────────────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ 1        │ Municipality Name    │ 234       │ 1567      │ 89        │ 456       │ 12        │
│ 2        │ Another Name         │ 123       │ 890       │ 45        │ 234       │ 8         │
│ ...      │ ...                  │ ...       │ ...       │ ...       │ ...       │ ...       │
├──────────┼──────────────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│          │ Totale Provincia     │ XXXX      │ XXXX      │ XXXX      │ XXXX      │ XXXX      │
└──────────┴──────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
```

**Some pages have a 6th column** `vendita_liquori` (rightmost). Include it when present, use `null` when absent.

**Key rule:** Col 2 (`complesso_addetti`) is almost always the LARGEST number in each row. Use this as a sanity check.

## JSON Schema

```json
{
  "provincia": "PROVINCE NAME (uppercase, as printed)",
  "comuni": [
    {
      "numero_ordine": 1,
      "comune": "Municipality Name",
      "complesso_esercizi": 234,
      "complesso_addetti": 1567,
      "commercio_esercizi": 89,
      "commercio_addetti": 456,
      "vendita_vino": 12,
      "vendita_liquori": null
    }
  ],
  "totale_provincia": {
    "complesso_esercizi": 9999,
    "complesso_addetti": 9999,
    "commercio_esercizi": 9999,
    "commercio_addetti": 9999,
    "vendita_vino": 9999,
    "vendita_liquori": null
  }
}
```

**Field types:** All numeric fields are `int` or `null`. Never use strings for numbers.

## Step-by-Step Workflow

```
1. python3 opus46/census_extractor.py status     → See progress
2. python3 opus46/census_extractor.py next        → Get next PDF path
3. Read the PDF (use Read tool on the PDF file)
4. Extract all rows into JSON following the schema above
5. Write JSON to opus46/pair_XXXX.json
6. python3 opus46/census_extractor.py validate opus46/pair_XXXX.json
7. python3 opus46/census_extractor.py compare  opus46/pair_XXXX.json
8. If issues found → fix and re-validate
9. If learnings → append to opus46/notes.md
```

## Edge Cases

- **".." in source** → `null` (census convention for "not available")
- **Space thousands separator** → "1 234" means `1234`
- **"Segue:" prefix** → Continuation page, same province as previous PDF
- **Column 1 often cropped** → Left margin cuts into `complesso_esercizi`. Double-check.
- **Noise column from facing page** → On some right-side pages, skewed scanning causes numbers
  from the facing page to appear as an extra column on the far LEFT. **Detection:**
  the column gets progressively truncated toward the bottom (3-digit → 2-digit → 1-digit).
  **Action:** exclude the column entirely, set corresponding field to `null`.
- **vendita_liquori absent** → Use `null` for all rows. This is normal for many provinces.
- **Multi-file provinces** → `numero_ordine` continues across PDFs.
- **totale_provincia** → Only include if the total row is visible on the page.

## Digit Confusion in 1927 Typeface

The census typeface causes systematic reading errors. **Always double-check** these digit pairs:
- **0 ↔ 9** (most common)
- **0 ↔ 6** (frequent)
- **2 ↔ 3** (occasional)

When `compare` flags a mismatch, check if the difference can be explained by these confusions.
If so, trust the CSV value as the correction.

## Quality Checks

After each extraction, verify:
1. `validate` passes (schema + totals match sums)
2. `compare` accuracy ≥ 90% for provinces with CSV data
3. `complesso_addetti` ≥ `commercio_addetti` for every row (addetti is a superset)
4. `numero_ordine` is sequential (1, 2, 3, ...)
