"""
Municipality Crosswalk Builder (1921–1954).

Maps each of the ~7,803 municipalities at 1950 backward to the 1921, 1927,
and 1929 census structures, classifying each as USABLE or DISCARD for a
balanced panel.

Data sources:
    - ANPR_archivio_comuni.csv       — primary timeline (alive sets)
    - sistat_data/comuni.tab         — sistat_id ↔ ISTAT bridge
    - sistat_data/event_extinction.tab — extinction destinations
    - sistat_data/event_creation.tab — creation sources
    - sistat_data/event_territory_*.tab — territory transfers
    - sistat_data/comuni_variations.tab — event dates
    - pre-war-covariates-complete_updated.csv — target 1950 structure
"""

import logging
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CENSUS_DATES = {
    "1921": pd.Timestamp("1921-12-01"),
    "1927": pd.Timestamp("1927-10-15"),
    "1929": pd.Timestamp("1929-10-15"),
}
REFERENCE_DATE = pd.Timestamp("1950-06-15")

EXPECTED_ALIVE = {"1921": 9202, "1927": 8928, "1929": 7308, "ref": 7803}

# Provinces ceded after WWII (Fiume, Istria, Zara)
# 701-703 are the 1927+ codes; 801-802 are the pre-1923 Venezia Giulia codes
LOST_PROVINCE_PREFIXES = {"701", "702", "703", "801", "802"}

MAX_CHAIN_DEPTH = 10


# ── Helpers ──────────────────────────────────────────────────────────────


def _pad_istat(code) -> str:
    """Zero-pad an ISTAT code to 6 digits."""
    if pd.isna(code):
        return ""
    return str(int(float(code))).zfill(6)


def _normalize_name(name: str) -> str:
    """Normalize a municipality name for fuzzy matching."""
    if pd.isna(name):
        return ""
    s = str(name).upper().strip()
    # Normalize unicode accents to ASCII equivalents
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    # Standardize apostrophes and punctuation
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("`", "'")
    # Remove extra whitespace
    s = " ".join(s.split())
    return s


def _normalize_for_matching(name: str) -> str:
    """Aggressive normalization for matching: strips apostrophes and hyphens."""
    s = _normalize_name(name)
    # Remove all apostrophes (inconsistent between ANPR and CSV sources)
    s = s.replace("'", "")
    # Remove hyphens (e.g. SAN DORLIGO DELLA VALLE-DOLINA vs without)
    s = s.replace("-", " ")
    s = " ".join(s.split())
    return s


def _province_from_codistat(code: str) -> str:
    """Extract 3-digit province prefix from a 6-digit CODISTAT."""
    return code[:3]


# ── Step 1: Load & build alive sets ─────────────────────────────────────


def load_anpr(path: Path) -> pd.DataFrame:
    """Load ANPR archive, parse dates, zero-pad codes."""
    logger.info("Loading ANPR archive …")
    # keep_default_na=False prevents pandas from treating "NA" (Napoli's
    # province sigla) as NaN — a classic pandas gotcha.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Replace '9999-12-31' (still-alive sentinel) with a representable date
    # to avoid pandas Timestamp nanosecond overflow
    df["DATACESSAZIONE"] = df["DATACESSAZIONE"].replace("9999-12-31", "2200-01-01")
    df["DATAISTITUZIONE"] = pd.to_datetime(df["DATAISTITUZIONE"], errors="coerce")
    df["DATACESSAZIONE"] = pd.to_datetime(df["DATACESSAZIONE"], errors="coerce")
    df["CODISTAT"] = df["CODISTAT"].str.strip().str.zfill(6)
    df["CODCATASTALE"] = df["CODCATASTALE"].str.strip()
    df["DENOMTRASLITTERATA"] = df["DENOMTRASLITTERATA"].str.strip()
    df["SIGLAPROVINCIA"] = df["SIGLAPROVINCIA"].str.strip()
    logger.info(f"  {len(df)} ANPR records loaded")
    return df


def build_alive_set(anpr: pd.DataFrame, date: pd.Timestamp) -> set[str]:
    """Return set of CODISTAT codes alive at a given date."""
    mask = (anpr["DATAISTITUZIONE"] <= date) & (anpr["DATACESSAZIONE"] >= date)
    codes = set(anpr.loc[mask, "CODISTAT"].unique())
    return codes


def step1_alive_sets(anpr: pd.DataFrame) -> dict[str, set[str]]:
    """Build alive sets for each census year and the 1950 reference."""
    alive = {}
    for label, date in CENSUS_DATES.items():
        alive[label] = build_alive_set(anpr, date)
        expected = EXPECTED_ALIVE[label]
        actual = len(alive[label])
        logger.info(f"  Alive at {label}: {actual} (expected {expected})")
        if actual != expected:
            logger.warning(f"  ⚠ Count mismatch for {label}: {actual} ≠ {expected}")

    alive["ref"] = build_alive_set(anpr, REFERENCE_DATE)
    actual_ref = len(alive["ref"])
    logger.info(f"  Alive at 1950 ref: {actual_ref} (expected {EXPECTED_ALIVE['ref']})")
    if actual_ref != EXPECTED_ALIVE["ref"]:
        logger.warning(f"  ⚠ Count mismatch for ref: {actual_ref} ≠ {EXPECTED_ALIVE['ref']}")

    return alive


# ── Step 2: Build forward mapping ───────────────────────────────────────


def load_sistat(root: Path) -> dict:
    """Load all SISTAT tables, return dict of DataFrames."""
    data = {}

    data["comuni"] = pd.read_csv(root / "comuni.tab", sep="\t")
    data["comuni"]["last_istat_cod_str"] = (
        data["comuni"]["last_istat_cod"].apply(_pad_istat)
    )

    data["extinction"] = pd.read_csv(root / "event_extinction.tab", sep="\t")
    data["extinction"]["event_to_istat_cod_str"] = (
        data["extinction"]["event_to_istat_cod"].apply(_pad_istat)
    )

    data["creation"] = pd.read_csv(root / "event_creation.tab", sep="\t")
    data["creation"]["event_from_istat_cod_str"] = (
        data["creation"]["event_from_istat_cod"].apply(_pad_istat)
    )

    data["variations"] = pd.read_csv(root / "comuni_variations.tab", sep="\t")
    data["variations"]["event_date"] = pd.to_datetime(
        data["variations"]["event_date"], errors="coerce"
    )

    data["territory_acq"] = pd.read_csv(
        root / "event_territory_acquisition.tab", sep="\t"
    )
    data["territory_ces"] = pd.read_csv(
        root / "event_territory_cession.tab", sep="\t"
    )

    # Build sistat_id → ISTAT code map
    data["sid_to_istat"] = dict(
        zip(data["comuni"]["sistat_id"], data["comuni"]["last_istat_cod_str"])
    )
    # Build ISTAT → sistat_id map
    data["istat_to_sid"] = dict(
        zip(data["comuni"]["last_istat_cod_str"], data["comuni"]["sistat_id"])
    )

    logger.info(
        f"  SISTAT loaded: {len(data['comuni'])} comuni, "
        f"{len(data['extinction'])} extinction rows, "
        f"{len(data['creation'])} creation rows"
    )
    return data


def _build_catastale_bridge(anpr: pd.DataFrame, ref_codes: set[str]) -> dict[str, str]:
    """
    Build mapping from non-ref CODISTAT → ref CODISTAT via shared CODCATASTALE.

    This handles province renumbering: same municipality, same cadastral code,
    but different numeric CODISTAT because the province prefix changed.
    """
    bridge = {}

    # Group unique CODISTAT codes by CODCATASTALE (vectorized)
    valid = anpr[anpr["CODCATASTALE"].notna() & (anpr["CODCATASTALE"] != "")]
    cat_groups: dict[str, set[str]] = defaultdict(set)
    for cat, code in zip(valid["CODCATASTALE"], valid["CODISTAT"]):
        cat_groups[cat].add(code)

    for cat, codes in cat_groups.items():
        ref_hits = [c for c in codes if c in ref_codes]
        non_ref = [c for c in codes if c not in ref_codes]
        if len(ref_hits) == 1:
            for nr in non_ref:
                if nr not in bridge:
                    bridge[nr] = ref_hits[0]

    logger.info(f"  CODCATASTALE bridge: {len(bridge)} mappings")
    return bridge


def _build_sistat_extinction_map(
    sistat: dict, ref_codes: set[str]
) -> dict[str, list[str]]:
    """
    Build mapping from extinct ISTAT code → list of destination ISTAT codes
    (following chains up to MAX_CHAIN_DEPTH).

    Returns destinations that exist in ref_codes.
    """
    # Build raw extinction graph: source_istat → [dest_istat, ...]
    ext_df = sistat["extinction"]
    raw_graph: dict[str, list[str]] = defaultdict(list)

    for _, row in ext_df.iterrows():
        src_sid = row["sistat_id"]
        src_istat = sistat["sid_to_istat"].get(src_sid, "")
        dst_istat = row["event_to_istat_cod_str"]
        if src_istat and dst_istat:
            raw_graph[src_istat].append(dst_istat)

    # Follow chains: resolve each source to final ref destinations
    resolved: dict[str, list[str]] = {}

    def _resolve(code: str, visited: set[str], depth: int) -> list[str]:
        if code in ref_codes:
            return [code]
        if depth >= MAX_CHAIN_DEPTH or code in visited:
            return []
        if code in resolved:
            return resolved[code]
        visited.add(code)
        destinations = []
        for dst in raw_graph.get(code, []):
            destinations.extend(_resolve(dst, visited, depth + 1))
        return destinations

    for src in raw_graph:
        if src not in ref_codes:
            resolved[src] = _resolve(src, set(), 0)

    mapped_count = sum(1 for v in resolved.values() if v)
    logger.info(f"  SISTAT extinction map: {len(resolved)} extinct codes, {mapped_count} resolved")
    return resolved


def step2_forward_mapping(
    anpr: pd.DataFrame,
    sistat: dict,
    alive: dict[str, set[str]],
) -> dict[str, dict[str, dict]]:
    """
    For each census year, build forward mapping: census_code → {
        'dest': list[str],   # 1950 CODISTAT(s)
        'method': str,       # DIRECT / CATASTALE / SISTAT / ANPR_FALLBACK / LOST_TERRITORY / UNTRACED
    }
    """
    ref_codes = alive["ref"]

    # Pre-compute bridges
    cat_bridge = _build_catastale_bridge(anpr, ref_codes)
    ext_map = _build_sistat_extinction_map(sistat, ref_codes)

    # Pre-build ANPR fallback indexes for fast lookups
    # CODCATASTALE → set of CODISTAT codes
    cat_to_codes: dict[str, set[str]] = defaultdict(set)
    for cat, code in zip(anpr["CODCATASTALE"], anpr["CODISTAT"]):
        if pd.notna(cat) and cat:
            cat_to_codes[cat].add(code)

    # CODISTAT → (CODCATASTALE, DENOMTRASLITTERATA)
    code_to_info: dict[str, tuple[str, str]] = {}
    for code, cat, name in zip(
        anpr["CODISTAT"], anpr["CODCATASTALE"], anpr["DENOMTRASLITTERATA"]
    ):
        if code not in code_to_info:
            code_to_info[code] = (
                cat if pd.notna(cat) else "",
                name if pd.notna(name) else "",
            )

    # (province_prefix, normalized_name) → set of CODISTAT in ref_codes
    prov_name_to_ref: dict[tuple[str, str], set[str]] = defaultdict(set)
    for code in ref_codes:
        info = code_to_info.get(code)
        if info:
            prov = _province_from_codistat(code)
            norm_name = _normalize_name(info[1])
            prov_name_to_ref[(prov, norm_name)].add(code)

    forward = {}

    for year, census_codes in {
        y: alive[y] for y in CENSUS_DATES
    }.items():
        logger.info(f"  Building forward mapping for {year} …")
        year_map: dict[str, dict] = {}
        stats = defaultdict(int)

        for code in sorted(census_codes):
            # 1) Direct: alive at both dates
            if code in ref_codes:
                year_map[code] = {"dest": [code], "method": "DIRECT"}
                stats["DIRECT"] += 1
                continue

            # 2) CODCATASTALE bridge
            if code in cat_bridge:
                dest = cat_bridge[code]
                year_map[code] = {"dest": [dest], "method": "CATASTALE"}
                stats["CATASTALE"] += 1
                continue

            # 3) SISTAT extinction chain
            if code in ext_map and ext_map[code]:
                dests = list(set(ext_map[code]))  # deduplicate
                year_map[code] = {"dest": dests, "method": "SISTAT"}
                stats["SISTAT"] += 1
                continue

            # 4) Lost territory check
            prov = _province_from_codistat(code)
            if prov in LOST_PROVINCE_PREFIXES:
                year_map[code] = {"dest": [], "method": "LOST_TERRITORY"}
                stats["LOST_TERRITORY"] += 1
                continue

            # 5) ANPR fallback via pre-indexed lookups
            resolved = _anpr_fallback_fast(
                code, ref_codes, cat_bridge,
                code_to_info, cat_to_codes, prov_name_to_ref,
            )
            if resolved:
                year_map[code] = {"dest": [resolved], "method": "ANPR_FALLBACK"}
                stats["ANPR_FALLBACK"] += 1
                continue

            # 6) Untraced
            year_map[code] = {"dest": [], "method": "UNTRACED"}
            stats["UNTRACED"] += 1

        forward[year] = year_map
        logger.info(f"    {year}: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))

    return forward


def _anpr_fallback_fast(
    code: str,
    ref_codes: set[str],
    cat_bridge: dict[str, str],
    code_to_info: dict[str, tuple[str, str]],
    cat_to_codes: dict[str, set[str]],
    prov_name_to_ref: dict[tuple[str, str], set[str]],
) -> str | None:
    """
    Resolve a code using pre-indexed ANPR data:
    1) Same CODCATASTALE → ref code
    2) Same province + normalized name → ref code
    """
    info = code_to_info.get(code)
    if not info:
        return None

    source_cat, source_name_raw = info
    source_prov = _province_from_codistat(code)

    # Try CODCATASTALE match
    if source_cat:
        for candidate in cat_to_codes.get(source_cat, set()):
            if candidate in ref_codes:
                return candidate
            bridged = cat_bridge.get(candidate)
            if bridged and bridged in ref_codes:
                return bridged

    # Try name match within same province
    source_name = _normalize_name(source_name_raw)
    hits = prov_name_to_ref.get((source_prov, source_name), set())
    if hits:
        return sorted(hits)[0]  # deterministic pick

    return None


# ── Step 3: Detect splits ───────────────────────────────────────────────


def step3_detect_splits(
    forward: dict[str, dict[str, dict]],
) -> dict[str, set[str]]:
    """
    If any census code maps to multiple 1950 codes, all those codes are
    split_children for that year.
    """
    split_children: dict[str, set[str]] = {y: set() for y in CENSUS_DATES}

    for year in CENSUS_DATES:
        for code, info in forward[year].items():
            if len(info["dest"]) > 1:
                split_children[year].update(info["dest"])

        logger.info(f"  Splits in {year}: {len(split_children[year])} affected 1950 codes")

    return split_children


# ── Step 4: Build reverse mapping & classify ────────────────────────────


def step4_classify(
    alive: dict[str, set[str]],
    forward: dict[str, dict[str, dict]],
    split_children: dict[str, set[str]],
    territory_codes: set[str],
) -> pd.DataFrame:
    """
    Invert the forward mapping and classify each 1950 municipality.

    Returns a DataFrame with one row per 1950 CODISTAT.
    """
    ref_codes = sorted(alive["ref"])

    # Pre-build reverse mapping: ref_code → {year: [(src_code, method), ...]}
    reverse: dict[str, dict[str, list[tuple[str, str]]]] = {
        rc: {y: [] for y in CENSUS_DATES} for rc in ref_codes
    }
    for year in CENSUS_DATES:
        for src, info in forward[year].items():
            for dest in info["dest"]:
                if dest in reverse:
                    reverse[dest][year].append((src, info["method"]))

    rows = []
    for ref_code in ref_codes:
        row = {"codistat_1950": ref_code}
        discard_reasons = []
        mapping_type = "DIRECT"

        for year in CENSUS_DATES:
            sources_info = reverse[ref_code][year]
            sources = [s[0] for s in sources_info]
            methods = {s[1] for s in sources_info}

            row[f"mapping_{year}"] = "|".join(sorted(sources)) if sources else ""
            row[f"n_constituents_{year}"] = len(sources)

            # Check for problems
            if not sources and ref_code in alive[year]:
                # Alive at census date but no mapping back — should not happen
                discard_reasons.append(f"unmapped_{year}")

            if "UNTRACED" in methods:
                discard_reasons.append(f"untraced_{year}")

            if "LOST_TERRITORY" in methods:
                discard_reasons.append(f"lost_territory_{year}")

            if ref_code in split_children[year]:
                discard_reasons.append(f"split_child_{year}")

            if len(sources) > 1:
                mapping_type = "AGGREGATE"

        # Territory transfer flag
        row["territory_transfer_flag"] = ref_code in territory_codes

        # Determine status
        discard_reasons = list(set(discard_reasons))  # deduplicate
        if discard_reasons:
            row["status"] = "DISCARD"
            row["discard_reason"] = "; ".join(sorted(discard_reasons))
        else:
            row["status"] = "USABLE"
            row["discard_reason"] = ""

        row["mapping_type"] = mapping_type
        rows.append(row)

    df = pd.DataFrame(rows)
    usable = (df["status"] == "USABLE").sum()
    discard = (df["status"] == "DISCARD").sum()
    logger.info(f"  Classification: {usable} USABLE, {discard} DISCARD out of {len(df)}")
    return df


def build_territory_transfer_set(
    sistat: dict, ref_codes: set[str]
) -> set[str]:
    """
    Identify 1950 municipalities that had partial territory transfers
    (acquisition or cession) between 1921 and 1954.
    """
    territory_codes = set()

    # Get events between 1921 and 1954
    var = sistat["variations"]
    terr_mask = var["event_type"].isin(["territory_acquisition", "territory_cession"])
    date_mask = (var["event_date"] >= "1921-01-01") & (var["event_date"] <= "1954-12-31")
    terr_events = var[terr_mask & date_mask]

    for _, row in terr_events.iterrows():
        sid = row["sistat_id"]
        istat = sistat["sid_to_istat"].get(sid, "")
        if istat in ref_codes:
            territory_codes.add(istat)

    # Also check acquisition/cession counterparts
    for tab_name, col in [
        ("territory_acq", "event_from_istat_cod"),
        ("territory_ces", "event_to_istat_cod"),
    ]:
        tab = sistat[tab_name]
        for _, row in tab.iterrows():
            # Check if the event_num is in our date range
            ev_num = row["event_num"]
            ev_match = terr_events[terr_events["event_num"] == ev_num]
            if not ev_match.empty:
                code_raw = row.get(col)
                if pd.notna(code_raw):
                    code = _pad_istat(code_raw)
                    if code in ref_codes:
                        territory_codes.add(code)

    logger.info(f"  Territory transfers: {len(territory_codes)} municipalities affected")
    return territory_codes


# ── Step 5: Match to reference CSV ──────────────────────────────────────

# Province sigla → full name mapping (built from known ISTAT province codes)
# This covers the pre-1950 province structure
SIGLA_TO_PROVINCE = {
    "TO": "Torino", "VC": "Vercelli", "NO": "Novara", "CN": "Cuneo",
    "AT": "Asti", "AL": "Alessandria", "AO": "Valle d'Aosta",
    "IM": "Imperia", "SV": "Savona", "GE": "Genova", "SP": "La Spezia",
    "VA": "Varese", "CO": "Como", "SO": "Sondrio", "MI": "Milano",
    "BG": "Bergamo", "BS": "Brescia", "PV": "Pavia", "CR": "Cremona",
    "MN": "Mantova", "BZ": "Bolzano", "TN": "Trento",
    "VR": "Verona", "VI": "Vicenza", "BL": "Belluno", "TV": "Treviso",
    "VE": "Venezia", "PD": "Padova", "RO": "Rovigo",
    "UD": "Udine", "GO": "Gorizia", "TS": "Trieste",
    "PC": "Piacenza", "PR": "Parma", "RE": "Reggio nell'Emilia",
    "MO": "Modena", "BO": "Bologna", "FE": "Ferrara",
    "RA": "Ravenna", "FC": "Forli'-Cesena", "FO": "Forli'",
    "RN": "Rimini",
    "MS": "Massa-Carrara", "LU": "Lucca", "PT": "Pistoia",
    "FI": "Firenze", "LI": "Livorno", "PI": "Pisa", "AR": "Arezzo",
    "SI": "Siena", "GR": "Grosseto", "PO": "Prato",
    "PG": "Perugia", "TR": "Terni",
    "VT": "Viterbo", "RI": "Rieti", "RM": "Roma", "LT": "Latina",
    "FR": "Frosinone",
    "AQ": "L'Aquila", "TE": "Teramo", "PE": "Pescara", "CH": "Chieti",
    "CB": "Campobasso", "IS": "Isernia",
    "CE": "Caserta", "BN": "Benevento", "NA": "Napoli",
    "AV": "Avellino", "SA": "Salerno",
    "FG": "Foggia", "BA": "Bari", "TA": "Taranto",
    "BR": "Brindisi", "LE": "Lecce",
    "PZ": "Potenza", "MT": "Matera",
    "CS": "Cosenza", "CZ": "Catanzaro", "RC": "Reggio di Calabria",
    "KR": "Crotone", "VV": "Vibo Valentia",
    "TP": "Trapani", "PA": "Palermo", "ME": "Messina",
    "AG": "Agrigento", "CL": "Caltanissetta", "EN": "Enna",
    "CT": "Catania", "RG": "Ragusa", "SR": "Siracusa",
    "SS": "Sassari", "NU": "Nuoro", "CA": "Cagliari",
    "OR": "Oristano",
    # Historical / special
    "AP": "Ascoli Piceno", "MC": "Macerata", "AN": "Ancona",
    "PU": "Pesaro e Urbino", "PS": "Pesaro e Urbino",
    "LC": "Lecco", "LO": "Lodi", "MB": "Monza e della Brianza",
    "BT": "Barletta-Andria-Trani",
    "FM": "Fermo", "CI": "Carbonia-Iglesias",
    "VS": "Medio Campidano", "OG": "Ogliastra", "OT": "Olbia-Tempio",
    "SU": "Sud Sardegna",
    # Lost territories
    "FU": "Fiume", "PL": "Pola", "ZA": "Zara",
}


# Province name aliases: crosswalk province → covariates province.
# These arise because the ANPR/SIGLA_TO_PROVINCE uses one convention
# while the covariates CSV uses another.
_PROVINCE_ALIASES = {
    "Bolzano": "Bolzano/Bozen",
    "Forli'": "Forli",
    "Taranto": "Jonio",
    "Napoli": "Napoli",  # identity, but needed for the NA→NaN fix
}

# Manual CODISTAT → CSV name overrides for municipalities where the 1950
# name differs from the covariates name (historical renames, I/J swaps,
# suffix changes, province-disambiguated names in CSV).
_MANUAL_CSV_OVERRIDES: dict[str, str] = {
    # Province-disambiguated names (duplicate municipality names)
    "005014": "CALLIANOAT",              # CALLIANO (Asti)
    "016065": "CASTROBG",                # CASTRO (Bergamo)
    "013178": "PEGLIOCO",                # PEGLIO (Como)
    "041041": "PEGLIOPU",                # PEGLIO (Pesaro e Urbino)
    "001235": "SAMONETO",                # SAMONE (Torino)
    "022165": "SAMONETN",                # SAMONE (Trento)
    "022106": "LIVOTN",                  # LIVO (Trento)
    "083090": "SAN TEODOROME",           # SAN TEODORO (Messina)
    # Historical renames
    "057071": "PETESCIA",                # TURANIA (renamed 1928)
    "001300": "VILLAFRANCA SABAUDA",     # VILLAFRANCA PIEMONTE (renamed)
    "079022": "CASINO",                  # CASTELSILANO (renamed)
    "001056": "MASINO",                  # CARAVINO (renamed/merged)
    "092099": "PALMAS SUERGIU",          # SAN GIOVANNI SUERGIU (renamed)
    "092163": "SAN PIETRO PULA",         # VILLA SAN PIETRO (renamed)
    # Spelling / suffix variations
    "079062": "JONADI",                  # IONADI (I/J variation)
    "064071": "PETRURO",                 # PETRURO IRPINO (suffix added later)
    "032004": "SAN DORLIGO DELLA VALLE", # -DOLINA bilingual suffix
    # Merged municipalities (both CW entries → same CSV row)
    "006072": "FRASSINELLO-OLIVOLA",     # FRASSINELLO MONFERRATO
    "006118": "FRASSINELLO-OLIVOLA",     # OLIVOLA
}


def step5_match_reference(
    crosswalk: pd.DataFrame,
    anpr: pd.DataFrame,
    ref_path: Path,
) -> pd.DataFrame:
    """
    Match 1950 CODISTAT codes to the reference CSV rows via name + province.

    Uses a multi-level matching strategy:
      1. Exact normalized (name, province) match
      2. Province alias + apostrophe-stripped match
      3. Manual override table (renames, disambiguated names)
      4. Name-only fallback (when unambiguous across all provinces)
    """
    ref = pd.read_csv(ref_path)
    logger.info(f"  Reference CSV: {len(ref)} rows")

    # Build lookup from ANPR: for 1950-alive codes, get name and province
    alive_1950 = anpr[
        (anpr["DATAISTITUZIONE"] <= REFERENCE_DATE)
        & (anpr["DATACESSAZIONE"] >= REFERENCE_DATE)
    ].drop_duplicates(subset="CODISTAT", keep="last")

    anpr_lookup = {}
    for _, row in alive_1950.iterrows():
        code = row["CODISTAT"]
        anpr_lookup[code] = {
            "name": row["DENOMTRASLITTERATA"],
            "sigla": row["SIGLAPROVINCIA"],
        }

    # Attach ANPR name and province to crosswalk
    names = []
    provinces = []
    for _, row in crosswalk.iterrows():
        code = row["codistat_1950"]
        info = anpr_lookup.get(code, {})
        names.append(info.get("name", ""))
        prov_sigla = info.get("sigla", "")
        provinces.append(SIGLA_TO_PROVINCE.get(prov_sigla, prov_sigla))

    crosswalk["name_1950"] = names
    crosswalk["province_1950"] = provinces

    # ── Build reference lookups ──
    # Level 1: exact normalized (name, province)
    ref_exact: dict[tuple[str, str], int] = {}
    # Level 2: apostrophe-stripped (name, province) with province aliases
    ref_fuzzy: dict[tuple[str, str], int] = {}
    # Level 4: name-only (apostrophe-stripped) → list of indices
    ref_by_name: dict[str, list[int]] = {}
    # CSV name lookup for manual overrides
    ref_by_csv_name: dict[str, int] = {}

    for idx, row in ref.iterrows():
        csv_name = row["comune_matchable_50"]
        csv_prov = row["province"] if pd.notna(row["province"]) else ""

        # Exact
        key_exact = (_normalize_name(csv_name), _normalize_name(csv_prov))
        ref_exact[key_exact] = idx

        # Fuzzy (apostrophe-stripped)
        key_fuzzy = (_normalize_for_matching(csv_name),
                     _normalize_for_matching(csv_prov))
        ref_fuzzy[key_fuzzy] = idx

        # Name-only
        name_key = _normalize_for_matching(csv_name)
        ref_by_name.setdefault(name_key, []).append(idx)

        # By CSV name (for manual overrides)
        ref_by_csv_name[csv_name] = idx

    # ── Match crosswalk → reference (multi-level) ──
    counts = {"exact": 0, "fuzzy": 0, "manual": 0, "name_only": 0, "unmatched": 0}
    csv_names = []

    for _, row in crosswalk.iterrows():
        code = row["codistat_1950"]
        cw_name = row["name_1950"]
        cw_prov = row["province_1950"]

        # Level 1: exact normalized match
        key = (_normalize_name(cw_name), _normalize_name(cw_prov))
        if key in ref_exact:
            csv_names.append(ref.loc[ref_exact[key], "comune_matchable_50"])
            counts["exact"] += 1
            continue

        # Level 2: province alias + apostrophe stripping
        mapped_prov = _PROVINCE_ALIASES.get(cw_prov, cw_prov)
        key_fuzzy = (_normalize_for_matching(cw_name),
                     _normalize_for_matching(mapped_prov))
        if key_fuzzy in ref_fuzzy:
            csv_names.append(ref.loc[ref_fuzzy[key_fuzzy], "comune_matchable_50"])
            counts["fuzzy"] += 1
            continue

        # Level 3: manual override table
        if code in _MANUAL_CSV_OVERRIDES:
            override_name = _MANUAL_CSV_OVERRIDES[code]
            if override_name in ref_by_csv_name:
                csv_names.append(override_name)
                counts["manual"] += 1
                continue

        # Level 4: name-only fallback (apostrophe-stripped, unambiguous)
        name_key = _normalize_for_matching(cw_name)
        candidates = ref_by_name.get(name_key, [])
        if len(candidates) == 1:
            csv_names.append(ref.loc[candidates[0], "comune_matchable_50"])
            counts["name_only"] += 1
            continue

        # No match found
        csv_names.append(None)
        counts["unmatched"] += 1

    crosswalk["csv_name_match"] = csv_names
    total_matched = counts["exact"] + counts["fuzzy"] + counts["manual"] + counts["name_only"]
    logger.info(
        f"  Matched {total_matched}/{len(crosswalk)} to reference CSV "
        f"(exact={counts['exact']}, fuzzy={counts['fuzzy']}, "
        f"manual={counts['manual']}, name_only={counts['name_only']}, "
        f"unmatched={counts['unmatched']})"
    )

    return crosswalk


# ── Step 6: Output ──────────────────────────────────────────────────────

OUTPUT_COLUMNS = [
    "codistat_1950",
    "name_1950",
    "province_1950",
    "csv_name_match",
    "status",
    "discard_reason",
    "mapping_1921",
    "mapping_1927",
    "mapping_1929",
    "mapping_type",
    "n_constituents_1921",
    "n_constituents_1927",
    "n_constituents_1929",
    "territory_transfer_flag",
]


def step6_output(
    crosswalk: pd.DataFrame,
    territory_codes: set[str],
    forward: dict[str, dict[str, dict]],
    output_dir: Path,
) -> None:
    """Write output files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main crosswalk CSV
    out_path = output_dir / "crosswalk_1921_1954.csv"
    crosswalk[OUTPUT_COLUMNS].to_csv(out_path, index=False)
    logger.info(f"  Wrote {out_path} ({len(crosswalk)} rows)")

    # Summary
    summary_path = output_dir / "crosswalk_summary.txt"
    with open(summary_path, "w") as f:
        total = len(crosswalk)
        usable = (crosswalk["status"] == "USABLE").sum()
        discard = (crosswalk["status"] == "DISCARD").sum()

        f.write("Municipality Crosswalk Summary (1921–1954)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total 1950 municipalities: {total}\n")
        f.write(f"USABLE:  {usable} ({usable/total*100:.1f}%)\n")
        f.write(f"DISCARD: {discard} ({discard/total*100:.1f}%)\n\n")

        # Discard reasons breakdown
        f.write("Discard reasons:\n")
        discard_df = crosswalk[crosswalk["status"] == "DISCARD"]
        reason_counts = defaultdict(int)
        for reasons in discard_df["discard_reason"]:
            for r in reasons.split("; "):
                if r:
                    # Group by base reason
                    base = r.rsplit("_", 1)[0] if r.count("_") > 1 else r
                    reason_counts[base] += 1
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {reason}: {count}\n")

        f.write("\nMapping types:\n")
        for mt, count in crosswalk["mapping_type"].value_counts().items():
            f.write(f"  {mt}: {count}\n")

        # Direct across all years
        direct_all = crosswalk[
            (crosswalk["mapping_type"] == "DIRECT")
            & (crosswalk["status"] == "USABLE")
        ]
        f.write(f"\nDIRECT & USABLE: {len(direct_all)}\n")

        # CSV match rate
        matched = crosswalk["csv_name_match"].notna().sum()
        unmatched = total - matched
        f.write(f"\nCSV match rate: {matched}/{total} ({matched/total*100:.1f}%)\n")
        f.write(f"Unmatched to reference CSV: {unmatched}\n")

        # Territory transfers
        terr = crosswalk["territory_transfer_flag"].sum()
        f.write(f"Territory transfer flags: {terr}\n")

        # Forward mapping method breakdown
        f.write("\nForward mapping methods (census_code → 1950_code):\n")
        for year in sorted(CENSUS_DATES.keys()):
            if year in forward:
                method_counts = defaultdict(int)
                for info in forward[year].values():
                    method_counts[info["method"]] += 1
                f.write(f"  {year}: ")
                f.write(", ".join(
                    f"{m}={c}" for m, c in sorted(method_counts.items())
                ))
                f.write(f" (total {sum(method_counts.values())})\n")

    logger.info(f"  Wrote {summary_path}")

    # Territory transfers CSV
    terr_df = crosswalk[crosswalk["territory_transfer_flag"]][
        ["codistat_1950", "name_1950", "province_1950", "status"]
    ]
    terr_path = output_dir / "territory_transfers.csv"
    terr_df.to_csv(terr_path, index=False)
    logger.info(f"  Wrote {terr_path} ({len(terr_df)} rows)")


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=== Municipality Crosswalk Builder (1921–1954) ===")

    # Load data
    anpr = load_anpr(PROJECT_ROOT / "ANPR_archivio_comuni.csv")
    sistat = load_sistat(PROJECT_ROOT / "sistat_data")

    # Step 1
    logger.info("Step 1: Building alive sets …")
    alive = step1_alive_sets(anpr)

    # Step 2
    logger.info("Step 2: Building forward mappings …")
    forward = step2_forward_mapping(anpr, sistat, alive)

    # Step 3
    logger.info("Step 3: Detecting splits …")
    split_children = step3_detect_splits(forward)

    # Territory transfers
    logger.info("Building territory transfer set …")
    territory_codes = build_territory_transfer_set(sistat, alive["ref"])

    # Step 4
    logger.info("Step 4: Classifying municipalities …")
    crosswalk = step4_classify(alive, forward, split_children, territory_codes)

    # Step 5
    logger.info("Step 5: Matching to reference CSV …")
    crosswalk = step5_match_reference(
        crosswalk, anpr, PROJECT_ROOT / "pre-war-covariates-complete_updated.csv"
    )

    # Step 6
    logger.info("Step 6: Writing output …")
    step6_output(crosswalk, territory_codes, forward, PROJECT_ROOT / "output")

    # Verification
    logger.info("=== Verification ===")
    total = len(crosswalk)
    usable = (crosswalk["status"] == "USABLE").sum()
    discard = (crosswalk["status"] == "DISCARD").sum()
    logger.info(f"  Total rows: {total} (expected {EXPECTED_ALIVE['ref']})")
    logger.info(f"  USABLE + DISCARD = {usable + discard} (should be {total})")
    untraced_usable = crosswalk[
        (crosswalk["status"] == "USABLE")
        & (crosswalk["discard_reason"].str.contains("untraced", na=False))
    ]
    logger.info(f"  USABLE with UNTRACED: {len(untraced_usable)} (should be 0)")
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
