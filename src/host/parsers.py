"""
Parsers & heuristics for MCP_AutoHost.

Centralizes all parsing based on regex (cars, trainer, Pokémon, utilities).
Leaves the orchestration/CLI modules to handle the business flow, while here
tasks related to extraction from natural language are handled.

Main functions (grouped):
- Cars:
    - parse_auto_from_text
    - parse_mileage_max_from_text
    - parse_budget_from_text_strict, parse_budget_from_text
    - parse_year_range_from_text, parse_year_min_from_text, parse_year_max_from_text
    - parse_count_from_text
- Trainer:
    - parse_trainer_metrics_from_text
    - parse_trainer_generic_from_text
    - parse_imperial_metrics
- Pokémon VGC:
    - parse_poke_constraints_from_text
- Common helpers:
    - _norm_key, _try_float, _parse_gender, _parse_int, _parse_float

Format notes/assumptions:
- The car dataset uses **kilometers** for "Mileage". Functions that
  return "Mileage_max" ALWAYS return kilometers.
- "budget" / "Price_max" return integers in USD.
- Year range returns (ymin, ymax) with None when not applicable.
- The functions are tolerant of EN/ES.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple, Dict, Any, List

# ── Common constants ───────────────────────────────────────────────────────────

_DASH = r"[–-]"            # en dash or hyphen
_YEAR = r"(?:19|20)\d{2}"  # full year

__all__ = [
    # Cars
    "parse_auto_from_text",
    "parse_mileage_max_from_text",
    "parse_budget_from_text_strict",
    "parse_budget_from_text",
    "parse_year_range_from_text",
    "parse_year_min_from_text",
    "parse_year_max_from_text",
    "parse_count_from_text",
    # Trainer
    "parse_trainer_metrics_from_text",
    "parse_trainer_generic_from_text",
    "parse_imperial_metrics",
    # Pokémon
    "parse_poke_constraints_from_text",
    # Helpers
    "_norm_key",
    "_try_float",
    "_parse_gender",
    "_parse_int",
    "_parse_float",
]

# ── Common helpers ──────────────────────────────────────────────────────────────

def _norm_key(k: str) -> str:
    """Normalizes key names for free→schema mapping."""
    return k.strip().lower().replace("_", " ")

def _try_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _parse_gender(text: str) -> Optional[str]:
    t = text.lower()
    if re.search(r"\b(male|man|masculino|hombre)\b", t):
        return "male"
    if re.search(r"\b(female|woman|femenino|mujer)\b", t):
        return "female"
    return None

def _parse_int(text: str, pat: str) -> Optional[int]:
    m = re.search(pat, text, re.I)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _parse_float(text: str, pat: str) -> Optional[float]:
    m = re.search(pat, text, re.I)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


# ── Cars: budget, mileage, years, general hints ──────────────────────────────

def parse_budget_from_text_strict(text: str) -> Optional[int]:
    """
    Extracts an explicit budget with reference to USD/$ to avoid collision with “mileage”.
    Accepts “$12,000”, “usd 12k”, “under 15k dollars”, “≤ 15k usd”.

    Returns:
        int | None  (in USD)

    Examples:
        >>> parse_budget_from_text_strict("under 15k dollars")
        15000
        >>> parse_budget_from_text_strict("$12,500")
        12500
    """
    if not text:
        return None
    t = text.lower()

    m = re.search(r"(\$|usd)\s*([\d,\.]+k?|\d+)", t)
    if m:
        num = m.group(2).replace(",", "")
        if num.endswith("k"):
            return int(float(num[:-1]) * 1000)
        return int(float(num))

    m = re.search(r"(under|<=|≤|less than)\s*([\d,\.]+k?)\s*(dollars|usd|\$)", t)
    if m:
        num = m.group(2).replace(",", "")
        return int(float(num[:-1]) * 1000) if num.endswith("k") else int(float(num))

    return None


def parse_budget_from_text(text: str) -> Optional[float]:
    """
    Extracts an amount like “$12,000”, “12000”, “10k”, etc. (Does not require USD explicitly).
    Useful for general detections, but can confuse “mileage” with money if
    there's no context.

    Returns:
        float | None  (in USD)

    Examples:
        >>> parse_budget_from_text("$12,000")
        12000.0
        >>> parse_budget_from_text("budget 8k")
        8000.0
    """
    if not text:
        return None
    t = text.lower()
    m = re.search(r'(\$?\s*\d{1,3}(?:[,\s]\d{3})+|\$?\s*\d{3,6}|\d{1,3}\s*k)\b', t)
    if not m:
        return None
    raw = m.group(1).lower().replace("$", "").replace(",", "").strip()
    if raw.endswith("k"):
        val = _try_float(raw[:-1])
        return val * 1000 if val is not None else None
    return _try_float(raw)


def parse_count_from_text(text: str):
    t = (text or "").lower()
    m = re.search(r'\b(?:top|best|show|list|up to|give me|dame|muéstrame|muestrame)\s*(\d{1,2})\b', t)
    if m: return int(m.group(1))
    m = re.search(r'\b(\d{1,2})\s*(?:results|cars|options|vehiculos?|recommendations|recs|exercises|ejercicios|items|candidates|pok[eé]mon)\b', t)
    if m: return int(m.group(1))
    return None



def parse_year_range_from_text(text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Robust detection of year range or open limits.

    Patterns:
        - “2021–2022”, “2016-2020”
        - “from 2016 to 2020”, “between 2016 and 2020”
        - “2020+”, “since 2020”
        - “<= 2020”, “up to 2020”

    Returns:
        (ymin, ymax) with None if missing.

    Examples:
        >>> parse_year_range_from_text("between 2017 and 2019")
        (2017, 2019)
        >>> parse_year_range_from_text("2018+")
        (2018, None)
        >>> parse_year_range_from_text("<= 2020")
        (None, 2020)
    """
    if not text:
        return (None, None)
    t = text.lower()

    # “2021–2022” | “2016-2020”
    m = re.search(rf"\b{_YEAR}\s*{_DASH}\s*{_YEAR}\b", t)
    if m:
        ys = re.findall(_YEAR, m.group(0))
        if len(ys) >= 2:
            y1, y2 = int(ys[0]), int(ys[1])
            return (min(y1, y2), max(y1, y2))

    # “from 2016 to 2020” | “between 2016 and 2020”
    m = re.search(rf"(from|between)\s*{_YEAR}\s*(to|and|through)\s*{_YEAR}", t)
    if m:
        ys = re.findall(_YEAR, m.group(0))
        if len(ys) >= 2:
            y1, y2 = int(ys[0]), int(ys[1])
            return (min(y1, y2), max(y1, y2))

    # “2020+” / “since 2020”
    m = re.search(rf"\b{_YEAR}\s*\+", t) or re.search(rf"(since|after)\s*{_YEAR}", t)
    if m:
        y = re.search(_YEAR, m.group(0))
        if y:
            return (int(y.group(0)), None)

    # “<= 2020” / “up to 2020” / “until 2020”
    m = re.search(rf"(<=|≤|till|until|up to|through)\s*{_YEAR}", t)
    if m:
        y = re.search(_YEAR, m.group(0))
        if y:
            return (None, int(y.group(0)))

    return (None, None)


def parse_year_min_from_text(text: str) -> Optional[int]:
    """
    Extracts minimum year (“from 2018”, “since 2018”, “2018+”, “>= 2018”).
    """
    if not text:
        return None
    t = text.lower().strip()

    m = re.search(r"(from|since|desde|de)\s+(\d{4})", t)
    if m:
        try:
            return int(m.group(2))
        except Exception:
            pass

    m = re.search(r"\b(\d{4})\s+(and\s+newer|plus|\+)\b", t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass

    m = re.search(r">=\s*(\d{4})", t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass

    m = re.search(r"\b(\d{4})\s*\+\b", t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass

    # redundant but tolerant
    m = re.search(r'\b((19|20)\d{2})\s*\+', t)
    if m:
        return int(m.group(1))

    return None


def parse_year_max_from_text(text: str) -> Optional[int]:
    """
    Extracts maximum year (“<= 2018”, “up to 2018”, “until 2018”, “2018 or earlier”).
    """
    if not text:
        return None
    t = text.lower().strip()

    m = re.search(r"(<=|≤|up to|until|through)\s*(\d{4})", t)
    if m:
        try:
            return int(m.group(2))
        except Exception:
            pass

    m = re.search(r"\b(\d{4})\s*(or earlier|and older)\b", t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return None


def parse_mileage_max_from_text(text: str) -> Optional[int]:
    """
    Extracts a maximum mileage limit from text.

    Accepts:
        - “≤ 80,000 km”, “< 60k km”, “under 60k miles”, “less than 60k mi”
        - “mileage ... under 60k mi”
        - “60k miles max”, “60000 km or less”
        - “low mileage” → heuristic 60,000 km

    ALWAYS RETURNS **KILOMETERS** (int), as the dataset is in km.

    Examples:
        >>> parse_mileage_max_from_text("under 60k miles")
        96561
        >>> parse_mileage_max_from_text("≤ 80,000 km")
        80000
    """
    if not text:
        return None
    t = text.lower()

    unit_pat = r"(km|kilometer|kilometre|kilometers|kilometres|mile|miles|mi)"
    key_pat  = r"(mileage|odometer|odo)"
    num_pat  = r"(\d[\d,\.]*k?)"

    patterns = [
        rf"(≤|<=|<|under|less than)\s*{num_pat}\s*{unit_pat}",
        rf"{key_pat}.*?(≤|<=|<|under|less than)\s*{num_pat}\s*{unit_pat}",
        rf"{num_pat}\s*{unit_pat}\s*(max|or less)"
    ]
    for p in patterns:
        m = re.search(p, t)
        if m:
            nums = re.findall(num_pat, m.group(0))
            unit = re.search(unit_pat, m.group(0))
            if nums and unit:
                raw = nums[0].replace(",", "").strip()
                val = int(float(raw[:-1]) * 1000) if raw.endswith("k") else int(float(raw))
                u = unit.group(0)
                # dataset in km → convert if unit is in miles
                if u in ("mile", "miles", "mi"):
                    # 1 mi = 1.60934 km
                    val = int(round(val * 1.60934))
                return val

    if "low mileage" in t:
        return 60000

    return None

def parse_auto_from_text(text: str) -> dict:
    t = (text or "").lower()
    out = {}

    # Fuel
    if "diesel" in t or "diésel" in t: out["Fuel Type"] = "Diesel"
    elif "hybrid" in t: out["Fuel Type"] = "Hybrid"
    elif "electric" in t or "ev" in t: out["Fuel Type"] = "Electric"
    elif any(w in t for w in ["gasoline", "gas", "petrol", "nafta"]): out["Fuel Type"] = "Gasoline"

    # Transmission
    if any(w in t for w in ["automatic", "auto"]): out["Transmission"] = "Automatic"
    elif any(w in t for w in ["manual", "stick"]): out["Transmission"] = "Manual"

    # Condition
    if "like new" in t: out["Condition"] = "Like New"
    elif "new" in t: out["Condition"] = "New"
    elif "used" in t: out["Condition"] = "Used"

    # Safety / Accident-free
    if any(w in t for w in ["accident-free", "no accidents", "sin accidentes", "safest", "safety"]):
        out["Accident"] = "No"

    # Body style (keep as internal hint)
    if any(w in t for w in ["suv", "crossover"]): out["__BODY_STYLE__"] = "SUV"
    elif any(w in t for w in ["truck", "pickup", "pick-up"]): out["__BODY_STYLE__"] = "Truck"
    elif "sedan" in t: out["__BODY_STYLE__"] = "Sedan"
    elif "hatchback" in t: out["__BODY_STYLE__"] = "Hatchback"
    elif "wagon" in t: out["__BODY_STYLE__"] = "Wagon"
    elif "van" in t: out["__BODY_STYLE__"] = "Van"

    # ---- Mileage_max (REQUIRES unit or 'mileage/odometer/odo' word)
    unit_pat = r"(?:miles|mi|kms?|km)"
    num_pat  = r"([0-9]{1,3}(?:[,\.\s]\d{3})*|\d+)\s*(k)?"

    m = (re.search(rf"(?:mileage|odometer|odo)\s*(?:<=|<|under|less than)?\s*{num_pat}\s*{unit_pat}", t)
         or re.search(rf"(?:<=|<|under|less than)\s*{num_pat}\s*{unit_pat}", t))
    if m:
        raw = m.group(1).replace(",", "").replace(".", "").replace(" ", "")
        val = int(raw) * (1000 if m.group(2) else 1)
        unit = re.search(unit_pat, t[m.start():m.end()]).group(0)
        if unit in ("miles", "mi"):
            val = int(round(val * 1.60934))  # dataset in km
        out["Mileage_max"] = val

    # ---- Price_max: requires currency or 'budget' word
    m = (re.search(rf"(?:<=|<|under|less than)\s*(?:\$|\busd\b|\bdollars?\b)\s*{num_pat}\b", t)
         or re.search(rf"\bbudget\b\s*(?:is|=|:)?\s*(?:\$|\busd\b|\bdollars?\b)?\s*{num_pat}\b", t))
    if m:
        raw = m.group(1).replace(",", "").replace(".", "").replace(" ", "")
        out["__PRICE_MAX__"] = int(raw) * (1000 if m.group(2) else 1)

    return out



# ── Trainer: metrics and generics (goal, sport, etc.) ─────────────────────

def parse_imperial_metrics(text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Converts imperial metrics to SI units:
        - Height in 5'9" or "5 ft 9 in" → centimeters
        - Weight in lb/lbs/pounds → kilograms

    Returns:
        (height_cm, weight_kg)  — each can be None
    """
    if not text:
        return (None, None)

    t = text.lower()
    height_cm = weight_kg = None

    # Height 5'9" or 5 ft 9 in
    m = re.search(r'(\d)\'\s*(\d{1,2})\"', t) or re.search(r'(\d)\s*ft\s*(\d{1,2})\s*in', t)
    if m:
        ft, inch = int(m.group(1)), int(m.group(2))
        height_cm = round(ft * 30.48 + inch * 2.54, 1)

    # Weight in lb
    m = re.search(r'\b(\d{2,3})\s*(lb|lbs|pounds?)\b', t)
    if m:
        weight_kg = round(int(m.group(1)) * 0.453592, 1)

    return height_cm, weight_kg


def parse_trainer_metrics_from_text(text: str) -> Dict[str, Any]:
    """
    Extracts {gender, age, height_cm, weight_kg} from free message (EN/ES).

    Accepts:
        - “male/female/masculino/femenino”
        - “age 28” / “28 years old” / “edad 28” / “28 años”
        - “175 cm” / “altura 175 cm” / “height 175 cm”
        - “78 kg” / “peso 78 kg” / “weight 78 kg”
        - Imperial metrics: “5'9"”, “5 ft 9 in”, “172 lb”

    Returns:
        dict with any of the detected keys.

    Example:
        >>> parse_trainer_metrics_from_text("male, 5'9\", 172 lb, 28 yo")["gender"]
        'male'
    """
    out: Dict[str, Any] = {}
    if not text:
        return out

    t = text.lower()

    gender = _parse_gender(t)
    if gender:
        out["gender"] = gender

    # age
    age = (
        _parse_int(t, r"\bage\s*[:\-]?\s*(\d{1,3})\b")
        or _parse_int(t, r"\b(\d{1,3})\s*(years|year|yo|años|año)\b")
        or _parse_int(t, r"\bedad\s*[:\-]?\s*(\d{1,3})\b")
    )
    if age is not None:
        out["age"] = age

    # height (cm)
    height_cm = (
        _parse_float(t, r"\bheight\s*[:\-]?\s*(\d{2,3})(?:\.\d+)?\s*cm\b")
        or _parse_float(t, r"\baltura\s*[:\-]?\s*(\d{2,3})(?:\.\d+)?\s*cm\b")
        or _parse_float(t, r"\b(\d{2,3})(?:\.\d+)?\s*cm\b")
    )

    # weight (kg)
    weight_kg = (
        _parse_float(t, r"\bweight\s*[:\-]?\s*(\d{2,3})(?:\.\d+)?\s*kg\b")
        or _parse_float(t, r"\bpeso\s*[:\-]?\s*(\d{2,3})(?:\.\d+)?\s*kg\b")
        or _parse_float(t, r"\b(\d{2,3})(?:\.\d+)?\s*kg\b")
    )

    # if no SI metrics, try imperial
    imp_h, imp_w = parse_imperial_metrics(text)
    if height_cm is None and imp_h is not None:
        height_cm = imp_h
    if weight_kg is None and imp_w is not None:
        weight_kg = imp_w

    if height_cm is not None:
        out["height_cm"] = height_cm
    if weight_kg is not None:
        out["weight_kg"] = weight_kg

    return out


def parse_trainer_generic_from_text(text: str) -> dict:
    t = (text or "").lower()
    out: dict = {}

    # goal
    if re.search(r'\b(fat\s*loss|lose\s*fat|weight\s*loss|burn\s*fat|perder\s*grasa|bajar\s*grasa)\b', t):
        out['goal'] = 'fat loss'
    elif (re.search(r'\b(gain(?:ing)?|build(?:ing)?|put on|add)\s+(muscle|masa muscular|size)\b', t) or
          re.search(r'\b(hypertrophy|hipertrofia|bulk(?:\s*up)?)\b', t) or
          re.search(r'\b(ganar|aumentar)\s+(m[úu]sculo|masa muscular)\b', t)):
        out['goal'] = 'gain muscle mass'
    elif re.search(r'\b(endurance|stamina|resistencia)\b', t):
        out['goal'] = 'endurance'
    elif re.search(r'\b(strength|fuerza)\b', t):
        out['goal'] = 'strength'

    # sport (same as before)
    sport_map = [
        (r'\brunn?ing|correr\b', 'running'),
        (r'\bcalisthenics|calistenia|bodyweight\b', 'calisthenics'),
        (r'\bcycling|bike|bici|ciclismo\b', 'cycling'),
        (r'\bpowerlifting\b', 'powerlifting'),
        (r'\bvolleyball|voleibol\b', 'volleyball'),
        (r'\bbox(ing)?|boxeo\b', 'boxing'),
        (r'\bswim(ming)?|nataci[óo]n\b', 'swimming'),
        (r'\bfootball|soccer|f[úu]tbol\b', 'soccer'),
    ]
    for pat, val in sport_map:
        if re.search(pat, t):
            out['sport'] = val
            break

    # days/week
    m = re.search(r'(\d+)\s*(days?|d[ií]as?)\s*(per\s*week|\/\s*week|a\s*la\s*semana|por\s*semana)?', t)
    if m:
        out['days_per_week'] = int(m.group(1))

    # minutes per session
    m = re.search(r'(\d+)\s*(min(?:s|utes)?|minutos)\s*(per\s*session|por\s*ses[ií]on)?', t)
    if m:
        out['minutes_per_session'] = int(m.group(1))

    # experience
    if re.search(r'\b(beginner|novice|principiante|novato)\b', t):
        out['experience'] = 'beginner'
    elif re.search(r'\b(intermediate|intermedio)\b', t):
        out['experience'] = 'intermediate'
    elif re.search(r'\b(advanced|avanzad[oa])\b', t):
        out['experience'] = 'advanced'

    # limit (uses synonyms like “recommendations”)
    m = re.search(r'\b(limit|l[ií]mite|max(?:imo)?)\s*[:=]?\s*(\d{1,2})\b', t)
    if m:
        out['limit'] = int(m.group(2))
    else:
        cnt = parse_count_from_text(text)
        if cnt:
            out['limit'] = cnt

    # sessions/week (synonyms)
    m = re.search(r'(\d{1,2})\s*(sessions?|workouts?|entrenamientos?)\s*(per\s*week|a\s*la\s*semana|por\s*semana)?', t)
    if m:
        out['days_per_week'] = int(m.group(1))

    return out


# ── Pokémon VGC: constraints from natural language ─────────────────────────────
import re

_POKE_TYPES = {
    "normal","fire","water","electric","grass","ice","fighting","poison","ground",
    "flying","psychic","bug","rock","ghost","dragon","dark","steel","fairy"
}

# minimal synonyms/spanish aliases
_TYPE_ALIASES = {
    "fuego":"fire", "agua":"water", "eléctrico":"electric", "electrico":"electric",
    "planta":"grass", "hielo":"ice", "lucha":"fighting", "veneno":"poison",
    "tierra":"ground", "volador":"flying", "psíquico":"psychic", "psiquico":"psychic",
    "bicho":"bug", "roca":"rock", "fantasma":"ghost", "dragón":"dragon", "dragon":"dragon",
    "siniestro":"dark", "acero":"steel", "hada":"fairy", "normal":"normal"
}

# abilities we want to capture (expand if needed)
_ABILITY_ALIASES = {
    "levitate":"Levitate",
    "levitar":"Levitate",
    # add if needed: "intimidate":"Intimidate", ...
}

def _find_types(text_lower: str):
    found = set()
    # “fire type”, “tipo fuego”, “type: fire”
    for m in re.finditer(r'(?:type\s*[:=]?\s*|tipo\s+)([a-záéíóúüñ\-]+)', text_lower):
        tok = m.group(1).strip()
        tok = _TYPE_ALIASES.get(tok, tok)
        if tok in _POKE_TYPES:
            found.add(tok)
    # simple list: “fire type pokemon”, “show fire and water …”
    #    Detects any token of types present
    tokens = re.findall(r'[a-záéíóúüñ\-]+', text_lower)
    for tok in tokens:
        base = _TYPE_ALIASES.get(tok, tok)
        if base in _POKE_TYPES:
            # Avoid obvious false positives (“steel” sometimes appears alone, but it's acceptable)
            found.add(base)
    return sorted(found)

def _find_required_abilities(text_lower: str):
    req = []
    for key, canon in _ABILITY_ALIASES.items():
        if re.search(r'\b' + re.escape(key) + r'\b', text_lower):
            req.append(canon)
    # patterns “with Levitate”, “con Levitate”
    m = re.search(r'\b(?:with|con)\s+([A-Za-z][A-Za-z \-]+)\b', text_lower)
    if m:
        cand = m.group(1).strip().lower()
        if cand in _ABILITY_ALIASES:
            req.append(_ABILITY_ALIASES[cand])
    return sorted(set(req))

def _find_min_speed(text_lower: str):
    # >=, ≥, “at least”, “or more”, “100+”, “spe 100 or more”
    patterns = [
        r'\bmin(?:imum)?\s*speed\s*[:=]?\s*(\d{2,3})\b',
        r'\b(?:speed|spe)\s*(?:>=|≥)\s*(\d{2,3})\b',
        r'\b(?:speed|spe)\s*(\d{2,3})\s*(?:\+|or\s+more|or\s+higher)\b',
        r'\b(?:at\s*least)\s*(\d{2,3})\s*(?:speed|spe)?\b',
        r'\b(\d{2,3})\s*(?:speed|spe)\s*(?:or\s+more|or\s+higher|\+)\b',
    ]
    for pat in patterns:
        m = re.search(pat, text_lower)
        if m:
            return int(m.group(1))
    # “faster than 100”, “over 100”
    m = (re.search(r'\bfaster\s+than\s*(\d{2,3})\b', text_lower) or
         re.search(r'\b(?:speed|spe)\s*(?:>|over|above)\s*(\d{2,3})\b', text_lower))
    if m:
        return int(m.group(1)) + 1
    return None

def parse_poke_constraints_from_text(text: str) -> dict:
    t = (text or "").lower()
    out = {}

    types = _find_types(t)
    if types:
        out.setdefault("include_types", []).extend(types)

    min_spe = _find_min_speed(t)
    if min_spe is not None:
        out["min_speed"] = min_spe

    req_abilities = _find_required_abilities(t)
    if req_abilities:
        out["require_abilities"] = req_abilities

    return out


def parse_remote_echo_command(user_msg: str):
    if "hello" in user_msg.lower():
        return {"params": {"text": "hello?"}}
    return {}

def parse_sum_command(user_msg: str):
    numbers = re.findall(r'\d+', user_msg)
    if len(numbers) == 2:
        return {"params": {"a": int(numbers[0]), "b": int(numbers[1])}}
    return {}
