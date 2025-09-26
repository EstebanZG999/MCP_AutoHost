import os, sys, json, asyncio
import re
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from dotenv import load_dotenv

from .memory import Memory
from .llm_client import LLMClient
from .logging_utils import tail_logs
from .server_manager import ServerManager
from .nl_router import NaturalLanguageOrchestrator

import yaml

# 👇 All parsers centralized in src/host/parsers.py
from .parsers import (
    # cars
    parse_year_range_from_text,
    parse_mileage_max_from_text,
    parse_budget_from_text_strict,
    parse_auto_from_text,
    parse_year_max_from_text, 
    parse_year_min_from_text, 
    # trainer
    parse_trainer_metrics_from_text,
    parse_trainer_generic_from_text,
    # pokémon
    parse_poke_constraints_from_text,
    # helpers
    _norm_key,
)

CONFIG_FILE = os.getenv("SERVERS_YAML", "configs/servers.yaml")
LOG_FILE = os.getenv("SESSION_LOG", "logs/session.jsonl")

# Load .env explicitly from CWD
load_dotenv(".env")
console = Console()

def print_help():
    tbl = Table(title="Available commands")
    tbl.add_column("Command")
    tbl.add_column("Description")
    tbl.add_row("/help", "Show this help")
    tbl.add_row("/reset", "Clear memory/context")
    tbl.add_row("/logs [N]", "Show last N log entries (default 50)")
    tbl.add_row("tools", "Show catalog of tools (server.tool + schema)")
    tbl.add_row("context", "Show short conversation history")
    tbl.add_row("/exit", "Exit")
    console.print(tbl)

def _safe_panel_text(x: str) -> str:
    """Ensures that what we send to the Panel is str and not None."""
    return x if isinstance(x, str) else str(x)

# 1) Conversion and miles/km columns
def _rows_with_units(rows, user_msg: str):
    import re
    t = (user_msg or "").lower()
    # use word boundaries to avoid confusing "mi" with "mini"
    wants_miles = bool(re.search(r"\bmi\b|\bmiles?\b", t))
    out = []
    for r in rows:
        r2 = dict(r)
        if "Mileage" in r2 and isinstance(r2["Mileage"], (int, float)):
            km = r2.pop("Mileage")
            r2["Mileage (km)"] = km
            if wants_miles:
                r2["Mileage (mi)"] = int(round(km / 1.60934))
        out.append(r2)
    return out

# 2) Table construction: calls _rows_with_units and sets column order
def _build_preview_table_from_json(obj, user_msg: str = "", max_rows: int = 3) -> Table | None:
    """
    If the MCP output has a list of results (e.g., 'recommendations' or 'results'),
    it builds a nice table with up to max_rows rows.
    """
    if not isinstance(obj, dict):
        return None

    rows = None
    if isinstance(obj.get("recommendations"), list):
        rows = obj["recommendations"]
    elif isinstance(obj.get("results"), list):
        rows = obj["results"]

    if not rows or not isinstance(rows, list):
        return None

    # <-- HERE integrate units
    rows = _rows_with_units(rows, user_msg)

    # Preferred order for cars; anything not here goes at the end
    preferred = [
        "Car Make","Car Model","Year","Mileage (km)","Mileage (mi)",
        "Price","Fuel Type","Transmission","Condition","Accident","Color"
    ]

    # Discover present columns while respecting the preferred order
    cols = [c for c in preferred if any(isinstance(r, dict) and c in r for r in rows)]
    # Add any other columns that appear
    for item in rows[:max_rows]:
        if isinstance(item, dict):
            for k in item.keys():
                if k not in cols:
                    cols.append(k)

    if not cols:
        return None

    tbl = Table(title=f"Top {min(len(rows), max_rows)} results", box=box.SIMPLE_HEAVY)
    for c in cols:
        tbl.add_column(str(c))

    # Light formatting to avoid showing ugly floats in Price
    def _fmt(c, v):
        if c == "Price" and isinstance(v, (int, float)):
            return f"{v:,.2f}"
        return str(v)

    for item in rows[:max_rows]:
        if isinstance(item, dict):
            tbl.add_row(*[_fmt(c, item.get(c, "")) for c in cols])
        else:
            tbl.add_row(str(item))

    return tbl


def _should_apply_trainer_packing(server_name: str, tool_name: str, schema: dict) -> bool:
    """
    Heuristic: apply force_trainer_params if the tool seems to be a training tool
    based on the server/tool name or because the schema has 'params' with trainer-like signals.
    """
    srv = (server_name or "").lower()
    tl  = (tool_name or "").lower()
    if "train" in srv or tl in {
        "compute_metrics",
        "recommend_exercises",
        "build_routine_tool",
        "recommend_by_metrics_tool",
    }:
        return True

    props = (schema or {}).get("properties") or {}
    if "params" in props and isinstance(props["params"], dict):
        pprops = props["params"].get("properties")
        if isinstance(pprops, dict):
            trainerish = {"gender","age","height_cm","weight_kg","goal","sport","limit","days_per_week","minutes_per_session","experience"}
            if any(k in pprops for k in trainerish):
                return True

    req = schema.get("required") or []
    if "params" in req:
        return True

    return False

# ────────────────────────────────────────────────────────────────────────────────
# Argument normalizer (not a parser — uses parsers to infer values)
# ────────────────────────────────────────────────────────────────────────────────

def force_trainer_params(user_msg: str, args: dict, schema: dict) -> dict:
    """
    Adaptive for training tools:
    - If the schema declares 'params', it packs them into {"params": {...}}
    - If it does NOT declare 'params', it places the fields at the ROOT level
    - Infers metrics and generics from text (ES/EN)
    - Normalizes limit/limite according to what the schema declares
    - Filters only by params_schema if it is NOT empty
    """
    args = (args or {}).copy()
    props = (schema or {}).get("properties") or {}
    has_params_prop = "params" in props and isinstance(props["params"], dict)
    params_schema = props["params"].get("properties") if has_params_prop else None

    # Infers from text
    metrics = parse_trainer_metrics_from_text(user_msg)   # gender, age, height_cm, weight_kg
    generic = parse_trainer_generic_from_text(user_msg)   # goal, sport, limit, days_..., minutes_..., experience

    # Helper to normalize integers
    def _to_int(v):
        try: return int(v)
        except: return None

    if has_params_prop:
        # ----- WITH PARAMS ----- 
        params = dict(args.get("params") or {})

        # Migrate loose ones → params (includes limit/limite if they came loose)
        for k in ("gender", "age", "height_cm", "weight_kg",
                  "goal", "sport", "days_per_week", "minutes_per_session", "experience",
                  "limit", "limit"):
            if k in args and k not in params:
                params[k] = args.pop(k)

        # Fill in missing
        for k, v in {**metrics, **generic}.items():
            params.setdefault(k, v)

        # Normalize limit→limite inside params if applicable
        if "limit" not in params and "limite" in params:
            v = _to_int(params.pop("limite"))
            if v is not None:
                params["limit"] = v

        # Filter only if params_schema is a non-empty dict
        if isinstance(params_schema, dict) and len(params_schema) > 0:
            params = {k: v for k, v in params.items() if k in params_schema}

        args["params"] = params
        return args

    else:
        # ----- WITHOUT PARAMS (root-level) ----- 
        out = args.copy()

        # Fill root with inferences
        for k, v in {**metrics, **generic}.items():
            out.setdefault(k, v)

        # Normalize limit→limite if the schema declares 'limit' (English) and does NOT declare 'limite'
        if "limit" in props and "limite" not in props and "limit" not in out and "limite" in out:
            v = _to_int(out.pop("limite"))
            if v is not None:
                out["limit"] = v

        # If the schema declares 'limite' but not 'limit', do the reverse conversion
        if "limite" in props and "limit" in out and "limite" not in out:
            v = _to_int(out.pop("limit"))
            if v is not None:
                out["limite"] = v

        return out


def _needs_auto_filters(user_msg: str) -> bool:
    t = (user_msg or "").lower()
    return any(kw in t for kw in [
        "$", "usd", "under", "less than", "≤", "<=",
        "year", "years", "since", "+", "accident", "accident-free",
        "mileage", "km", "kms", "miles", "mi"
    ])

def _force_auto_filter_cars_if_needed(user_msg: str, tool_ref: str, args: dict) -> tuple[str, dict]:
    """
    If the LLM chose top_cars but there are filters → change it to filter_cars.
    Keep useful args (e.g., Transmission) and add limit if "top N" was given.
    """
    if tool_ref == "auto_advisor top_cars" and _needs_auto_filters(user_msg):
        tool_ref = "auto_advisor filter_cars"
        m = re.search(r'\btop\s*(\d{1,2})\b', (user_msg or "").lower())
        if m and "limit" not in (args or {}):
            args = dict(args or {})
            args["limit"] = int(m.group(1))
    return tool_ref, args


def conform_args_to_schema(user_msg: str, args: dict, schema: dict) -> dict:
    """
    - Maps aliases to schema names.
    - If the schema contains a 'params' object, it packs keys there.
    - Infers budget/limit/year from text (using parsers).
    - Heuristics for AutoAdvisor (fuel/transmission/condition/accident/price_max).
    - Heuristics for PokeVGC (format/playstyle/constraints.strategy.trick_room).
    - Converts team (list) to {"pokemon":[{"name":...}]} if applicable.
    - Fallback for 'trainer': packs 'params' if required.
    - Filters out properties that are not allowed at the end.
    """
    args = args or {}
    if not isinstance(schema, dict):
        return args

    props = schema.get("properties") or {}
    req = set(schema.get("required") or [])
    auto_hints = parse_auto_from_text(user_msg or "")

    if not isinstance(props, dict):
        return args

    canon_by_norm = {_norm_key(k): k for k in props.keys()}
    aliases = {
        # cars
        "make": "Car Make", "brand": "Car Make", "car make": "Car Make",
        "model": "Car Model", "car model": "Car Model",
        "year": "Year", "min year": "Year_min", "year min": "Year_min",
        "max year": "Year_max", "year max": "Year_max",
        "mileage": "Mileage",
        "price": "Price", "max price": "Price_max", "min price": "Price_min",
        "fuel": "Fuel Type", "fuel type": "Fuel Type",
        "transmission": "Transmission", "condition": "Condition",
        "accident": "Accident",
        "limit": "limit", "n": "n", "count": "count",
        "sort": "sort_order", "sort order": "sort_order",
        "budget": "budget_max", "max budget": "budget_max",
        # trainer
        "gender": "gender", "age": "age", "height cm": "height_cm", "weight kg": "weight_kg",
        "goal": "goal", "sport": "sport",
        "days per week": "days_per_week",
        "minutes per session": "minutes_per_session",
        "experience": "experience",
        "params": "params",
        # pokevgc
        "format": "format", "playstyle": "playstyle", "constraints": "constraints",
        "team": "team", "role": "role", "required ability": "required_ability",
        "min speed": "min_speed",
    }

    # params schema if it exists
    params_schema = None
    if "params" in props and isinstance(props["params"], dict):
        params_schema = props["params"].get("properties") or {}

    out: dict = {}

    # Map args → schema keys (root or params)
    for k, v in args.items():
        nk = _norm_key(k)
        target = aliases.get(nk) or canon_by_norm.get(nk)
        if not target:
            if params_schema:
                canon_params = {_norm_key(p): p for p in params_schema.keys()}
                t2 = aliases.get(nk) or canon_params.get(nk)
                if t2 and t2 in params_schema:
                    out.setdefault("params", {})[t2] = v
            continue

        if target == "params":
            if isinstance(v, dict) and params_schema:
                out["params"] = {subk: v[subk] for subk in v if subk in params_schema}
            elif isinstance(v, dict):
                out["params"] = v
            continue

        if target in props:
            out[target] = v
        elif params_schema and target in params_schema:
            out.setdefault("params", {})[target] = v

    # If params_schema exists and there are loose keys that belong to params → pack them
    if params_schema:
        keys_for_params = [k for k in list(out.keys()) if k in params_schema and k != "params"]
        if keys_for_params:
            out.setdefault("params", {})
            for k in keys_for_params:
                out["params"][k] = out.pop(k)

    # budget_max
    if "budget_max" in props and "budget_max" not in out:
        b = parse_budget_from_text_strict(user_msg)
        if b is not None:
            out["budget_max"] = b

    # mileage_max
    if "Mileage_max" in props and "Mileage_max" not in out:
        mmax = parse_mileage_max_from_text(user_msg)
        if mmax is not None:
            out["Mileage_max"] = mmax

    # Year_min / Year_max by range / plus
    ymin, ymax = parse_year_range_from_text(user_msg)
    if "Year_min" in props and "Year_min" not in out:
        ysolo = parse_year_min_from_text(user_msg)
        if ysolo:
            out["Year_min"] = ysolo
    if "Year_max" in props and "Year_max" not in out:
        ysolo = parse_year_max_from_text(user_msg)
        if ysolo:
            out["Year_max"] = ysolo

    # Normalize “petrol” → “Gasoline”
    if "Fuel Type" in out and isinstance(out["Fuel Type"], str):
        if out["Fuel Type"].lower() == "petrol":
            out["Fuel Type"] = "Gasoline"

    # Apply auto_hints ALWAYS (don't rely on Fuel Type)
    if "Price_max" in props and "Price_max" not in out:
        b = parse_budget_from_text_strict(user_msg)
        if b is not None:
            out["Price_max"] = b
    if "Mileage_max" in props and "Mileage_max" not in out and "Mileage_max" in auto_hints:
        out["Mileage_max"] = auto_hints["Mileage_max"]
    if "Body Style" in props and "__BODY_STYLE__" in auto_hints and "Body Style" not in out:
        out["Body Style"] = auto_hints["__BODY_STYLE__"]

    for k in ("Transmission", "Condition", "Accident"):
        if k in props and k not in out and k in auto_hints:
            out[k] = auto_hints[k]

    # Special trainer fallback: pack params if needed
    params_required = "params" in req
    if (params_required or any(k in args for k in ("gender","age","height_cm","weight_kg"))) \
       and ("params" not in out) and params_schema:
        candidate = {}
        for k in ("gender", "age", "height_cm", "weight_kg", "goal", "sport",
                  "days_per_week", "minutes_per_session", "experience", "limit", "limit"):
            if k in args and k in params_schema:
                candidate[k] = args[k]
        for k in list(out.keys()):
            if k in params_schema and k != "params":
                candidate[k] = out.pop(k)
        if candidate:
            out["params"] = candidate

    # PokeVGC heuristics
    t = (user_msg or "").lower()
    no_tr = re.search(r'\b(no|without|avoid|sin)\s+trick\s*room\b', t)

    if "playstyle" in props:
        if no_tr:
            pass  # DO NOT set trick_room
        elif re.search(r'\btrick\s*room\b', t):
            out["playstyle"] = "trick_room"

    if "format" in props:
        if "series 12" in t or "vgc series 12" in t:
            out["format"] = "vgc2022"
        elif "series 11" in t or "2021" in t:
            out.setdefault("format", "vgc2021")
        elif "2020" in t:
            out.setdefault("format", "vgc2020")

    if "constraints" in props and isinstance(props["constraints"], dict):
        cs = out.setdefault("constraints", {})
        strat_def = props["constraints"].get("properties", {}).get("strategy", {})
        if isinstance(strat_def, dict):
            if no_tr:
                cs.setdefault("strategy", {})["trick_room"] = False
            elif out.get("playstyle") == "trick_room":
                cs.setdefault("strategy", {})["trick_room"] = True

    # team.synergy → expected object
    if "team" in props and isinstance(out.get("team"), list):
        out["team"] = {"pokemon": [{"name": n} if isinstance(n, str) else n for n in out["team"]]}

    if "constraints" in props:
        cs = out.setdefault("constraints", {})
        hints = parse_poke_constraints_from_text(user_msg)
        for k, v in hints.items():
            cs.setdefault(k, v)

    # Final filtering (root and params)
    out = {k: v for k, v in out.items() if (k in props) or (k == "params" and params_schema)}
    if "params" in out and params_schema:
        out["params"] = {k: v for k, v in out["params"].items() if k in params_schema}

    if "required_ability" in props:
        if "intimidate" in t: out["required_ability"] = "Intimidate"
    if "role" in props:
        if re.search(r'\bfake\s*out\b', t): out["role"] = "fake_out"
        elif "redirection" in t: out["role"] = "redirection"
        elif "speed control" in t: out["role"] = "speed_control"

    return out

# ────────────────────────────────────────────────────────────────────────────────
# Short summaries of outputs (UI)
# ────────────────────────────────────────────────────────────────────────────────

def summarize_tool_output(parsed, user_msg: str, server: str, tool: str, args: dict) -> str:
    """Deterministic summary (1–2 sentences) ONLY from the tool output."""
    # 1) auto_advisor.average_price
    if isinstance(parsed, dict) and "average_price" in parsed:
        avg = parsed.get("average_price")
        samples = parsed.get("samples")
        filters = parsed.get("filters")
        ftxt = ""
        if isinstance(filters, dict) and filters:
            pairs = [f"{k}: {v}" for k, v in filters.items()]
            ftxt = " (" + ", ".join(pairs) + ")"
        try:
            avg_str = f"{float(avg):,.2f}"
        except Exception:
            avg_str = str(avg)
        if isinstance(samples, int):
            return f"The average price is {avg_str} based on {samples} samples{ftxt}."
        return f"The average price is {avg_str}{ftxt}."

    # auto_advisor.recommend / auto_advisor.top_cars / auto_advisor.filter_cars
    if isinstance(parsed, dict):
        rows = None
        if isinstance(parsed.get("results"), list) and parsed["results"]:
            rows = parsed["results"]
        elif isinstance(parsed.get("recommendations"), list) and parsed["recommendations"]:
            rows = parsed["recommendations"]

        if rows:
            def label(x):
                if isinstance(x, dict):
                    if "Car Make" in x and "Car Model" in x:
                        return f"{x.get('Car Make')} {x.get('Car Model')}".strip()
                    for key in ("name", "Name", "title", "model"):
                        if key in x and x[key]:
                            return str(x[key])
                    for v in x.values():
                        if v:
                            return str(v)
                return str(x)

            tops = ", ".join(label(it) for it in rows[:3])
            n = len(rows)
            hint = ""
            if "budget_max" in parsed:
                try:
                    hint = f" under {float(parsed['budget_max']):,.2f}"
                except Exception:
                    hint = f" under {parsed['budget_max']}"
            return f"Top matches{hint} include {tops}. Showing {min(3, n)} of {n}."

    # trainer.compute_metrics
    if isinstance(parsed, dict) and ("bmi" in parsed or "bmr" in parsed):
        bmi = parsed.get("bmi")
        bmr = parsed.get("bmr")
        cls = parsed.get("bmi_clase") or parsed.get("bmi_class")
        parts = []
        if bmi is not None:
            try:
                parts.append(f"BMI {float(bmi):.2f}")
            except Exception:
                parts.append(f"BMI {bmi}")
        if cls:
            parts.append(str(cls))
        if bmr is not None:
            try:
                parts.append(f"BMR {float(bmr):.1f}")
            except Exception:
                parts.append(f"BMR {bmr}")
        return ", ".join(parts) + "."

    # pokevgc.suggest_team / suggest_member / pool.filter / team.synergy
    if isinstance(parsed, dict) and "team" in parsed:
        team = parsed.get("team", {})
        names = []
        if isinstance(team, dict) and isinstance(team.get("pokemon"), list):
            for p in team["pokemon"][:6]:
                if isinstance(p, dict) and p.get("name"):
                    names.append(str(p["name"]))
        if names:
            return f"Suggested team: {', '.join(names)}."

    # some poke builder endpoints return a list directly
    if isinstance(parsed, list) and parsed:
        def name_of(x):
            if isinstance(x, dict):
                return x.get("name") or x.get("Name") or x.get("title") or next((v for v in x.values() if v), None)
            return x
        labels = [str(name_of(it)) for it in parsed[:3]]
        return f"Top matches include {', '.join(labels)}. Showing {min(3,len(parsed))} of {len(parsed)}."

    # plain text
    if isinstance(parsed, str) and parsed.strip():
        return parsed.strip()

    # slast resort
    return f"Here is the result from {server}.{tool}."

# ────────────────────────────────────────────────────────────────────────────────
# Main REPL
# ────────────────────────────────────────────────────────────────────────────────

async def repl_async():
    console.rule("[bold cyan]MCP_AutoHost")
    provider_hint = os.getenv("LLM_PROVIDER", "auto")
    model_hint = os.getenv("LLM_MODEL", "(default)")
    console.print(f"[bold]Provider:[/bold] {provider_hint}   [bold]Model:[/bold] {model_hint}")
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")):
        console.print("[red]No API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")
        sys.exit(1)

    mem = Memory(max_messages=20)
    client = LLMClient()

    # Start ALL MCP servers and build the tools catalog
    cfg = yaml.safe_load(open(CONFIG_FILE, "r", encoding="utf-8"))
    sm = ServerManager(cfg["servers"], workspace=os.getcwd())
    await sm.start_all()

    tool_index = await sm.build_tool_index()  # {(server, tool): schema}
    orchestrator = NaturalLanguageOrchestrator(tool_index=tool_index)

    print_help()
    try:
        while True:
            try:
                user = console.input("\n[bold green]> [/bold green]").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\nClosing…")
                break

            if not user:
                continue
            if user == "/exit":
                break
            if user == "/help":
                print_help()
                continue
            if user.startswith("/logs"):
                parts = user.split()
                n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 50
                events = tail_logs(n)
                if not events:
                    console.print("[yellow]No logs yet.]")
                else:
                    for e in events:
                        console.print(e)
                continue
            if user == "tools":
                catalog = orchestrator.pretty_tools_catalog()
                console.print(Panel.fit(catalog, title="🛠️ Tools Catalog", border_style="magenta"))
                continue

            if user == "context":
                try:
                    ctx = mem.dump_json()
                except Exception:
                    ctx = "[no dump_json() implemented]"
                console.print(Panel.fit(ctx, title="Context", border_style="cyan"))
                continue

            if user == "/reset":
                mem.reset()
                console.print("[cyan]Memory cleared.")
                continue

            # ================== NL → (server.tool + args) → JSON ==================
            selection = await orchestrator.select_tool_and_args(client, user, mem)
            tool_ref = selection.get("tool_ref")             # "server.tool" or None
            tool_args = selection.get("arguments", {}) or {}
            reasoning = selection.get("reasoning_summary", "")

            result_json = {
                "input": user,
                "plan": {"tool_ref": tool_ref, "arguments": tool_args, "why": reasoning},
            }

            if tool_ref:
                result_json = {
                    "input": user,
                    "plan": {"tool_ref": tool_ref, "arguments": tool_args, "why": reasoning},
                }
                try:
                    server_name, tool_name = tool_ref.split(".", 1)
                except ValueError:
                    result_json["error"] = f"Bad tool_ref: {tool_ref}"
                    body = json.dumps(result_json, ensure_ascii=False, indent=2)
                    console.print(Panel(_safe_panel_text(body), title="Result", border_style="red"))
                else:
                    try:
                        # 1) Normalize args against the schema of the tool (alias, year_min, budget, count, etc.)
                        schema = tool_index.get((server_name, tool_name), {}) or {}
                        norm_args = conform_args_to_schema(user, tool_args, schema)

                        if _should_apply_trainer_packing(server_name, tool_name, schema):
                            norm_args = force_trainer_params(user, norm_args, schema)

                        #print("DEBUG tool_ref:", server_name, tool_name)
                        #print("DEBUG required:", schema.get("required"))
                        #print("DEBUG props:", list((schema.get("properties") or {}).keys()))
                        #print("DEBUG norm_args:", json.dumps(norm_args, ensure_ascii=False))

                        # Defensive fallback: if the schema requires 'params' and it is not yet
                        #    and the tool DECLARES the 'params' property, force packing.
                        req = schema.get("required") or []
                        has_params_prop = isinstance(schema.get("properties"), dict) and "params" in schema["properties"]
                        if ("params" in req) and has_params_prop and ("params" not in norm_args):
                            norm_args = force_trainer_params(user, norm_args, schema)

                        # Call the tool
                        tool_output_text = await sm.call_tool(server_name, tool_name, norm_args)

                        # Try to parse JSON (if not, leave raw text)
                        parsed = None
                        try:
                            parsed = json.loads(tool_output_text)
                            pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
                        except Exception:
                            pretty = tool_output_text

                        # Deterministic summary ONLY from the tool's JSON/text output
                        summary = summarize_tool_output(parsed if parsed is not None else tool_output_text,
                                                        user, server_name, tool_name, norm_args)

                        # Show paragraph
                        console.print(Panel.fit(_safe_panel_text(summary), title="Assistant", border_style="blue"))

                        # preview in table if there is a list
                        if isinstance(parsed, dict):
                            preview_tbl = _build_preview_table_from_json(parsed, user_msg=user, max_rows=3)
                            if preview_tbl is not None:
                                console.print(preview_tbl)
                                 
                        # Full JSON / raw text
                        console.print(Panel(_safe_panel_text(pretty), title="Result", border_style="magenta"))

                    except Exception as e:
                        debug_payload = {
                            "error": f"{type(e).__name__}: {e}",
                            "server": server_name,
                            "tool": tool_name,
                            "schema": schema,
                            "norm_args": norm_args if 'norm_args' in locals() else tool_args,
                        }
                        body = json.dumps(debug_payload, ensure_ascii=False, indent=2)
                        console.print(Panel(_safe_panel_text(body), title="Result", border_style="red"))
            else:
                # Fallback: ONLY nice text (no JSON)
                fallback = await orchestrator.basic_fallback(client, user, mem)
                mem.add_user(user)
                mem.add_assistant(fallback)
                console.print(Panel.fit(fallback, title="Assistant", border_style="blue"))
            # =====================================================================

    finally:
        try:
            await sm.stop_all()
        except Exception:
            pass

def repl():
    """Wrapper to run the async REPL with a clean event loop."""
    asyncio.run(repl_async())

if __name__ == "__main__":
    repl()
