# src/host/nl_router.py
from typing import Dict, Any, Tuple, Optional, List

class NaturalLanguageOrchestrator:
    def __init__(self, tool_index: Dict[Tuple[str, str], Dict[str, Any]]):
        # {(server, tool): inputSchema}
        self.tool_index = tool_index

    def pretty_tools_catalog(self) -> str:
        lines = []
        for (srv, tool), schema in sorted(self.tool_index.items()):
            lines.append(f"{srv}.{tool}")
        return "\n".join(lines) if lines else "(no tools available)"

    async def basic_fallback(self, llm, user_message: str, memory) -> str:
        """
        Respuesta genérica cuando no hay herramienta que usar.
        Usa el historial de 'memory' si existe.
        """
        system = (
            "You are a helpful assistant. Answer clearly and concisely. "
            "If the user asks general knowledge, just answer directly."
        )
        history = []
        try:
            history = memory.history()  # [{"role": "...", "content": "..."}]
        except Exception:
            history = []

        return await llm.chat_turn(
            system=system,
            history=history,
            user=user_message,
            temperature=0.2,
        )

    async def select_tool_and_args(self, llm, user_message: str, memory) -> Dict[str, Any]:
        """
        Devuelve: {"tool_ref": "server.tool" | None, "arguments": {...} | {}, "reasoning_summary": "..."}
        Pide al LLM una propuesta; valida contra tool_index; aplica filtro de intención
        (filesystem/git) y fallback por keywords si la selección no es apropiada.
        """
        # Utilidades locales
        def _mentions_any(text: str, kws: List[str]) -> bool:
            t = (text or "").lower()
            return any(k in t for k in kws)

        # 1) Pide al LLM una selección en JSON
        system = (
            "You are a router. Pick exactly one available tool and output JSON with keys: "
            "{\"tool_ref\": \"server.tool\" | null, \"arguments\": {..}, \"reasoning_summary\": \"...\"}. "
            "Only choose from the provided tools."
        )
        tools_catalog_str = "\n".join([f"- {srv}.{tool}" for (srv, tool) in self.tool_index.keys()])
        user = (
            f"User query: {user_message}\n\n"
            f"Available tools:\n{tools_catalog_str}\n\n"
            "Return ONLY JSON, no prose."
        )

        out = await llm.complete_json(
            system=system,
            user=user,
            temperature=0.0,
            json_fallback={"tool_ref": None, "arguments": {}, "reasoning_summary": "No tools available."},
        )

        # 2) Normaliza y valida tool_ref contra el catálogo
        valid_map = {f"{s}.{t}": (s, t) for (s, t) in self.tool_index.keys()}
        pred = (out.get("tool_ref") or "").strip()

        # 3) Filtro por intención (no permitas filesystem/git sin intención clara)
        FILE_SERVERS = {"filesystem"}
        GIT_SERVERS  = {"git"}

        if pred in valid_map:
            srv, tl = valid_map[pred]
            file_intent = _mentions_any(user_message, [
                "file", "files", "open", "read", "path", "folder", "directory",
                "list files", "ls", "contenido", "archivo", "carpeta", "ruta"
            ])
            git_intent = _mentions_any(user_message, [
                "git", "commit", "branch", "push", "pull", "merge", "rebase",
                "repo", "repository", "tag", "staging"
            ])
            if (srv in FILE_SERVERS and not file_intent) or (srv in GIT_SERVERS and not git_intent):
                pred = None

        # 4) Si la elección quedó inválida, aplica fallback heurístico por dominio
        if pred not in valid_map:
            text = (user_message or "").lower()
            candidate: Optional[str] = None

            # BMI/BMR / entrenador (chatbot_server.compute_metrics)
            if any(k in text for k in ["bmi", "bmr", "height", "weight", "peso", "altura", "edad", "calories", "metabolism"]):
                if "chatbot_server.compute_metrics" in valid_map:
                    candidate = "chatbot_server.compute_metrics"

            # Pokémon VGC
            if candidate is None and any(k in text for k in ["pokemon", "pokémon", "vgc"]):
                if any(k in text for k in ["team", "equipo", "balanced", "trick room", "trickroom"]):
                    if "pokevgc.suggest_team" in valid_map:
                        candidate = "pokevgc.suggest_team"
                else:
                    if "pokevgc.suggest_member" in valid_map:
                        candidate = "pokevgc.suggest_member"
                    elif "pokevgc.pool.filter" in valid_map:
                        candidate = "pokevgc.pool.filter"

            # Autos
            if candidate is None and any(k in text for k in [
                "car", "cars", "auto", "coche", "mileage", "diesel", "gasoline", "hybrid",
                "accident", "price", "budget", "cheap", "expensive", "safest", "seguro"
            ]):
                if "average" in text or "promedio" in text:
                    if "auto_advisor.average_price" in valid_map:
                        candidate = "auto_advisor.average_price"
                elif any(k in text for k in ["recommend", "recomendar", "budget", "barato", "cheap"]):
                    if "auto_advisor.recommend" in valid_map:
                        candidate = "auto_advisor.recommend"
                elif any(k in text for k in ["safe", "safest", "accident", "seguro"]):
                    if "auto_advisor.filter_cars" in valid_map:
                        candidate = "auto_advisor.filter_cars"
                if candidate is None and "auto_advisor.filter_cars" in valid_map:
                    candidate = "auto_advisor.filter_cars"

            pred = candidate if (candidate and candidate in valid_map) else None

        # =========================
        # >>>>>>> PARCHE AQUÍ <<<<<
        # =========================
        # Ajustes finos a la selección y a los argumentos antes de devolver.
        args = out.get("arguments") or {}
        lowered = (user_message or "").lower()

        # 4.a) Si el LLM eligió top_cars pero el usuario dio filtros avanzados,
        #      forzamos filter_cars y mapeamos argumentos.
        if pred == "auto_advisor.top_cars":
            wants_accident_free   = any(k in lowered for k in ["accident-free", "no accidents"])
            wants_transmission    = any(k in lowered for k in ["automatic", "manual"])
            wants_fuel           = any(k in lowered for k in ["diesel", "hybrid", "electric", "gasoline", "petrol"])
            wants_year_or_mileage = any(k in lowered for k in ["year", "since", "+", "≤", "<=", "under", "mileage", "km", "miles"])

            # Importa parsers del paquete local
            try:
                from .parsers import (
                    parse_budget_from_text_strict,
                    parse_mileage_max_from_text,
                    parse_year_range_from_text,
                    parse_count_from_text,
                )
            except Exception:
                parse_budget_from_text_strict = None
                parse_mileage_max_from_text = None
                parse_year_range_from_text = None
                parse_count_from_text = None

            has_budget = parse_budget_from_text_strict and (parse_budget_from_text_strict(user_message) is not None)
            yr = parse_year_range_from_text(user_message) if parse_year_range_from_text else (None, None)
            mm = parse_mileage_max_from_text(user_message) if parse_mileage_max_from_text else None
            has_year_or_mileage = wants_year_or_mileage or (yr != (None, None)) or (mm is not None)

            if wants_accident_free or wants_transmission or wants_fuel or has_budget or has_year_or_mileage:
                if "auto_advisor.filter_cars" in valid_map:
                    pred = "auto_advisor.filter_cars"

                    # Construir args para filter_cars
                    limit = args.get("limit") or args.get("n")
                    if not limit and parse_count_from_text:
                        limit = parse_count_from_text(user_message)
                    if not limit:
                        limit = 3

                    new_args = {"limit": limit}

                    # Precio
                    if parse_budget_from_text_strict:
                        budget = parse_budget_from_text_strict(user_message)
                        if budget is not None:
                            new_args["Price_max"] = budget

                    # Año
                    if parse_year_range_from_text:
                        ymin, ymax = parse_year_range_from_text(user_message)
                        if ymin: new_args["Year_min"] = ymin
                        if ymax: new_args["Year_max"] = ymax

                    # Kilometraje
                    if parse_mileage_max_from_text:
                        mmax = parse_mileage_max_from_text(user_message)
                        if mmax is not None:
                            new_args["Mileage_max"] = mmax

                    # Transmisión / Fuel / Accident
                    if "automatic" in lowered: new_args["Transmission"] = "automatic"
                    if "manual" in lowered:    new_args["Transmission"] = "manual"
                    if "diesel" in lowered:    new_args["Fuel Type"] = "diesel"
                    if "hybrid" in lowered:    new_args["Fuel Type"] = "hybrid"
                    if "electric" in lowered:  new_args["Fuel Type"] = "electric"
                    if any(k in lowered for k in ["gasoline", "petrol", "nafta"]):
                        new_args["Fuel Type"] = "gasoline"
                    if wants_accident_free:
                        new_args["Accident"] = "No"

                    # Conserva lo ya sugerido si aplica
                    for k in ("Car Make","Car Model","Transmission","Fuel Type","Condition","Accident","Year_min","Year_max","Mileage_max","Price_max"):
                        if k in args and k not in new_args:
                            new_args[k] = args[k]

                    args = new_args

        # 4.b) Si el usuario pidió rutina y falta 'objetivo', infiérelo o usa default.
        if pred == "chatbot_server.build_routine_tool":
            params = (args.get("params") or {}).copy()
            if "objetivo" not in params:
                try:
                    from .parsers import parse_trainer_generic_from_text
                except Exception:
                    parse_trainer_generic_from_text = None
                objetivo = None
                if parse_trainer_generic_from_text:
                    g = parse_trainer_generic_from_text(user_message) or {}
                    objetivo = g.get("objetivo")
                params["objetivo"] = objetivo or "endurance"
            args["params"] = params

        # 4.c) recommend_exercises: si falta objetivo o limite, infiere/default
        if pred == "chatbot_server.recommend_exercises":
            params = (args.get("params") or {}).copy()
            try:
                from .parsers import parse_trainer_generic_from_text, parse_count_from_text
                g = parse_trainer_generic_from_text(user_message) or {}
                if "objetivo" not in params and g.get("objetivo"):
                    params["objetivo"] = g["objetivo"]
                if "limite" not in params:
                    c = parse_count_from_text(user_message)
                    if c:
                        params["limite"] = c
            except Exception:
                if "objetivo" not in params:
                    params["objetivo"] = "endurance"
            args["params"] = params
        # =========================
        # >>>>> FIN PARCHE <<<<<<<
        # =========================

        # 5) Retorno final
        if pred is None:
            return {
                "tool_ref": None,
                "arguments": {},
                "reasoning_summary": out.get("reasoning_summary") or "No tools available.",
            }

        return {
            "tool_ref": pred,
            "arguments": args or {},
            "reasoning_summary": out.get("reasoning_summary") or "",
        }
