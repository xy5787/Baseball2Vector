"""Phase 5 tool signatures (§7) — real implementations, not fakes.

Only the LM *call* is stubbed in this phase (see lm_arms/client.py). These
five functions genuinely compute their answers by wrapping the already-built
Phase 1-4 code: predict_value_delta uses the Phase 3 extended LR fit,
check_delta_feasibility uses the Phase 2 empirical lookup, find_comparables
reuses the Phase 3 KNN pool. Age is not a parameter — dropped throughout this
project per the Phase 0 data-availability finding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from lm_experiments.baseline_models import build_knn_pool, fit_lr, time_split
from lm_experiments.transitions import drop_ambiguous_keys, load_v3_results

TOOL_ORDER_DEFAULT = ["Power", "Contact", "Discipline", "Defense", "Speed"]


def _vec5_schema(description: str, scouting_scale: bool = False) -> dict:
    value_schema = {"type": "number"}
    if scouting_scale:
        value_schema.update({"minimum": 20, "maximum": 80})
    return {
        "type": "object",
        "description": description,
        "properties": {t: dict(value_schema) for t in TOOL_ORDER_DEFAULT},
        "required": TOOL_ORDER_DEFAULT,
    }


GET_PLAYER_VECTOR_SCHEMA = {
    "name": "get_player_vector",
    "description": (
        "Look up a real player's 5-tool vector for a specific season. Call "
        "this when a comparable name from find_comparables is worth "
        "inspecting directly, or to compare a hypothetical vector against a "
        "known real player."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "player_id": {"type": "string", "description": "Player name, e.g. 'Juan Soto'."},
            "season": {"type": "integer", "description": "Season year, e.g. 2023."},
        },
        "required": ["player_id", "season"],
    },
}

APPLY_TOOL_DELTA_SCHEMA = {
    "name": "apply_tool_delta",
    "description": (
        "Shift one tool in a 5-tool vector by a magnitude, returning the "
        "resulting vector. Use this to construct a hypothetical comparison "
        "point rather than adding the numbers yourself."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "vec5": _vec5_schema("The base 5-tool vector."),
            "tool": {"type": "string", "enum": TOOL_ORDER_DEFAULT, "description": "Which tool to shift."},
            "magnitude": {"type": "number", "description": "Amount to add (can be negative)."},
        },
        "required": ["vec5", "tool", "magnitude"],
    },
}

PREDICT_VALUE_DELTA_SCHEMA = {
    "name": "predict_value_delta",
    "description": (
        "Run the project's fitted regression model (delta-tool features + "
        "season-t base level + interaction terms) to get a numeric point "
        "estimate for delta WAR, delta wRC+, and delta OPS. Call this before "
        "finalizing your prediction -- it is your numeric anchor, computed "
        "by code, not a suggestion to estimate yourself."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "delta_vec": _vec5_schema("The observed delta vector (season t -> t+1)."),
            "base_vec": _vec5_schema("The season-t vector."),
        },
        "required": ["delta_vec", "base_vec"],
    },
}

FIND_COMPARABLES_SCHEMA = {
    "name": "find_comparables",
    "description": (
        "Find the k real players (by name and season) whose season-t vector "
        "is nearest to a given 5-tool vector. Returns real identities even "
        "when your own subject is anonymized -- use it to reason about "
        "comparable profiles, not to guess who your subject is."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "vec5": _vec5_schema(
                "Current-level query vector on the 20-80 scouting scale; "
                "do not pass a delta vector.",
                scouting_scale=True,
            ),
            "k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Number of neighbors, default 10.",
                "default": 10,
            },
        },
        "required": ["vec5"],
    },
}

CHECK_DELTA_FEASIBILITY_SCHEMA = {
    "name": "check_delta_feasibility",
    "description": (
        "Return the empirical percentile of a delta-tool magnitude among "
        "real observed transitions starting from a similar current value. "
        "Call this to judge whether an observed or hypothetical delta is "
        "typical or extreme for a player at that starting level -- e.g. "
        "whether pushing an already-elite tool further is something that "
        "actually happens in the data."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": TOOL_ORDER_DEFAULT},
            "current_value": {"type": "number", "description": "Season-t tool value (20-80 scale)."},
            "magnitude": {"type": "number", "description": "The delta magnitude to evaluate."},
        },
        "required": ["tool", "current_value", "magnitude"],
    },
}

SUBMIT_PREDICTION_SCHEMA = {
    "name": "submit_prediction",
    "description": (
        "Submit your final prediction. This is the only way to answer -- "
        "always call this exactly once, as your last action."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "delta_war": {"type": "number"},
            "delta_wrc": {"type": "number"},
            "delta_ops": {"type": "number"},
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "0-1 confidence in this prediction.",
            },
            "comparables": {
                "type": "array",
                "items": {"type": "string", "pattern": "^.+ \\(\\d{4}\\)$"},
                "description": "Comparable player ids, if any were used.",
                "maxItems": 10,
            },
            "rationale": {
                "type": "string",
                "description": "Brief explanation of your reasoning and which tool outputs (if any) drove it.",
            },
        },
        "required": ["delta_war", "delta_wrc", "delta_ops"],
    },
}

TOOL_SCHEMAS_D = [
    GET_PLAYER_VECTOR_SCHEMA,
    APPLY_TOOL_DELTA_SCHEMA,
    PREDICT_VALUE_DELTA_SCHEMA,
    FIND_COMPARABLES_SCHEMA,
    CHECK_DELTA_FEASIBILITY_SCHEMA,
]

_TARGET_KEY = {"WAR": "war", "wRC+": "wrc", "OPS": "ops"}


class ToolBox:
    """Loads Phase 1-3 artifacts once and dispatches the five tool calls."""

    def __init__(self, cfg: dict, excluded_player_id: str | None = None):
        self.tool_order = cfg["transitions"]["tool_order"]
        self.source_col_prefix = cfg["transitions"]["source_col_prefix"]
        self.default_k = int(cfg["baselines"]["knn_k"])
        self.excluded_player_id = excluded_player_id
        self._authorized_comparable_ids: set[str] = set()

        raw = load_v3_results(cfg["data"]["v3_results"])
        self.v3, _ = drop_ambiguous_keys(raw)

        windows = [tuple(w) for w in cfg["transitions"]["windows"]]
        train_seasons = sorted({s for w in windows[:-1] for s in w})
        self.allowed_seasons = set(train_seasons)
        self.lookup_v3 = self.v3[self.v3["Season"].isin(self.allowed_seasons)].copy()
        if self.excluded_player_id is not None:
            self.lookup_v3 = self.lookup_v3[
                self.lookup_v3["Name"] != self.excluded_player_id
            ].copy()

        tables_dir = Path(cfg["output"]["tables_dir"])
        transitions = pd.read_csv(tables_dir / "transitions.csv")
        train_df, _ = time_split(transitions, windows[-1])
        self._lr_fits = {
            target: fit_lr(
                train_df,
                self.tool_order,
                target,
                extended=True,
                alpha=cfg["baselines"]["ci_alpha"],
            )
            for target in ["WAR", "wRC+", "OPS"]
        }

        self.knn_pool = build_knn_pool(
            self.lookup_v3, train_seasons, self.tool_order, cfg["transitions"]["min_pa"]
        )

        with open(tables_dir / "feasibility_lookup.json") as f:
            self._feasibility_lookup = json.load(f)

    def get_player_vector(self, player_id: str, season: int) -> dict[str, float] | None:
        unique_name = f"{player_id} ({season})"
        if unique_name not in self._authorized_comparable_ids:
            return None
        row = self.lookup_v3[
            (self.lookup_v3["Name"] == player_id)
            & (self.lookup_v3["Season"] == season)
        ]
        if row.empty:
            return None
        r = row.iloc[0]
        return {t: float(r[f"{self.source_col_prefix}{t}"]) for t in self.tool_order}

    def apply_tool_delta(
        self, vec5: dict[str, float], tool: str, magnitude: float
    ) -> dict[str, float]:
        if tool not in self.tool_order:
            raise ValueError(f"unknown tool {tool!r}, expected one of {self.tool_order}")
        out = dict(vec5)
        out[tool] = out[tool] + magnitude
        return out

    def predict_value_delta(
        self, delta_vec: dict[str, float], base_vec: dict[str, float]
    ) -> dict[str, float]:
        row = {}
        for t in self.tool_order:
            row[f"delta_vec_{t.lower()}"] = delta_vec[t]
            row[f"vec_t_{t.lower()}"] = base_vec[t]
        df_row = pd.DataFrame([row])
        return {
            _TARGET_KEY[target]: float(fit.predict(df_row)[0])
            for target, fit in self._lr_fits.items()
        }

    def find_comparables(
        self, vec5: dict[str, float], k: int | None = None
    ) -> list[str]:
        k = self.default_k if k is None else int(k)
        if not 1 <= k <= 10:
            raise ValueError("k must be between 1 and 10")
        values = [float(vec5[t]) for t in self.tool_order]
        if not all(np.isfinite(value) and 20 <= value <= 80 for value in values):
            raise ValueError(
                "find_comparables vec5 must contain current tool levels in [20, 80]; "
                "a delta vector is not a valid query"
            )
        pool_cols = [f"{self.source_col_prefix}{t}" for t in self.tool_order]
        pool_X = self.knn_pool[pool_cols].to_numpy(dtype=float)
        query = np.array([values], dtype=float)
        nn = NearestNeighbors(n_neighbors=min(k, len(pool_X))).fit(pool_X)
        _, idx = nn.kneighbors(query)
        comparable_ids = self.knn_pool.iloc[idx[0]]["UniqueName"].tolist()
        self._authorized_comparable_ids.update(comparable_ids)
        return comparable_ids

    def check_delta_feasibility(
        self, tool: str, current_value: float, magnitude: float
    ) -> float | None:
        from lm_experiments.feasibility import check_delta_feasibility

        return check_delta_feasibility(self._feasibility_lookup, tool, current_value, magnitude)

    def dispatch(self, name: str, tool_input: dict) -> object:
        """Execute a tool call by name; raises on an unknown tool name."""
        if name == "get_player_vector":
            return self.get_player_vector(tool_input["player_id"], tool_input["season"])
        if name == "apply_tool_delta":
            return self.apply_tool_delta(
                tool_input["vec5"], tool_input["tool"], tool_input["magnitude"]
            )
        if name == "predict_value_delta":
            return self.predict_value_delta(tool_input["delta_vec"], tool_input["base_vec"])
        if name == "find_comparables":
            return self.find_comparables(
                tool_input["vec5"], tool_input.get("k", self.default_k)
            )
        if name == "check_delta_feasibility":
            return self.check_delta_feasibility(
                tool_input["tool"], tool_input["current_value"], tool_input["magnitude"]
            )
        raise ValueError(f"unknown tool {name!r}")
