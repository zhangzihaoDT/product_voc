import argparse
import hashlib
import math
from pathlib import Path

import pandas as pd


THEME_FACT_COLUMNS = [
    "theme_id",
    "need_theme_l1",
    "need_theme_l2",
    "issue_count",
    "affected_models",
    "app_count",
    "phone400_count",
    "avg_severity",
    "avg_sentiment",
    "priority_score",
    "sample_quotes",
]


def _to_str(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return str(v).strip()


def _to_float(v, default: float = 0.0) -> float:
    s = _to_str(v)
    if not s:
        return default
    try:
        return float(s)
    except Exception:
        return default


def _to_int(v, default: int = 0) -> int:
    s = _to_str(v)
    if not s:
        return default
    try:
        return int(float(s))
    except Exception:
        return default


def _channel_weight(channel: str) -> float:
    c = _to_str(channel).lower()
    if "400" in c:
        return 1.2
    if "app" in c:
        return 1.0
    return 1.0


def _theme_id(l1: str, l2: str) -> str:
    base = f"{_to_str(l1)}|{_to_str(l2)}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def _compact_text(text: str) -> str:
    t = _to_str(text).replace("\r\n", "\n").replace("\r", "\n")
    t = " ".join(t.split())
    return t


def build_theme_fact(issue_fact_refined: pd.DataFrame) -> pd.DataFrame:
    df = issue_fact_refined.copy()
    if "is_voc" in df.columns:
        is_voc = df["is_voc"].astype(str).str.lower().isin(["true", "1", "yes"])
        df = df[is_voc].copy()

    df["need_theme_l1"] = df.get("need_theme_l1", "").map(_to_str)
    df["need_theme_l2"] = df.get("need_theme_l2", "").map(_to_str)
    df.loc[df["need_theme_l2"] == "", "need_theme_l2"] = "其他"
    df.loc[df["need_theme_l1"] == "", "need_theme_l1"] = "其他"

    df["frequency_score"] = df.get("frequency_score", 1).map(lambda x: _to_int(x, 1))
    df["impact_score"] = df.get("impact_score", 0.0).map(lambda x: _to_float(x, 0.0))
    df["severity"] = df.get("severity", 1).map(lambda x: _to_int(x, 1))
    df["sentiment"] = df.get("sentiment", 0).map(lambda x: _to_int(x, 0))
    df["channel_weight"] = df.get("channel", "").map(_channel_weight)

    df["issue_priority_raw"] = df["frequency_score"] * df["impact_score"] * df["severity"] * df["channel_weight"]

    def format_models(s: pd.Series) -> str:
        counts = s.map(_to_str)
        counts = counts[counts != ""]
        if counts.empty:
            return ""
        vc = counts.value_counts()
        parts = [f"{k}:{int(v)}" for k, v in vc.head(8).items()]
        return "|".join(parts)

    def sample_quotes(sub: pd.DataFrame) -> str:
        items = []
        for _, r in sub.sort_values("issue_priority_raw", ascending=False).head(3).iterrows():
            q = _compact_text(_to_str(r.get("raw_text")) or _to_str(r.get("problem")))
            if q:
                items.append(q[:180])
        return " | ".join(items)

    grouped = []
    for (l1, l2), sub in df.groupby(["need_theme_l1", "need_theme_l2"], dropna=False):
        l1s = _to_str(l1) or "其他"
        l2s = _to_str(l2) or "其他"
        issue_count = int(len(sub))
        app_count = int(sub["channel"].astype(str).str.contains("APP", case=False, na=False).sum()) if "channel" in sub.columns else 0
        phone400_count = int(sub["channel"].astype(str).str.contains("400", case=False, na=False).sum()) if "channel" in sub.columns else 0

        avg_sev = float(round(sub["severity"].mean(), 3)) if issue_count else 0.0
        avg_sent = float(round(sub["sentiment"].mean(), 3)) if issue_count else 0.0
        raw_score = float(sub["issue_priority_raw"].sum()) if issue_count else 0.0

        grouped.append(
            {
                "theme_id": _theme_id(l1s, l2s),
                "need_theme_l1": l1s,
                "need_theme_l2": l2s,
                "issue_count": issue_count,
                "affected_models": format_models(sub.get("car_series", pd.Series(dtype="object"))),
                "app_count": app_count,
                "phone400_count": phone400_count,
                "avg_severity": avg_sev,
                "avg_sentiment": avg_sent,
                "priority_score_raw": raw_score,
                "sample_quotes": sample_quotes(sub),
            }
        )

    theme = pd.DataFrame(grouped)
    if theme.empty:
        return theme.reindex(columns=THEME_FACT_COLUMNS)

    max_raw = float(theme["priority_score_raw"].max()) if not theme["priority_score_raw"].isna().all() else 0.0
    if max_raw > 0:
        theme["priority_score"] = (theme["priority_score_raw"] / max_raw * 100.0).round(2)
    else:
        theme["priority_score"] = 0.0

    theme = theme.sort_values(["priority_score", "issue_count"], ascending=[False, False])
    theme = theme.drop(columns=["priority_score_raw"], errors="ignore").reindex(columns=THEME_FACT_COLUMNS)
    return theme


def export_need_theme_other(df: pd.DataFrame, output_path: Path) -> int:
    work = df.copy()
    if "is_voc" in work.columns:
        is_voc = work["is_voc"].astype(str).str.lower().isin(["true", "1", "yes"])
        work = work[is_voc].copy()
    work["need_theme_l2"] = work.get("need_theme_l2", "").map(_to_str)
    other = work[work["need_theme_l2"] == "其他"].copy()
    cols = [
        "feedback_id",
        "issue_no",
        "source",
        "channel",
        "car_series",
        "created_time",
        "raw_text",
        "domain",
        "module",
        "scenario",
        "problem",
        "severity",
        "sentiment",
        "impact_score",
        "frequency_score",
        "priority_score",
    ]
    cols = [c for c in cols if c in other.columns]
    other.to_csv(output_path, index=False, encoding="utf-8-sig", columns=cols)
    return int(len(other))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="issue_fact_refined_v3.csv")
    parser.add_argument("--output-theme", default="theme_fact.csv")
    parser.add_argument("--output-other", default="need_theme_other_issues.csv")
    args = parser.parse_args()

    issue = pd.read_csv(args.input, dtype="object")
    theme = build_theme_fact(issue)
    theme.to_csv(args.output_theme, index=False, encoding="utf-8-sig")

    other_count = export_need_theme_other(issue, Path(args.output_other))
    print(f"Wrote {len(theme)} rows to {args.output_theme}")
    print(f"Wrote {other_count} rows to {args.output_other}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

