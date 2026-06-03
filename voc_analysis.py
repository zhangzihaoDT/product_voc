import argparse
import json
from pathlib import Path

import pandas as pd


def load_csv(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path, dtype="object", encoding="utf-8-sig")


def load_definition(path: Path | str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _clean_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def _score_bucket(cnt: int, max_cnt: int) -> int:
    if max_cnt <= 0:
        return 1
    ratio = cnt / max_cnt
    if ratio >= 0.8:
        return 5
    if ratio >= 0.6:
        return 4
    if ratio >= 0.3:
        return 3
    if ratio >= 0.1:
        return 2
    return 1


def parse_definition(definition: dict) -> dict:
    stages = []
    jtbd_order = []
    jtbd_stage_map = {}
    jtbd_scenarios = {}
    scenario_needs = {}

    for lifecycle in definition.get("lifecycles", []):
        stage_name = lifecycle["name"]
        jtbd_names = []
        for jtbd in lifecycle.get("jtbds", []):
            jtbd_name = jtbd["name"]
            jtbd_names.append(jtbd_name)
            jtbd_order.append(jtbd_name)
            jtbd_stage_map[jtbd_name] = stage_name
            scenarios = []
            for sc in jtbd.get("scenarios", []):
                sc_name = sc["name"]
                scenarios.append(sc_name)
                needs = [(nt["need_theme_l1"], nt["need_theme_l2"])
                         for nt in sc.get("need_themes", [])]
                scenario_needs[(jtbd_name, sc_name)] = needs
            jtbd_scenarios[jtbd_name] = scenarios
        stages.append({"stage": stage_name, "jtbd_l1": jtbd_names})

    need_l1_set = {}
    need_l2_set = {}
    for needs in scenario_needs.values():
        for l1, l2 in needs:
            need_l1_set[l1] = True
            need_l2_set[(l1, l2)] = True

    return {
        "stages": stages,
        "jtbd_order": jtbd_order + ["其他"],
        "jtbd_stage_map": jtbd_stage_map,
        "jtbd_scenarios": jtbd_scenarios,
        "scenario_needs": scenario_needs,
        "need_l1_order": list(need_l1_set) + ["新主题候选", "其他"],
        "need_l2_order": list(need_l2_set),
    }


def _prepare_voc(source: pd.DataFrame) -> pd.DataFrame:
    work = source.copy()
    if "is_voc" in work.columns:
        voc = work["is_voc"].astype(str).str.lower().isin(["true", "1", "yes"])
        work = work[voc]
    work["scenario_l2"] = work.get("scenario_l2", "").fillna("其他").astype(str)
    work["need_theme_l1"] = work.get("need_theme_l1", "").fillna("其他").astype(str)
    work["need_theme_l2"] = work.get("need_theme_l2", "").fillna("其他").astype(str)
    work["raw_text"] = work.get("raw_text", "").fillna("")
    work["severity"] = pd.to_numeric(work.get("severity", 0), errors="coerce").fillna(0)
    return work


def build_journey_data(source: pd.DataFrame, def_idx: dict) -> list[dict]:
    work = source.copy()
    if "is_voc" in work.columns:
        voc = work["is_voc"].astype(str).str.lower().isin(["true", "1", "yes"])
        work = work[voc]
    work["jtbd_l1"] = work.get("jtbd_l1", "").fillna("其他").astype(str)
    work["scenario_l2"] = work.get("scenario_l2", "").fillna("其他").astype(str)
    work["need_theme_l1"] = work.get("need_theme_l1", "").fillna("其他").astype(str)
    work["need_theme_l2"] = work.get("need_theme_l2", "").fillna("其他").astype(str)

    data = []
    for stage in def_idx["stages"]:
        jtbd_items = []
        for j in stage["jtbd_l1"]:
            jsub = work[work["jtbd_l1"] == j]
            if jsub.empty:
                continue
            scenarios = []
            for sc in def_idx["jtbd_scenarios"].get(j, []):
                ssub = jsub[jsub["scenario_l2"] == sc]
                if ssub.empty:
                    continue
                l1_counts = ssub.groupby("need_theme_l1").size().to_dict()
                l2_gb = ssub.groupby(["need_theme_l1", "need_theme_l2"]).size()
                l2_list = [
                    {"l1": l1, "l2": l2, "count": int(c)}
                    for (l1, l2), c in l2_gb.items()
                ]
                l2_list.sort(key=lambda x: x["count"], reverse=True)
                scenarios.append({
                    "scenario": sc,
                    "count": len(ssub),
                    "need_l1": l1_counts,
                    "need_l2": l2_list,
                })
            scenarios.sort(key=lambda x: x["count"], reverse=True)
            jtbd_items.append({
                "jtbd": j,
                "count": len(jsub),
                "scenarios": scenarios,
            })
        jtbd_items.sort(key=lambda x: x["count"], reverse=True)
        if jtbd_items:
            data.append({"stage": stage["stage"], "items": jtbd_items})
    return data


def journey_mermaid(source: pd.DataFrame, def_idx: dict) -> str:
    data = build_journey_data(source, def_idx)
    all_counts = [item["count"] for s in data for item in s["items"]]
    max_cnt = max(all_counts) if all_counts else 1

    lines = ["flowchart LR", "  title 用户旅程反馈深度分布"]
    for si, s in enumerate(data):
        lines.append(f"  subgraph S{si}[{s['stage']}]")
        for item in s["items"]:
            jid = item["jtbd"].replace(" ", "_")
            lines.append(f"    {jid}[{item['jtbd']} {item['count']}]")
            for sc in item["scenarios"]:
                scid = f"{jid}_{sc['scenario'].replace(' ', '_')}"
                lines.append(f"    {jid} --> {scid}[{sc['scenario']} {sc['count']}]")
                for l1_name, l1_cnt in sc["need_l1"].items():
                    l1id = f"{scid}_{l1_name.replace(' ', '_')}"
                    lines.append(f"    {scid} --> {l1id}[{l1_name} {l1_cnt}]")
                    for li in sc["need_l2"]:
                        if li["l1"] != l1_name:
                            continue
                        l2id = f"{l1id}_{li['l2'].replace(' ', '_')}"
                        lines.append(f"    {l1id} --> {l2id}[{li['l2']} {li['count']}]")
        lines.append("  end")
    return "\n".join(lines)


def journey_mermaid_compact(source: pd.DataFrame, def_idx: dict) -> str:
    data = build_journey_data(source, def_idx)
    all_counts = [item["count"] for s in data for item in s["items"]]
    max_cnt = max(all_counts) if all_counts else 1

    lines = ["journey", "    title 用户旅程反馈分布"]
    for s in data:
        lines.append(f"    section {s['stage']}")
        for item in s["items"]:
            score = _score_bucket(item["count"], max_cnt)
            lines.append(f"      {item['jtbd']}({item['count']}): {score}: Issues")
    return "\n".join(lines)


def _stage_summary(source: pd.DataFrame, def_idx: dict) -> list[dict]:
    work = source.copy()
    if "is_voc" in work.columns:
        voc = work["is_voc"].astype(str).str.lower().isin(["true", "1", "yes"])
        work = work[voc]
    work["jtbd_l1"] = work.get("jtbd_l1", "").fillna("其他").astype(str)

    rows = []
    for stage in def_idx["stages"]:
        sub = work[work["jtbd_l1"].isin(stage["jtbd_l1"])]
        rows.append({
            "stage": stage["stage"],
            "jtbd_l1": " / ".join(stage["jtbd_l1"]),
            "count": len(sub),
        })
    rows.append({"stage": "TOTAL", "jtbd_l1": "", "count": len(work)})
    return rows


def _deep_dive_text(work: pd.DataFrame, def_idx: dict, stage_name: str, jtbd: str, section_num: str) -> list[str]:
    sub = work[work["jtbd_l1"] == jtbd].copy()
    if sub.empty:
        return []
    total = len(sub)

    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"  {section_num}. {stage_name} > {jtbd}")
    lines.append("=" * 60)
    lines.append(f"  总反馈: {total} 条")
    lines.append("")

    gb = sub.groupby(["scenario_l2", "need_theme_l1", "need_theme_l2"]).size().reset_index(name="cnt")
    gb = gb.sort_values("cnt", ascending=False)

    lines.append(f"  {'场景':<12} {'need_theme_l1':<16} {'need_theme_l2':<24} {'数量':>6} {'占比':>6}")
    lines.append(f"  {'-'*68}")
    for _, r in gb.iterrows():
        pct = r["cnt"] / total * 100
        lines.append(f"  {r['scenario_l2']:<12} {r['need_theme_l1']:<16} {r['need_theme_l2']:<24} {int(r['cnt']):>6} {pct:>5.1f}%")

    top5 = gb.head(5)
    lines.append("")
    lines.append("  典型原话:")
    for _, r in top5.iterrows():
        mask = (
            (sub["scenario_l2"] == r["scenario_l2"]) &
            (sub["need_theme_l1"] == r["need_theme_l1"]) &
            (sub["need_theme_l2"] == r["need_theme_l2"])
        )
        quotes = sub[mask].sort_values("severity", ascending=False).head(2)["raw_text"]
        label = f"{r['scenario_l2']} > {r['need_theme_l2']}"
        lines.append(f"    [{label}]")
        for q in quotes:
            if str(q).strip():
                lines.append(f"      -> {str(q)}")
    return lines


def _deep_dive_html(work: pd.DataFrame, def_idx: dict, stage_name: str, jtbd: str) -> str:
    sub = work[work["jtbd_l1"] == jtbd].copy()
    if sub.empty:
        return ""
    total = len(sub)

    html = f'<h2>{stage_name} > {jtbd}</h2>'
    html += f'<div class="stat-card" style="display:inline-block;margin-bottom:16px;"><div class="num">{total}</div><div class="label">总反馈</div></div>'

    gb = sub.groupby(["scenario_l2", "need_theme_l1", "need_theme_l2"]).size().reset_index(name="cnt")
    gb = gb.sort_values("cnt", ascending=False)

    html += '<table><tr><th>场景</th><th>need_theme_l1</th><th>need_theme_l2</th><th>数量</th><th>占比</th></tr>'
    for _, r in gb.iterrows():
        pct = r["cnt"] / total * 100
        html += f'<tr><td>{r["scenario_l2"]}</td><td>{r["need_theme_l1"]}</td><td>{r["need_theme_l2"]}</td><td>{int(r["cnt"])}</td><td>{pct:.1f}%</td></tr>'
    html += '</table>'

    top5 = gb.head(5)
    html += '<h3 style="margin-top:16px;">典型原话</h3>'
    for _, r in top5.iterrows():
        mask = (
            (sub["scenario_l2"] == r["scenario_l2"]) &
            (sub["need_theme_l1"] == r["need_theme_l1"]) &
            (sub["need_theme_l2"] == r["need_theme_l2"])
        )
        quotes = sub[mask].sort_values("severity", ascending=False).head(2)["raw_text"]
        label = f"{r['scenario_l2']} > {r['need_theme_l2']}"
        html += f'<div style="margin:8px 0;"><strong>{label}</strong></div>'
        for q in quotes:
            if str(q).strip():
                html += f'<div class="quote" style="margin:2px 0 2px 16px;">→ {str(q)}</div>'
    return html


def _car_series_text(work: pd.DataFrame, def_idx: dict) -> list[str]:
    lines = []
    series_col = "car_series"
    if series_col not in work.columns:
        return lines

    total = len(work)
    series_counts = work[series_col].fillna("未知").astype(str).value_counts()

    lines.append("")
    lines.append(f"  {'车型':<12} {'数量':>6} {'占比':>6}")
    lines.append(f"  {'-'*28}")
    for s, cnt in series_counts.items():
        if s == "nan" or not s.strip():
            continue
        pct = cnt / total * 100
        lines.append(f"  {s:<12} {cnt:>6} {pct:>5.1f}%")
    if "" in series_counts.index or "nan" in [str(x) for x in series_counts.index]:
        unknown = series_counts.get("", 0) + series_counts.get("nan", 0)
        pct = unknown / total * 100
        lines.append(f"  {'未知':<12} {unknown:>6} {pct:>5.1f}%")

    # Car series × Stage cross table
    work["_stage"] = work["jtbd_l1"].map(def_idx["jtbd_stage_map"]).fillna("其他阶段")
    stages_in_order = [s["stage"] for s in def_idx["stages"]]
    cross = work.groupby(["_stage", series_col]).size().unstack(fill_value=0)
    lines.append("")
    header = f"  {'车型':<10}"
    hdr_cols = []
    for s in stages_in_order:
        if s in cross.index:
            header += f" {s:<8}"
            hdr_cols.append(s)
    header += f" {'总计':>6}"
    lines.append(header)
    lines.append(f"  {'-' * (10 + 11 * len(hdr_cols) + 6)}")
    for s_name in series_counts.index[:10]:
        if s_name == "nan" or not s_name.strip():
            continue
        row = f"  {s_name:<10}"
        row_total = 0
        for s in hdr_cols:
            val = int(cross.loc[s, s_name]) if s_name in cross.columns else 0
            row += f" {val:>6}  "
            row_total += val
        row += f" {row_total:>6}"
        lines.append(row)

    # Per-car-series top themes
    lines.append("")
    lines.append("  各车型 Top 5 主题:")
    lines.append(f"  {'-'*80}")
    for s_name in series_counts.head(5).index:
        if s_name == "nan" or not s_name.strip():
            continue
        sub = work[work[series_col].astype(str) == s_name]
        top = sub.groupby("need_theme_l2").size().sort_values(ascending=False).head(5)
        themes_str = ", ".join(f"{t}({c})" for t, c in top.items())
        lines.append(f"  {s_name:<12} | {themes_str}")

    return lines


def _car_series_html(work: pd.DataFrame, def_idx: dict) -> str:
    series_col = "car_series"
    if series_col not in work.columns:
        return ""
    total = len(work)
    series_counts = work[series_col].fillna("未知").astype(str).value_counts()

    html = '<h2>分车型统计</h2>'
    html += '<table><tr><th>车型</th><th>数量</th><th>占比</th></tr>'
    for s, cnt in series_counts.items():
        if s == "nan" or not s.strip():
            continue
        pct = cnt / total * 100
        html += f'<tr><td>{s}</td><td>{cnt}</td><td>{pct:.1f}%</td></tr>'
    html += '</table>'

    # Car series × Stage cross table
    work["_stage"] = work["jtbd_l1"].map(def_idx["jtbd_stage_map"]).fillna("其他阶段")
    stages_in_order = [s["stage"] for s in def_idx["stages"]]
    cross = work.groupby(["_stage", series_col]).size().unstack(fill_value=0)

    html += '<h3 style="margin-top:16px;">车型 × 阶段交叉</h3>'
    html += '<table><tr><th>车型</th>'
    for s in stages_in_order:
        if s in cross.index:
            html += f'<th>{s}</th>'
    html += '<th>总计</th></tr>'
    for s_name in series_counts.index[:10]:
        if s_name == "nan" or not s_name.strip():
            continue
        html += f'<tr><td><strong>{s_name}</strong></td>'
        row_total = 0
        for s in stages_in_order:
            if s not in cross.index:
                continue
            val = int(cross.loc[s, s_name]) if s_name in cross.columns else 0
            html += f'<td>{val}</td>'
            row_total += val
        html += f'<td><strong>{row_total}</strong></td></tr>'
    html += '</table>'

    # Per-car-series top themes
    html += '<h3 style="margin-top:16px;">各车型 Top 5 主题</h3>'
    for s_name in series_counts.head(5).index:
        if s_name == "nan" or not s_name.strip():
            continue
        sub = work[work[series_col].astype(str) == s_name]
        top = sub.groupby("need_theme_l2").size().sort_values(ascending=False).head(5)
        html += f'<div style="margin:8px 0;"><strong>{s_name}</strong></div>'
        html += '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-left:12px;">'
        for t, c in top.items():
            html += f'<span style="background:#f0eeff;padding:2px 10px;border-radius:4px;font-size:12px;">{t}({c})</span>'
        html += '</div>'

    return html


def _business_deep_dive_text(work: pd.DataFrame, business_def: dict) -> list[str]:
    def_idx = parse_definition(business_def)
    target_l2s = []
    seen = set()
    for stage in def_idx["stages"]:
        for jtbd in stage["jtbd_l1"]:
            for sc in def_idx["jtbd_scenarios"].get(jtbd, []):
                for l1, l2 in def_idx["scenario_needs"].get((jtbd, sc), []):
                    if l2 not in seen:
                        target_l2s.append(l2)
                        seen.add(l2)

    lines = []
    for l2 in target_l2s:
        sub = work[work["need_theme_l2"] == l2].copy()
        if sub.empty:
            continue
        lines.append("")
        lines.append("-" * 60)
        lines.append(f"  [{l2}]  共 {len(sub)} 条反馈")
        lines.append("-" * 60)
        for idx, (_, r) in enumerate(sub.iterrows(), 1):
            series = str(r.get("car_series", "")).strip()
            text = str(r.get("raw_text", "")).strip()
            lines.append(f"  {idx:>3}. [{series}] {text}")
    return lines


def _business_deep_dive_html(work: pd.DataFrame, business_def: dict) -> str:
    def_idx = parse_definition(business_def)
    target_l2s = []
    seen = set()
    for stage in def_idx["stages"]:
        for jtbd in stage["jtbd_l1"]:
            for sc in def_idx["jtbd_scenarios"].get(jtbd, []):
                for l1, l2 in def_idx["scenario_needs"].get((jtbd, sc), []):
                    if l2 not in seen:
                        target_l2s.append(l2)
                        seen.add(l2)

    html = '<h2>部门重点主题 — 全量明细</h2>'
    for l2 in target_l2s:
        sub = work[work["need_theme_l2"] == l2].copy()
        if sub.empty:
            continue
        html += f'<h3 style="margin-top:20px;">{l2} <span style="font-weight:400;color:#888;">(共 {len(sub)} 条)</span></h3>'
        html += '<table><tr><th>#</th><th>车系</th><th>反馈原文</th></tr>'
        for idx, (_, r) in enumerate(sub.iterrows(), 1):
            series = str(r.get("car_series", "")).strip()
            text = str(r.get("raw_text", "")).strip()
            html += f'<tr><td>{idx}</td><td>{series}</td><td class="quote">{text}</td></tr>'
        html += '</table>'
    return html


def report(source: pd.DataFrame, theme: pd.DataFrame, jtbd: pd.DataFrame, definition: dict) -> str:
    def_idx = parse_definition(definition)
    _clean_numeric(theme, ["issue_count", "priority_score", "avg_severity", "avg_sentiment"])
    _clean_numeric(jtbd, ["issue_count", "priority_score", "avg_severity", "avg_sentiment"])

    lines = []

    def h1(title: str):
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"  {title}")
        lines.append("=" * 60)

    total_issues = int(theme["issue_count"].sum())
    total_source = len(source)

    h1("VOC Analysis Report")
    lines.append(f"  Source issues:    {total_source}")
    lines.append(f"  Theme groups:     {len(theme)}")
    lines.append(f"  JTBD groups:      {len(jtbd)}")
    lines.append(f"  Total VOC issues: {total_issues}")

    h1("0. TOP 10 最高优先级改进点")
    top10 = theme.sort_values("priority_score", ascending=False).head(10).copy()
    for i, (_, r) in enumerate(top10.iterrows(), 1):
        lines.append(f"  {i:>2}. [{r['priority_score']:>6.1f}] {r['need_theme_l1']} > {r['need_theme_l2']}  (反馈量={int(r['issue_count'])}, 严重度={r['avg_severity']})")

    h1("1. Journey Stages")
    lines.append(f"  {'Stage':<12} {'JTBD L1':<20} {'Issues':>8} {'%':>6}")
    lines.append(f"  {'-'*48}")
    stage_rows = _stage_summary(source, def_idx)
    for r in stage_rows:
        pct = r["count"] / total_source * 100 if total_source else 0
        pct_str = f"{pct:.1f}%" if r["stage"] != "TOTAL" else "100.0%"
        lines.append(f"  {r['stage']:<12} {r['jtbd_l1']:<20} {r['count']:>8} {pct_str:>6}")

    work = _prepare_voc(source)
    sec = 2
    for stage in def_idx["stages"]:
        for jtbd_name in stage["jtbd_l1"]:
            lines.extend(_deep_dive_text(work, def_idx, stage["stage"], jtbd_name, str(sec)))
            sec += 1

    cs_lines = _car_series_text(work, def_idx)
    if cs_lines:
        h1(f"{sec}. 分车型统计")
        lines.extend(cs_lines)
        sec += 1

    h1(f"{sec}. User Journey 图")
    lines.append("```mermaid")
    lines.append(journey_mermaid_compact(source, def_idx))
    lines.append("```")

    return "\n".join(lines)


def report_with_business(source: pd.DataFrame, theme: pd.DataFrame, jtbd: pd.DataFrame,
                          definition: dict, business_definition: dict) -> str:
    base = report(source, theme, jtbd, definition)
    work = _prepare_voc(source)
    biz_lines = _business_deep_dive_text(work, business_definition)
    if biz_lines:
        extra = "\n".join(["\n", "=" * 60,
                           "  部门重点主题 — 全量明细",
                           "=" * 60] + biz_lines)
        return base + extra
    return base


def report_html(source: pd.DataFrame, theme: pd.DataFrame, jtbd: pd.DataFrame, definition: dict,
                business_definition: dict | None = None) -> str:
    def_idx = parse_definition(definition)
    _clean_numeric(theme, ["issue_count", "priority_score", "avg_severity", "avg_sentiment"])
    _clean_numeric(jtbd, ["issue_count", "priority_score", "avg_severity", "avg_sentiment"])

    total_issues = int(theme["issue_count"].sum())
    total_source = len(source)

    mermaid_code = journey_mermaid_compact(source, def_idx)

    colors = ["#E3F2FD", "#E8F5E9", "#FFF3E0", "#FCE4EC"]

    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#f5f7fa; color:#333; padding:32px; }
h1 { font-size:24px; margin:28px 0 12px; }
h2 { font-size:18px; margin:20px 0 8px; color:#555; }
.stats { display:flex; gap:16px; margin-bottom:24px; flex-wrap:wrap; }
.stat-card { background:#fff; border-radius:10px; padding:16px 24px; box-shadow:0 1px 3px rgba(0,0,0,0.08); }
.stat-card .num { font-size:28px; font-weight:700; color:#0f3460; }
.stat-card .label { font-size:13px; color:#888; }
.stage-card { background:#fff; border-radius:10px; padding:12px 16px; box-shadow:0 1px 3px rgba(0,0,0,0.08); display:inline-flex; align-items:center; gap:12px; }
.stage-card .tag { display:inline-block; padding:2px 10px; border-radius:4px; font-size:12px; font-weight:600; }
.stage-card .num { font-size:20px; font-weight:700; color:#0f3460; }
.stage-card .label { font-size:12px; color:#888; }
table { border-collapse:collapse; width:100%; margin:8px 0 16px; background:#fff; box-shadow:0 1px 3px rgba(0,0,0,0.08); border-radius:8px; overflow:hidden; font-size:13px; }
th { background:#1a1a2e; color:#fff; padding:10px 12px; text-align:left; font-weight:500; }
td { padding:8px 12px; border-bottom:1px solid #eee; }
tr:hover td { background:#f0f4ff; }
.quote { color:#555; font-size:13px; line-height:1.5; white-space:pre-wrap; word-break:break-word; }
.mermaid-wrapper { background:#fff; border:1px solid #dee2e6; border-radius:8px; padding:24px; overflow-x:auto; }
</style>
</head>
<body>
<h1>VOC Analysis Report</h1>
<div class="stats">
  <div class="stat-card"><div class="num">""" + str(total_issues) + """</div><div class="label">总反馈量</div></div>
  <div class="stat-card"><div class="num">""" + str(len(theme)) + """</div><div class="label">主题数</div></div>
  <div class="stat-card"><div class="num">""" + str(len(jtbd)) + """</div><div class="label">场景数</div></div>
  <div class="stat-card"><div class="num">""" + str(total_source) + """</div><div class="label">原始 issues</div></div>
</div>"""

    top10 = theme.sort_values("priority_score", ascending=False).head(10)
    html += '<h2>TOP 10 最高优先级改进点</h2>'
    html += '<table><tr><th>#</th><th>优先级</th><th>主题</th><th>反馈量</th><th>严重度</th></tr>'
    for i, (_, r) in enumerate(top10.iterrows(), 1):
        html += f'<tr><td>{i}</td><td>{r["priority_score"]:.1f}</td><td>{r["need_theme_l1"]} > {r["need_theme_l2"]}</td><td>{int(r["issue_count"])}</td><td>{r["avg_severity"]}</td></tr>'
    html += '</table>'

    stage_rows = _stage_summary(source, def_idx)
    html += '<div class="stats">'
    for i, r in enumerate(stage_rows):
        if r["stage"] == "TOTAL":
            html += f'<div class="stage-card"><span class="tag" style="background:#eee">TOTAL</span><div><div class="num">{r["count"]}</div><div class="label">全部</div></div></div>'
        else:
            color = colors[i % len(colors)]
            html += f'<div class="stage-card"><span class="tag" style="background:{color}">{r["stage"]}</span><div><div class="num">{r["count"]}</div><div class="label">{r["jtbd_l1"]}</div></div></div>'
    html += "</div>"

    work = _prepare_voc(source)
    for stage in def_idx["stages"]:
        for jtbd_name in stage["jtbd_l1"]:
            html += _deep_dive_html(work, def_idx, stage["stage"], jtbd_name)

    html += _car_series_html(work, def_idx)

    html += f"""
<h2>用户旅程图</h2>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<div class="mermaid-wrapper">
<div class="mermaid">
{mermaid_code}
</div>
</div>
"""

    if business_definition is not None:
        work = _prepare_voc(source)
        html += _business_deep_dive_html(work, business_definition)

    html += "\n</body>\n</html>"
    return html


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme", default="theme_fact_full_v3.csv")
    parser.add_argument("--jtbd", default="jtbd_fact_full_v3.csv")
    parser.add_argument("--source", default="issue_fact_refined_full_v3.csv")
    parser.add_argument("--definition", default="journey_definition.json")
    parser.add_argument("--business", default="business_definition.json")
    parser.add_argument("--other", default="need_theme_other_issues_full_v3.csv")
    parser.add_argument("--output", default="voc_analysis_report.html")
    parser.add_argument("--no-html", action="store_true", help="Only print to terminal, skip HTML")
    args = parser.parse_args()

    theme = load_csv(args.theme)
    jtbd = load_csv(args.jtbd)
    source = load_csv(args.source)
    definition = load_definition(args.definition)

    business_definition = None
    business_path = Path(args.business)
    if business_path.exists():
        business_definition = load_definition(business_path)

    other = None
    other_path = Path(args.other)
    if other_path.exists():
        other = load_csv(other_path)

    if business_definition is not None:
        print(report_with_business(source, theme, jtbd, definition, business_definition))
    else:
        print(report(source, theme, jtbd, definition))

    if other is not None:
        print("")
        print("=" * 60)
        print("  未分类问题 (need_theme_other)")
        print("=" * 60)
        print(f"  共 {len(other)} 条")
        print("")
        for i, (_, r) in enumerate(other.iterrows(), 1):
            txt = str(r.get("raw_text", "")).strip()
            if txt:
                print(f"  [{i}] {txt}")
                print()

    if not args.no_html:
        html = report_html(source, theme, jtbd, definition, business_definition)

        if other is not None:
            oh = '<h2>未分类问题 (need_theme_other)</h2>'
            oh += f'<p>共 {len(other)} 条</p>'
            oh += '<table><tr><th>#</th><th>raw_text</th></tr>'
            for i, (_, r) in enumerate(other.iterrows(), 1):
                txt = str(r.get("raw_text", "")).strip()
                if txt:
                    oh += f'<tr><td>{i}</td><td class="quote">{txt}</td></tr>'
            oh += '</table>'
            html = html.replace('</body>', oh + '\n</body>')

        Path(args.output).write_text(html, encoding="utf-8")
        print(f"\nHTML report saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
