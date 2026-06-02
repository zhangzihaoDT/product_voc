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


STAGES = [
    {"stage": "购车阶段", "jtbd_l1": ["交付"]},
    {"stage": "数字阶段", "jtbd_l1": ["数字资产管理"]},
    {"stage": "用车阶段", "jtbd_l1": ["用车准备", "进入车辆", "驾驶", "智驾", "停车", "驻车体验", "娱乐", "补能"]},
    {"stage": "服务阶段", "jtbd_l1": ["售后服务"]},
]

JTBD_ORDER = [
    "交付", "数字资产管理", "用车准备", "进入车辆", "驾驶",
    "智驾", "停车", "驻车体验", "娱乐", "补能", "售后服务", "其他",
]

NEED_THEME_ORDER = [
    "交付体验", "APP体验", "账号与权益", "外观设计", "车联车控",
    "智驾体验", "车辆品质", "智舱体验", "灯光体验", "内容生态",
    "补能", "服务体验", "新主题候选", "其他",
]


def build_journey_data(source: pd.DataFrame, jtbd_scenarios: dict) -> list[dict]:
    work = source.copy()
    if "is_voc" in work.columns:
        voc = work["is_voc"].astype(str).str.lower().isin(["true", "1", "yes"])
        work = work[voc]
    work["jtbd_l1"] = work.get("jtbd_l1", "").fillna("其他").astype(str)

    data = []
    for stage in STAGES:
        jtbd_items = []
        for j in stage["jtbd_l1"]:
            if j not in jtbd_scenarios:
                continue
            sub = work[work["jtbd_l1"] == j]
            cnt = len(sub)
            scenarios = []
            for sc in jtbd_scenarios[j]:
                sc_cnt = int((sub["scenario_l2"].fillna("其他").astype(str) == sc).sum()) if "scenario_l2" in sub.columns else 0
                scenarios.append({"scenario": sc, "count": sc_cnt})
            jtbd_items.append({"jtbd": j, "count": cnt, "scenarios": scenarios})
        if jtbd_items:
            data.append({"stage": stage["stage"], "items": jtbd_items})
    return data


def journey_mermaid(source: pd.DataFrame, definition: dict) -> str:
    data = build_journey_data(source, definition.get("jtbd_scenarios", {}))
    all_counts = [item["count"] for s in data for item in s["items"]]
    max_cnt = max(all_counts) if all_counts else 1

    lines = ["flowchart LR", "  title 用户旅程反馈分布"]
    for si, s in enumerate(data):
        sid = f"S{si}"
        lines.append(f"  subgraph {sid}[{s['stage']}]")
        for item in s["items"]:
            score = _score_bucket(item["count"], max_cnt)
            lines.append(f"    {item['jtbd']}[{item['jtbd']} {item['count']}]")
            for sc in item["scenarios"]:
                sc_score = _score_bucket(sc["count"], max_cnt) if sc["count"] else 0
                if sc["count"] > 0:
                    lines.append(f"    {item['jtbd']} --> {item['jtbd']}_{sc['scenario']}[{sc['scenario']} {sc['count']}]")
        lines.append("  end")
    return "\n".join(lines)


def journey_mermaid_compact(source: pd.DataFrame, definition: dict) -> str:
    data = build_journey_data(source, definition.get("jtbd_scenarios", {}))
    all_counts = [item["count"] for s in data for item in s["items"]]
    max_cnt = max(all_counts) if all_counts else 1

    lines = ["journey", "    title 用户旅程反馈分布"]
    for s in data:
        lines.append(f"    section {s['stage']}")
        for item in s["items"]:
            score = _score_bucket(item["count"], max_cnt)
            lines.append(f"      {item['jtbd']}({item['count']}): {score}: Issues")
    return "\n".join(lines)


def _stage_summary(source: pd.DataFrame) -> list[dict]:
    work = source.copy()
    if "is_voc" in work.columns:
        voc = work["is_voc"].astype(str).str.lower().isin(["true", "1", "yes"])
        work = work[voc]
    work["jtbd_l1"] = work.get("jtbd_l1", "").fillna("其他").astype(str)

    rows = []
    for stage in STAGES:
        sub = work[work["jtbd_l1"].isin(stage["jtbd_l1"])]
        rows.append({
            "stage": stage["stage"],
            "jtbd_l1": " / ".join(stage["jtbd_l1"]),
            "count": len(sub),
        })
    rows.append({"stage": "TOTAL", "jtbd_l1": "", "count": len(work)})
    return rows


def _prepare_voc(source: pd.DataFrame) -> pd.DataFrame:
    work = source.copy()
    if "is_voc" in work.columns:
        voc = work["is_voc"].astype(str).str.lower().isin(["true", "1", "yes"])
        work = work[voc]
    work["scenario_l2"] = work.get("scenario_l2", "").fillna("其他").astype(str)
    work["raw_text"] = work.get("raw_text", "").fillna("")
    work["severity"] = pd.to_numeric(work.get("severity", 0), errors="coerce").fillna(0)
    return work


def _deep_dive_text(work: pd.DataFrame, stage_name: str, jtbd: str, section_num: str) -> list[str]:
    lines = []
    sub = work[work["jtbd_l1"] == jtbd].copy()
    if sub.empty:
        return lines
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"  {section_num}. {stage_name} > {jtbd}")
    lines.append("=" * 60)
    lines.append(f"  总反馈: {len(sub)} 条")
    lines.append("")
    lines.append(f"  {'场景':<12} {'数量':>6} {'占比':>6}  {'涉及主题'}")
    lines.append(f"  {'-'*65}")
    sc_counts = sub["scenario_l2"].value_counts()
    for sc, cnt in sc_counts.items():
        pct = cnt / len(sub) * 100
        themes = sub[sub["scenario_l2"] == sc]["need_theme_l2"].dropna().unique()
        lines.append(f"  {sc:<12} {cnt:>6} {pct:>5.1f}%  {' / '.join(str(t) for t in themes[:3])}")
    lines.append("")
    lines.append("  典型原话:")
    for sc in sc_counts.index[:5]:
        sub_sc = sub[sub["scenario_l2"] == sc]
        quotes = sub_sc.sort_values("severity", ascending=False).head(2)["raw_text"]
        lines.append(f"    [{sc}]")
        for q in quotes:
            if str(q).strip():
                lines.append(f"      -> {str(q)}")
    return lines


def _deep_dive_html(work: pd.DataFrame, stage_name: str, jtbd: str) -> str:
    sub = work[work["jtbd_l1"] == jtbd].copy()
    if sub.empty:
        return ""
    html = f'<h2>{stage_name} > {jtbd}</h2>'
    html += f'<div class="stat-card" style="display:inline-block;margin-bottom:16px;"><div class="num">{len(sub)}</div><div class="label">总反馈</div></div>'
    sc_counts = sub["scenario_l2"].value_counts()
    html += '<table><tr><th>场景</th><th>数量</th><th>占比</th><th>涉及主题</th></tr>'
    for sc, cnt in sc_counts.items():
        pct = cnt / len(sub) * 100
        themes = sub[sub["scenario_l2"] == sc]["need_theme_l2"].dropna().unique()
        html += f'<tr><td>{sc}</td><td>{cnt}</td><td>{pct:.1f}%</td><td>{" / ".join(str(t) for t in themes[:3])}</td></tr>'
    html += '</table>'
    html += '<h3 style="margin-top:16px;">典型原话</h3>'
    for sc in sc_counts.index[:5]:
        sub_sc = sub[sub["scenario_l2"] == sc]
        quotes = sub_sc.sort_values("severity", ascending=False).head(2)["raw_text"]
        html += f'<div style="margin:8px 0;"><strong>{sc}</strong></div>'
        for q in quotes:
            if str(q).strip():
                html += f'<div class="quote" style="margin:2px 0 2px 16px;">→ {str(q)}</div>'
    return html


def report(source: pd.DataFrame, theme: pd.DataFrame, jtbd: pd.DataFrame, definition: dict) -> str:
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

    # ── 0. Overview ──
    h1("VOC Analysis Report")
    lines.append(f"  Source issues:    {total_source}")
    lines.append(f"  Theme groups:     {len(theme)}")
    lines.append(f"  JTBD groups:      {len(jtbd)}")
    lines.append(f"  Total VOC issues: {total_issues}")

    # ── 1. Top 10 Priority ──
    h1("0. TOP 10 最高优先级改进点")
    top10 = theme.sort_values("priority_score", ascending=False).head(10).copy()
    for i, (_, r) in enumerate(top10.iterrows(), 1):
        lines.append(f"  {i:>2}. [{r['priority_score']:>6.1f}] {r['need_theme_l1']} > {r['need_theme_l2']}  (反馈量={int(r['issue_count'])}, 严重度={r['avg_severity']})")

    # ── 2. Stages summary ──
    h1("1. Journey Stages")
    lines.append(f"  {'Stage':<12} {'JTBD L1':<20} {'Issues':>8} {'%':>6}")
    lines.append(f"  {'-'*48}")
    stage_rows = _stage_summary(source)
    for r in stage_rows:
        pct = r["count"] / total_source * 100 if total_source else 0
        pct_str = f"{pct:.1f}%" if r["stage"] != "TOTAL" else "100.0%"
        lines.append(f"  {r['stage']:<12} {r['jtbd_l1']:<20} {r['count']:>8} {pct_str:>6}")

    # ── 2-n. Stage Deep Dives ──
    work = _prepare_voc(source)
    sec = 2
    for stage in STAGES:
        for jtbd_name in stage["jtbd_l1"]:
            lines.extend(_deep_dive_text(work, stage["stage"], jtbd_name, str(sec)))
            sec += 1

    # ── n+1. Mermaid Journey ──
    h1(f"{sec}. User Journey 图")
    lines.append("```mermaid")
    lines.append(journey_mermaid_compact(source, definition))
    lines.append("```")

    return "\n".join(lines)


def report_html(source: pd.DataFrame, theme: pd.DataFrame, jtbd: pd.DataFrame, definition: dict) -> str:
    _clean_numeric(theme, ["issue_count", "priority_score", "avg_severity", "avg_sentiment"])
    _clean_numeric(jtbd, ["issue_count", "priority_score", "avg_severity", "avg_sentiment"])

    total_issues = int(theme["issue_count"].sum())
    total_source = len(source)

    mermaid_code = journey_mermaid_compact(source, definition)

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

    # ── Top 10 Priority ──
    top10 = theme.sort_values("priority_score", ascending=False).head(10)
    html += '<h2>TOP 10 最高优先级改进点</h2>'
    html += '<table><tr><th>#</th><th>优先级</th><th>主题</th><th>反馈量</th><th>严重度</th></tr>'
    for i, (_, r) in enumerate(top10.iterrows(), 1):
        html += f'<tr><td>{i}</td><td>{r["priority_score"]:.1f}</td><td>{r["need_theme_l1"]} > {r["need_theme_l2"]}</td><td>{int(r["issue_count"])}</td><td>{r["avg_severity"]}</td></tr>'
    html += '</table>'

    # ── Stages summary ──
    stage_rows = _stage_summary(source)
    html += '<div class="stats">'
    for i, r in enumerate(stage_rows):
        if r["stage"] == "TOTAL":
            html += f'<div class="stage-card"><span class="tag" style="background:#eee">TOTAL</span><div><div class="num">{r["count"]}</div><div class="label">全部</div></div></div>'
        else:
            color = colors[i % len(colors)]
            html += f'<div class="stage-card"><span class="tag" style="background:{color}">{r["stage"]}</span><div><div class="num">{r["count"]}</div><div class="label">{r["jtbd_l1"]}</div></div></div>'
    html += "</div>"

    # ── Stage Deep Dives ──
    work = _prepare_voc(source)
    for stage in STAGES:
        for jtbd_name in stage["jtbd_l1"]:
            html += _deep_dive_html(work, stage["stage"], jtbd_name)

    # ── Mermaid Journey ──
    html += f"""
<h2>用户旅程图</h2>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<div class="mermaid-wrapper">
<div class="mermaid">
{mermaid_code}
</div>
</div>
</body>
</html>"""

    return html


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theme", default="theme_fact_full_v3.csv")
    parser.add_argument("--jtbd", default="jtbd_fact_full_v3.csv")
    parser.add_argument("--source", default="issue_fact_refined_full_v3.csv")
    parser.add_argument("--definition", default="journey_definition.json")
    parser.add_argument("--other", default="need_theme_other_issues_full_v3.csv")
    parser.add_argument("--output", default="voc_analysis_report.html")
    parser.add_argument("--no-html", action="store_true", help="Only print to terminal, skip HTML")
    args = parser.parse_args()

    theme = load_csv(args.theme)
    jtbd = load_csv(args.jtbd)
    source = load_csv(args.source)
    definition = load_definition(args.definition)

    other = None
    other_path = Path(args.other)
    if other_path.exists():
        other = load_csv(other_path)

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
        html = report_html(source, theme, jtbd, definition)

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
