import pandas as pd
import argparse
from pathlib import Path

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

DEFAULT_THEME_CSV = "theme_fact_full.csv"
DEFAULT_OTHER_CSV = "need_theme_other_issues_full.csv"
DEFAULT_OUTPUT_HTML = "voc_dashboard.html"

parser = argparse.ArgumentParser()
parser.add_argument("--theme-csv", default=DEFAULT_THEME_CSV)
parser.add_argument("--other-issues-csv", default=DEFAULT_OTHER_CSV)
parser.add_argument("--output", default=DEFAULT_OUTPUT_HTML)
args = parser.parse_args()

theme_csv_path = Path(args.theme_csv)
other_csv_path = Path(args.other_issues_csv) if args.other_issues_csv else None
output_path = Path(args.output)

df = pd.read_csv(theme_csv_path, encoding="utf-8-sig")

df.columns = df.columns.str.strip()
for col in ["issue_count", "app_count", "phone400_count"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
for col in ["avg_severity", "avg_sentiment", "priority_score"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

total_themes = len(df)
total_issues = int(df["issue_count"].sum())
avg_priority = round(df["priority_score"].mean(), 1)
top_l1 = df.groupby("need_theme_l1")["issue_count"].sum().idxmax()
other_issues_count = None
if other_csv_path is not None and other_csv_path.exists():
    other_df = pd.read_csv(other_csv_path, dtype="object", encoding="utf-8-sig")
    other_issues_count = int(len(other_df))

if px is None or go is None:
    top20 = df.sort_values(["priority_score", "issue_count"], ascending=[False, False]).head(20).copy()
    top20 = top20[["need_theme_l1", "need_theme_l2", "issue_count", "avg_severity", "avg_sentiment", "priority_score", "sample_quotes"]]
    top20["avg_severity"] = top20["avg_severity"].round(2)
    top20["avg_sentiment"] = top20["avg_sentiment"].round(2)
    top20["priority_score"] = top20["priority_score"].round(2)

    l1_sum = df.groupby("need_theme_l1", as_index=False)["issue_count"].sum().sort_values("issue_count", ascending=False)
    html = f"""<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>VOC Dashboard</title></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 24px;">
<h1 style="margin: 0 0 8px 0;">VOC Dashboard</h1>
<div style="color:#666; margin-bottom:16px;">基于 {theme_csv_path.name}（plotly 未安装，使用简化版）</div>
<ul>
  <li>主题数：{total_themes}</li>
  <li>总反馈量：{total_issues}</li>
  <li>平均优先级：{avg_priority}</li>
  <li>最多反馈L1：{top_l1}</li>
  {f'<li>need_theme=其他 明细数：{other_issues_count}</li>' if other_issues_count is not None else ''}
</ul>
<h2>L1 汇总</h2>
{l1_sum.to_html(index=False)}
<h2>Top 20 优先级主题</h2>
{top20.to_html(index=False)}
</body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard saved to {output_path}")
    raise SystemExit(0)

l1_agg = df.groupby("need_theme_l1").agg(
    issue_count=("issue_count", "sum"),
    theme_count=("theme_id", "count"),
    avg_priority=("priority_score", "mean"),
).reset_index().sort_values("issue_count", ascending=True)

fig_bar = px.bar(
    l1_agg,
    x="issue_count",
    y="need_theme_l1",
    orientation="h",
    text="issue_count",
    title="各L1分类反馈量分布",
    labels={"issue_count": "反馈量", "need_theme_l1": "L1分类"},
    color="issue_count",
    color_continuous_scale="Blues",
)
fig_bar.update_traces(textposition="outside")
fig_bar.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))

top15 = df.nlargest(15, "priority_score")
fig_priority = px.bar(
    top15,
    x="priority_score",
    y="need_theme_l2",
    orientation="h",
    text="priority_score",
    title="Top 15 高优先级主题",
    labels={"priority_score": "优先级得分", "need_theme_l2": "L2主题"},
    color="priority_score",
    color_continuous_scale="Reds",
)
fig_priority.update_traces(textposition="outside")
fig_priority.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))

l1_pie = df.groupby("need_theme_l1")["issue_count"].sum().reset_index()
fig_pie = px.pie(
    l1_pie,
    values="issue_count",
    names="need_theme_l1",
    title="L1 反馈量占比",
    hole=0.4,
)
fig_pie.update_traces(textposition="inside", textinfo="percent+label")
fig_pie.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10))

fig_scatter = px.scatter(
    df,
    x="avg_sentiment",
    y="avg_severity",
    size="issue_count",
    color="need_theme_l1",
    hover_name="need_theme_l2",
    hover_data={
        "issue_count": True,
        "priority_score": True,
        "avg_sentiment": ":.2f",
        "avg_severity": ":.2f",
        "need_theme_l1": True,
    },
    title="情感 vs 严重度 散点图（气泡大小 = 反馈量）",
    labels={"avg_sentiment": "平均情感得分", "avg_severity": "平均严重度"},
)
fig_scatter.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))

fig_channels = go.Figure()
channels_df = df.sort_values(["issue_count", "priority_score"], ascending=[False, False]).head(30).copy()
fig_channels.add_trace(go.Bar(
    name="APP反馈",
    x=channels_df["need_theme_l2"],
    y=channels_df["app_count"],
    marker_color="rgb(55, 126, 184)",
))
fig_channels.add_trace(go.Bar(
    name="400电话",
    x=channels_df["need_theme_l2"],
    y=channels_df["phone400_count"],
    marker_color="rgb(228, 26, 28)",
))
fig_channels.update_layout(
    title="各主题渠道分布（APP vs 400电话，Top 30 by 反馈量）",
    xaxis_tickangle=-45,
    height=500,
    barmode="group",
    margin=dict(l=10, r=10, t=40, b=120),
)

table_data = df.nlargest(20, "priority_score")[
    ["need_theme_l1", "need_theme_l2", "issue_count", "avg_severity", "avg_sentiment", "priority_score"]
].copy()
table_data["avg_severity"] = table_data["avg_severity"].round(2)
table_data["avg_sentiment"] = table_data["avg_sentiment"].round(2)
table_data["priority_score"] = table_data["priority_score"].round(1)

fig_table = go.Figure(data=[go.Table(
    header=dict(
        values=["L1分类", "L2主题", "反馈量", "严重度", "情感", "优先级"],
        fill_color="paleturquoise",
        align="center",
        font=dict(size=13),
    ),
    cells=dict(
        values=[
            table_data["need_theme_l1"],
            table_data["need_theme_l2"],
            table_data["issue_count"],
            table_data["avg_severity"],
            table_data["avg_sentiment"],
            table_data["priority_score"],
        ],
        align="center",
        font=dict(size=12),
        height=28,
    ),
)])
fig_table.update_layout(title="Top 20 优先级主题明细", height=500, margin=dict(l=10, r=10, t=40, b=10))

html_parts = [f"""
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VOC Dashboard</title>
<script src="https://cdn.plot.ly/plotly-3.0.1.min.js" charset="utf-8"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: #333; }}
.header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); color: white; padding: 32px 40px; }}
.header h1 {{ font-size: 28px; font-weight: 600; }}
.header p {{ font-size: 14px; opacity: 0.8; margin-top: 6px; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; padding: 24px 40px; }}
.stat-card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.stat-card .num {{ font-size: 32px; font-weight: 700; color: #0f3460; }}
.stat-card .label {{ font-size: 13px; color: #888; margin-top: 4px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 0 40px 20px; }}
.full {{ grid-column: 1 / -1; }}
.chart-card {{ background: white; border-radius: 12px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.footer {{ text-align: center; padding: 20px; font-size: 12px; color: #aaa; }}
</style>
</head>
<body>
<div class="header">
<h1>VOC Dashboard</h1>
<p>基于 {theme_csv_path.name} 的反馈主题分析</p>
</div>
<div class="stats">
<div class="stat-card"><div class="num">{total_themes}</div><div class="label">主题数</div></div>
<div class="stat-card"><div class="num">{total_issues}</div><div class="label">总反馈量</div></div>
<div class="stat-card"><div class="num">{avg_priority}</div><div class="label">平均优先级</div></div>
<div class="stat-card"><div class="num">{top_l1}</div><div class="label">最多反馈L1</div></div>
{f'<div class="stat-card"><div class="num">{other_issues_count}</div><div class="label">need_theme=其他 明细数</div></div>' if other_issues_count is not None else ''}
</div>
<div class="grid">
<div class="chart-card">{fig_bar.to_html(full_html=False, include_plotlyjs=False)}</div>
<div class="chart-card">{fig_pie.to_html(full_html=False, include_plotlyjs=False)}</div>
<div class="chart-card full">{fig_priority.to_html(full_html=False, include_plotlyjs=False)}</div>
<div class="chart-card full">{fig_scatter.to_html(full_html=False, include_plotlyjs=False)}</div>
<div class="chart-card full">{fig_channels.to_html(full_html=False, include_plotlyjs=False)}</div>
<div class="chart-card full">{fig_table.to_html(full_html=False, include_plotlyjs=False)}</div>
</div>
<div class="footer">Generated by VOC Dashboard | {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</div>
</body>
</html>
"""]

with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_parts[0])

print(f"Dashboard saved to {output_path}")
