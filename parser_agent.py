import argparse
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


FEEDBACK_FACT_COLUMNS = [
    "feedback_id",
    "source",
    "channel",
    "user_id",
    "car_series",
    "vehicle_version",
    "feedback_text",
    "domain",
    "module",
    "scenario",
    "problem",
    "suggestion",
    "emotion_score",
    "frequency_score",
    "impact_score",
    "priority_score",
    "created_time",
]

ISSUE_FACT_COLUMNS = [
    "feedback_id",
    "issue_no",
    "source",
    "channel",
    "user_id",
    "car_series",
    "vehicle_version",
    "created_time",
    "raw_text",
    "issue_type",
    "domain",
    "module",
    "scenario",
    "problem",
    "need",
    "suggestion",
    "sentiment",
    "severity",
    "emotion_score",
    "impact_score",
    "frequency_score",
    "priority_score",
]


@dataclass(frozen=True)
class RowContext:
    source: str
    channel: str | None
    feedback_id: str | None


def _to_str(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text


def _build_feedback_id(ctx: RowContext, row_index: int) -> str:
    if ctx.feedback_id:
        return ctx.feedback_id
    base = f"{ctx.source}|{ctx.channel or ''}|{row_index}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]


def _extract_feedback_text(overview: str, remark: str) -> str:
    overview = _to_str(overview)
    remark = _to_str(remark)
    if overview and remark and remark not in overview:
        return f"{overview}\n{remark}".strip()
    return (overview or remark).strip()


def _clean_issue_text(text: str) -> str:
    t = _to_str(text)
    t = re.sub(r"\r\n", "\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    t = re.sub(r"^\s*(用户反馈内容|问题描述|内容)\s*[:：]\s*", "", t).strip()
    return t


def _split_issues(feedback_text: str) -> list[str]:
    t = _clean_issue_text(feedback_text)
    if not t:
        return []

    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    numbered = []
    for ln in lines:
        if re.match(r"^\s*(\d{1,2}\s*[).、）]|[（(]?\d{1,2}[)）])\s*", ln):
            numbered.append(re.sub(r"^\s*(\d{1,2}\s*[).、）]|[（(]?\d{1,2}[)）])\s*", "", ln).strip())
        elif re.match(r"^\s*[一二三四五六七八九十]\s*[、.．]\s*", ln):
            numbered.append(re.sub(r"^\s*[一二三四五六七八九十]\s*[、.．]\s*", "", ln).strip())

    if len(numbered) >= 2:
        return [x for x in numbered if x]

    parts = re.split(r"\n\s*(?:此外|另外|同时|并且|再者|其次)\s*", t)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        return parts

    if len(t) > 160:
        sent = re.split(r"[。！？!?]\s*", t)
        sent = [s.strip() for s in sent if s.strip()]
        if len(sent) >= 2:
            merged = []
            buf = ""
            for s in sent:
                if not buf:
                    buf = s
                elif len(buf) < 35:
                    buf = f"{buf}。{s}".strip("。")
                else:
                    merged.append(buf)
                    buf = s
            if buf:
                merged.append(buf)
            merged = [m.strip() for m in merged if m.strip()]
            if len(merged) >= 2:
                return merged

    return [t]



def _extract_problem_suggestion(text: str) -> tuple[str, str]:
    t = re.sub(r"\s+", " ", _to_str(text)).strip()
    if not t:
        return "", ""

    suggestion_triggers = [
        "建议",
        "希望",
        "期望",
        "能否",
        "能不能",
        "尽快",
        "优化",
        "增加",
        "支持",
        "实现",
        "提供",
        "更新",
    ]
    problem_triggers = [
        "无法",
        "不能",
        "不可以",
        "不好用",
        "故障",
        "异常",
        "卡顿",
        "黑屏",
        "死机",
        "不方便",
        "投诉",
        "不接受",
        "继续向前",
    ]

    suggestion = t if any(k in t for k in suggestion_triggers) else ""
    problem = t if any(k in t for k in problem_triggers) else t
    return problem, suggestion


def _emotion_score(text: str) -> float:
    t = _to_str(text)
    if not t:
        return 0.0

    positive = [
        "满意",
        "认可",
        "接受",
        "方便",
        "好用",
        "喜欢",
        "感谢",
        "期待",
    ]
    negative = [
        "投诉",
        "不接受",
        "无法",
        "不能",
        "故障",
        "异常",
        "差",
        "卡",
        "黑屏",
        "死机",
        "危险",
        "不满",
        "生气",
        "失控",
    ]

    pos = sum(1 for k in positive if k in t)
    neg = sum(1 for k in negative if k in t)
    raw = (pos - neg) / 4.0
    return float(max(-1.0, min(1.0, raw)))

def _impact_score(text: str) -> float:
    t = _to_str(text)
    if not t:
        return 0.0

    high = ["刹车", "转向", "失控", "碰撞", "安全", "起火", "气囊", "制动"]
    medium = ["无法启动", "无法充电", "无法解锁", "黑屏", "死机", "断连", "失灵"]
    low = ["卡顿", "不灵敏", "不方便", "体验", "优化", "希望", "建议"]

    if any(k in t for k in high):
        return 1.0
    if any(k in t for k in medium):
        return 0.7
    if any(k in t for k in low):
        return 0.4
    return 0.2


def _priority_score(freq: int, impact: float, emotion: float, max_freq: int) -> float:
    freq_norm = 0.0
    if max_freq > 0:
        freq_norm = math.log1p(freq) / math.log1p(max_freq)
    emotion_neg = max(0.0, -emotion)
    score = 0.5 * impact + 0.3 * freq_norm + 0.2 * emotion_neg
    return float(round(score * 100.0, 2))


def _sentiment_label(emotion_score: float) -> int:
    if emotion_score <= -0.2:
        return -1
    if emotion_score >= 0.2:
        return 1
    return 0


def _severity_label(impact_score: float) -> int:
    if impact_score >= 0.9:
        return 3
    if impact_score >= 0.6:
        return 2
    return 1


def _issue_type(text: str) -> str:
    t = _to_str(text)
    if not t:
        return "其他"

    if any(k in t for k in ["变丑", "不美观", "不如之前", "退化", "变差", "从", "变成"]) and any(
        k in t for k in ["变丑", "不美观", "不如之前", "退化", "变差", "变成"]
    ):
        return "体验退化"
    if any(k in t for k in ["无法", "不能", "异常", "故障", "黑屏", "死机", "不亮", "断连", "失灵"]):
        return "缺陷/故障"
    if any(k in t for k in ["建议", "希望", "期望", "能否", "能不能", "增加", "支持", "实现", "提供", "更新"]):
        return "需求"
    if any(k in t for k in ["卡顿", "不方便", "不灵敏", "体验"]):
        return "体验优化"
    return "其他"


def _infer_taxonomy(text: str, default_domain: str, default_module: str, default_scenario: str) -> tuple[str, str, str]:
    t = _to_str(text)

    rules = [
        ("智能座舱", "360影像", "倒车/环视影像", ["360", "环车", "全景影像", "环视", "影像"]),
        ("智能座舱", "导航", "路线规划/导航", ["导航", "高德", "地图", "ETC"]),
        ("智能驾驶", "辅助驾驶", "行车/智驾", ["智驾", "辅助驾驶", "NOA", "ACC", "拨杆"]),
        ("车联车控", "蓝牙钥匙", "近车解锁/离车落锁", ["蓝牙钥匙", "近车解锁", "离车落锁", "解锁", "落锁"]),
        ("智能座舱", "音响", "播放/音效", ["音响", "扬声器", "音效", "声音", "分区"]),
        ("智能座舱", "多屏协同", "屏幕/联动", ["副驾屏", "多屏", "屏幕", "中控", "仪表", "联动"]),
        ("售后服务", "服务流程", "维修进度/服务体验", ["售后", "维修", "进度", "服务群", "门店"]),
        ("车身灯光", "数字灯光", "智慧灯语/灯光", ["灯语", "灯光", "小蓝灯", "智慧灯语"]),
    ]

    for d, m, s, keys in rules:
        if any(k in t for k in keys):
            return d, m, s

    return default_domain, default_module, default_scenario


def _derive_need_suggestion(raw_text: str, issue_type: str) -> tuple[str, str]:
    t = re.sub(r"\s+", " ", _to_str(raw_text)).strip()
    if not t:
        return "", ""

    suggestion = ""
    if any(k in t for k in ["建议", "希望", "期望", "能否", "能不能", "要求"]):
        suggestion = t

    if issue_type == "体验退化":
        if any(k in t for k in ["360", "环车", "影像", "背景"]):
            return "保持影像界面美观与一致性", "恢复透明背景或提供样式切换"
        return "避免升级引入体验退化", suggestion
    if issue_type == "缺陷/故障":
        return "确保功能稳定可用", suggestion
    if issue_type == "需求":
        need = re.sub(r"(我)?(希望|期望|建议|能否|能不能|要求)", "需要", t)
        need = re.sub(r"\s+", " ", need).strip()
        return need, suggestion
    if issue_type == "体验优化":
        return "提升使用体验与效率", suggestion
    return "", suggestion


def _read_one_excel(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="工单数据", dtype="object")
    df["__source_file"] = path.name
    return df


def _coalesce_first_row(row: pd.Series, candidates: list[str]) -> str:
    for c in candidates:
        if c in row.index:
            v = _to_str(row.get(c))
            if v:
                return v
    return ""


def _read_raw_feedback(input_dir: Path) -> pd.DataFrame:
    files = sorted([p for p in input_dir.glob("*.xlsx") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in: {input_dir}")

    raw_frames = [_read_one_excel(p) for p in files]
    raw = pd.concat(raw_frames, ignore_index=True)
    return raw


def build_feedback_fact(input_dir: Path) -> pd.DataFrame:
    raw = _read_raw_feedback(input_dir)

    rows: list[dict] = []
    for idx, r in raw.iterrows():
        source = _to_str(r.get("__source_file"))
        channel = _coalesce_first_row(r, ["进线渠道", "来源渠道"])
        feedback_id_raw = _coalesce_first_row(r, ["工单编号", "工单ID", "编号", "id"])
        ctx = RowContext(source=source, channel=channel or None, feedback_id=feedback_id_raw or None)

        overview = _coalesce_first_row(r, ["概要", "问题描述", "内容", "反馈内容", "描述"])
        remark = _coalesce_first_row(r, ["备注", "补充说明", "补充"])
        feedback_text = _extract_feedback_text(overview, remark)
        if not feedback_text:
            continue

        domain = _coalesce_first_row(r, ["一级分类", "大类", "领域", "domain"])
        module = _coalesce_first_row(r, ["二级分类", "模块", "module"])
        scenario = _coalesce_first_row(r, ["三级分类", "场景", "scenario"])

        user_id = _coalesce_first_row(r, ["用户SID", "用户id", "user_id", "用户ID"])
        car_series = _coalesce_first_row(r, ["车系", "车系名称", "car_series"])
        vehicle_version = _coalesce_first_row(r, ["整车版本号", "版本号", "vehicle_version"])
        created_time = _coalesce_first_row(r, ["创建时间", "created_time", "创建日期", "时间"])

        problem, suggestion = _extract_problem_suggestion(feedback_text)
        emotion = _emotion_score(feedback_text)
        impact = _impact_score(feedback_text)

        rows.append(
            {
                "feedback_id": _build_feedback_id(ctx, idx),
                "source": source,
                "channel": channel,
                "user_id": user_id,
                "car_series": car_series,
                "vehicle_version": vehicle_version,
                "feedback_text": feedback_text,
                "domain": domain,
                "module": module,
                "scenario": scenario,
                "problem": problem,
                "suggestion": suggestion,
                "emotion_score": emotion,
                "impact_score": impact,
                "created_time": created_time,
            }
        )

    fact = pd.DataFrame(rows)
    if fact.empty:
        return fact.reindex(columns=FEEDBACK_FACT_COLUMNS)

    fact["__norm"] = fact["feedback_text"].map(_normalize_text)
    freq = fact.groupby("__norm").size().rename("frequency_score").reset_index()
    fact = fact.merge(freq, on="__norm", how="left")

    max_freq = int(fact["frequency_score"].max()) if not fact["frequency_score"].isna().all() else 0
    fact["priority_score"] = [
        _priority_score(int(f), float(i), float(e), max_freq)
        for f, i, e in zip(fact["frequency_score"], fact["impact_score"], fact["emotion_score"])
    ]

    fact = fact.drop(columns=["__norm"], errors="ignore").reindex(columns=FEEDBACK_FACT_COLUMNS)
    return fact


def build_issue_fact(feedback_fact: pd.DataFrame) -> pd.DataFrame:
    if feedback_fact.empty:
        return pd.DataFrame(columns=ISSUE_FACT_COLUMNS)

    issues_rows: list[dict] = []
    for _, r in feedback_fact.iterrows():
        feedback_id = _to_str(r.get("feedback_id"))
        source = _to_str(r.get("source"))
        channel = _to_str(r.get("channel"))
        user_id = _to_str(r.get("user_id"))
        car_series = _to_str(r.get("car_series"))
        vehicle_version = _to_str(r.get("vehicle_version"))
        created_time = _to_str(r.get("created_time"))
        feedback_text = _to_str(r.get("feedback_text"))

        default_domain = _to_str(r.get("domain"))
        default_module = _to_str(r.get("module"))
        default_scenario = _to_str(r.get("scenario"))

        issue_texts = _split_issues(feedback_text)
        if not issue_texts:
            continue

        for issue_no, raw_text in enumerate(issue_texts, 1):
            raw_text = _clean_issue_text(raw_text)
            itype = _issue_type(raw_text)
            domain, module, scenario = _infer_taxonomy(raw_text, default_domain, default_module, default_scenario)
            problem, extracted_suggestion = _extract_problem_suggestion(raw_text)
            emotion = _emotion_score(raw_text)
            impact = _impact_score(raw_text)
            sentiment = _sentiment_label(emotion)
            severity = _severity_label(impact)
            need, suggested = _derive_need_suggestion(raw_text, itype)
            suggestion = suggested or extracted_suggestion

            issues_rows.append(
                {
                    "feedback_id": feedback_id,
                    "issue_no": issue_no,
                    "source": source,
                    "channel": channel,
                    "user_id": user_id,
                    "car_series": car_series,
                    "vehicle_version": vehicle_version,
                    "created_time": created_time,
                    "raw_text": raw_text,
                    "issue_type": itype,
                    "domain": domain,
                    "module": module,
                    "scenario": scenario,
                    "problem": problem,
                    "need": need,
                    "suggestion": suggestion,
                    "sentiment": sentiment,
                    "severity": severity,
                    "emotion_score": emotion,
                    "impact_score": impact,
                }
            )

    issue_fact = pd.DataFrame(issues_rows)
    if issue_fact.empty:
        return issue_fact.reindex(columns=ISSUE_FACT_COLUMNS)

    issue_fact["__norm"] = issue_fact["raw_text"].map(_normalize_text)
    freq = issue_fact.groupby("__norm").size().rename("frequency_score").reset_index()
    issue_fact = issue_fact.merge(freq, on="__norm", how="left")
    max_freq = int(issue_fact["frequency_score"].max()) if not issue_fact["frequency_score"].isna().all() else 0
    issue_fact["priority_score"] = [
        _priority_score(int(f), float(i), float(e), max_freq)
        for f, i, e in zip(issue_fact["frequency_score"], issue_fact["impact_score"], issue_fact["emotion_score"])
    ]
    issue_fact = issue_fact.drop(columns=["__norm"], errors="ignore").reindex(columns=ISSUE_FACT_COLUMNS)
    return issue_fact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="1～4 月用户反馈",
        help="Directory containing .xlsx source files",
    )
    parser.add_argument(
        "--output-feedback",
        default="feedback_fact.csv",
        help="Output feedback_fact CSV file path",
    )
    parser.add_argument(
        "--output-issue",
        default="issue_fact_raw.csv",
        help="Output issue_fact CSV file path",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    feedback_out = Path(args.output_feedback)
    issue_out = Path(args.output_issue)

    fact = build_feedback_fact(input_dir)
    fact.to_csv(feedback_out, index=False, encoding="utf-8-sig")

    issue_fact = build_issue_fact(fact)
    issue_fact.to_csv(issue_out, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(fact)} rows to {feedback_out}")
    print(f"Wrote {len(issue_fact)} rows to {issue_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
