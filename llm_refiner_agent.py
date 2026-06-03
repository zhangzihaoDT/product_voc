import argparse
import ast
import hashlib
import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, local

import pandas as pd


REFINE_FIELDS = [
    "is_voc",
    "issue_type",
    "domain",
    "module",
    "scenario",
    "problem",
    "need_theme_l2",
]

DEFAULT_TAXONOMY = [
    ("智驾体验", "智驾接管安全", "智驾,接管,拨杆,红绿灯,刹车,安全"),
    ("智驾体验", "自动泊车体验", "泊车,自动泊车,APA,遥控泊车"),
    ("智驾体验", "智驾决策优化", "变道,加塞,决策,跟车"),
    ("智驾体验", "智驾安全策略", "雨天,动态降速,安全策略"),
    ("智舱体验", "360影像视觉体验", "360,环车,全景影像,环视,影像,背景,补盲,雨夜模式,A柱盲区,侧区"),
    ("智舱体验", "导航信息准确性", "导航,地图,高德,ETC,收费,路线,测速摄像头,海拔,充电站信息,剩余电量"),
    ("智舱体验", "语音识别精准性", "语音,识别,唤醒,听不懂"),
    ("智舱体验", "多屏娱乐协同", "多屏,副驾屏,中控,仪表,联动"),
    ("智舱体验", "音响分区控制", "音响,扬声器,音效,分区"),
    ("车联车控", "蓝牙钥匙稳定性", "蓝牙钥匙,近车解锁,离车落锁,解锁,落锁"),
    ("车联车控", "车门状态提醒", "车门,没关好,锁车,提示,语音提示"),
    ("车联车控", "门把手误触优化", "门把手,误触"),
    ("灯光体验", "智慧灯语自定义", "灯语,智慧灯语,灯光,小蓝灯"),
    ("灯光体验", "灯光秀异常", "灯光秀,灯光异常"),
    ("APP体验", "APP体验优化", "APP,卡顿,闪退,流畅,桌面卡片,行程轨迹,手表APP,启动页,邀请海报,里程统计"),
    ("账号与权益", "账号换绑与登录", "换绑,登录,账号,手机号"),
    ("账号与权益", "签到规则优化", "签到,积分,规则"),
    ("内容生态", "腾讯视频功能", "腾讯视频,视频"),
    ("内容生态", "音乐展示优化", "音乐,歌名,名称显示"),
    ("服务体验", "维修进度透明度", "售后,维修,进度,服务群,门店"),
    ("服务体验", "服务响应效率", "响应,及时,回电,处理"),
    ("服务体验", "服务流程精细化", "流程,服务精细化"),
    ("服务体验", "门店休息区体验", "等候区,休息区,舒适"),
    ("服务体验", "洗车服务质量", "洗车"),
    ("服务体验", "维修质量问题", "维修质量,未解决,返修"),
    ("交付体验", "提车服务效率", "提车,交付,排队"),
    ("交付体验", "配件交付延迟", "配件,到货,延迟"),
    ("交付体验", "销售承诺兑现", "销售,承诺,兑现"),
    ("车辆品质", "胎压告警位置问题", "胎压,告警"),
    ("车辆品质", "车门异响问题", "车门,异响"),
    ("车辆品质", "方向盘设计优化", "方向盘"),
    ("智舱体验", "车机应用与功能扩展", "应用商店,APP下载,自定义安装,酷狗,优酷,NAS播放,AI宠物,交互机器人,多媒体外放"),
    ("智舱体验", "音响音质优化", "音响,音质,杂音,声浪,喇叭,播放,失真"),
    ("账号与权益", "车主权益与政策", "质保,延保,权益,补贴,拒保,退订,退款,保值,二手车,保险"),
    ("服务体验", "商城体验", "商城,放电枪,屏幕保护膜,原石支付,配件购买,商品丰富度,发货"),
    ("其他", "其他", ""),
]


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    base_url: str
    model: str
    timeout_s: int
    max_tokens: int
    temperature: float


class SqliteCache:
    def __init__(self, path: Path):
        self._path = path
        self._thread_local = local()
        self._init_lock = Lock()
        self._initialized = False

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._thread_local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._path), timeout=30, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            self._thread_local.conn = conn
        if not self._initialized:
            with self._init_lock:
                if not self._initialized:
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS cache ("
                        "k TEXT PRIMARY KEY,"
                        "v TEXT NOT NULL,"
                        "created_at INTEGER NOT NULL"
                        ")"
                    )
                    conn.commit()
                    self._initialized = True
        return conn

    def get(self, key: str) -> dict | None:
        conn = self._conn()
        row = conn.execute("SELECT v FROM cache WHERE k = ?", (key,)).fetchone()
        if row is None:
            return None
        try:
            v = json.loads(row[0])
            return v if isinstance(v, dict) else None
        except Exception:
            return None

    def set(self, key: str, value: dict) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO cache(k, v, created_at) VALUES(?, ?, ?)",
            (key, json.dumps(value, ensure_ascii=False), int(time.time())),
        )
        conn.commit()


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k and k not in os.environ:
            os.environ[k] = v


def _to_str(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def _cache_key(model: str, system_prompt: str, user_json: dict) -> str:
    raw = "\n".join(
        [
            _to_str(model),
            hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            json.dumps(user_json, ensure_ascii=False, sort_keys=True),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()



def _deepseek_chat_json(cfg: DeepSeekConfig, system_prompt: str, user_json: dict, max_tokens: int | None = None) -> dict:
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_json, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
        "max_tokens": int(max_tokens if max_tokens is not None else cfg.max_tokens),
        "temperature": cfg.temperature,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e}") from e

    data = json.loads(body)
    choice0 = data.get("choices", [{}])[0] if isinstance(data, dict) else {}
    finish_reason = choice0.get("finish_reason")
    if finish_reason == "length":
        raise ValueError("Truncated model output (finish_reason=length)")

    message = choice0.get("message", {}) if isinstance(choice0, dict) else {}
    content = (message.get("content") if isinstance(message, dict) else "") or ""
    content = str(content)
    if not content.strip():
        raise ValueError("Empty model content")
    return _parse_json_object(content)


def _parse_json_object(content: str) -> dict:
    raw = content.strip()
    if raw.count("{") > raw.count("}"):
        raise ValueError(f"Truncated JSON (missing }}): {raw[:200]}")

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start : end + 1].strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            try:
                obj = ast.literal_eval(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    raise ValueError(f"Model output is not a valid JSON object: {raw[:200]}")


def _mask_pii(text: str) -> str:
    t = _to_str(text)
    if not t:
        return ""

    t = re.sub(r"\b1[3-9]\d{9}\b", "手机号<REDACTED>", t)
    t = re.sub(r"\b[A-HJ-NPR-Z0-9]{17}\b", "VIN<REDACTED>", t, flags=re.IGNORECASE)
    t = re.sub(r"(?i)\b(tp|工单|ticket)[-_:]?\d+\b", "工单<REDACTED>", t)

    t = re.sub(r"(?m)^\s*(姓名|联系方式|车架号|VIN|手机号|来电号码|回电号码)\s*[:：].*$", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def _load_taxonomy(taxonomy_path: Path) -> tuple[list[tuple[str, str, str]], dict[str, tuple[str, str]]]:
    if not taxonomy_path.exists():
        return list(DEFAULT_TAXONOMY), {}

    df = pd.read_csv(taxonomy_path, dtype="object")
    rows: list[tuple[str, str, str]] = []
    mapping: dict[str, tuple[str, str]] = {}
    for _, r in df.iterrows():
        l1 = _to_str(r.get("need_theme_l1"))
        l2 = _to_str(r.get("need_theme_l2"))
        keywords = _to_str(r.get("keywords"))
        if not l2:
            continue
        rows.append((l1 or "其他", l2, keywords))
        jtbd_l1 = _to_str(r.get("jtbd_l1"))
        scenario_l2 = _to_str(r.get("scenario_l2"))
        if jtbd_l1 or scenario_l2:
            mapping[l2] = (jtbd_l1 or "其他", scenario_l2 or "其他")

    if not rows:
        return list(DEFAULT_TAXONOMY), mapping
    if not any(l2 == "其他" for _, l2, _ in rows):
        rows.append(("其他", "其他", ""))
    if "其他" not in mapping:
        mapping["其他"] = ("其他", "其他")
    return rows, mapping


def _match_taxonomy_by_keywords(taxonomy: list[tuple[str, str, str]], text: str) -> tuple[str, str, int]:
    t = _to_str(text)
    if not t:
        return "其他", "其他", 0

    best_l1 = "其他"
    best_l2 = "其他"
    best_score = 0
    for l1, l2, keywords in taxonomy:
        keys = [k.strip() for k in _to_str(keywords).split(",") if k.strip()]
        if not keys:
            continue
        score = sum(1 for k in keys if k in t)
        if score > best_score:
            best_score = score
            best_l1 = l1
            best_l2 = l2

    return best_l1, best_l2, best_score


def _build_system_prompt(taxonomy: list[tuple[str, str, str]], allow_new_theme: bool) -> str:
    options = "\n".join(f"- {l1} / {l2}" for l1, l2, _ in taxonomy)
    return (
        "你是 VOC issue 结构化标注器，只能输出 JSON 对象，不能输出任何多余文本。\n"
        "任务：基于输入字段 raw_text + rule_domain/module/scenario + car_series + channel，输出下列字段：\n"
        f"{json.dumps(REFINE_FIELDS, ensure_ascii=False)}\n"
        "字段约束：\n"
        "- is_voc: boolean，若不是用户反馈/需求/缺陷/体验问题（例如纯寒暄、无意义文本），输出 false。\n"
        "- issue_type: string，优先从以下集合选取：['功能增强','体验退化','缺陷/故障','体验优化','服务/流程','其他']。\n"
        "- domain/module/scenario/problem: string，尽量具体，若无法判断可沿用规则值。\n"
        "- need_theme_l2: string，必须从候选主题列表中选择一个；若都不匹配，选择 '其他'。\n"
        "候选主题列表（need_theme_l1 / need_theme_l2）：\n"
        f"{options}\n"
        "输出要求：\n"
        "- 必须是可被 json.loads 解析的 JSON 对象\n"
        "- 必须使用双引号作为 JSON 键名与字符串边界\n"
        "- 只输出上述字段，不要输出其它字段\n"
        "- 如果 is_voc=false，则 issue_type/domain/module/scenario/problem/need_theme_l2 置为空字符串\n"
        + (
            ""
            if not allow_new_theme
            else "- 仅当候选主题完全无法覆盖时，need_theme_l2 可以输出 '新主题候选:XXXX'（XXXX<=12字）。\n"
        )
    )


def _sanitize_llm_output(
    rule_row: dict,
    llm: dict,
    allowed_need_theme_l2: set[str],
    allow_new_theme: bool,
) -> dict:
    out = {k: llm.get(k) for k in REFINE_FIELDS}
    out["is_voc"] = bool(out.get("is_voc"))

    for k in ["issue_type", "domain", "module", "scenario", "problem", "need_theme_l2"]:
        v = out.get(k)
        if v is None:
            v = ""
        out[k] = _to_str(v)

    if not out["is_voc"]:
        for k in ["issue_type", "domain", "module", "scenario", "problem", "need_theme_l2"]:
            out[k] = ""
        return out

    if not out["issue_type"]:
        out["issue_type"] = _to_str(rule_row.get("issue_type"))
    if not out["domain"]:
        out["domain"] = _to_str(rule_row.get("domain"))
    if not out["module"]:
        out["module"] = _to_str(rule_row.get("module"))
    if not out["scenario"]:
        out["scenario"] = _to_str(rule_row.get("scenario"))
    if not out["problem"]:
        out["problem"] = _to_str(rule_row.get("problem")) or _to_str(rule_row.get("raw_text"))

    theme = out["need_theme_l2"]
    if allow_new_theme and theme.startswith("新主题候选:"):
        candidate = _to_str(theme.split(":", 1)[1])[:12]
        out["need_theme_l2"] = f"新主题候选:{candidate}" if candidate else "其他"
        return out

    if theme not in allowed_need_theme_l2:
        out["need_theme_l2"] = "其他"
    if not out["need_theme_l2"]:
        out["need_theme_l2"] = "其他"

    return out


def refine_issue_fact(
    issue_fact_raw: pd.DataFrame,
    cfg: DeepSeekConfig,
    sample_size: int,
    seed: int,
    taxonomy: list[tuple[str, str, str]],
    jtbd_scenario_map: dict[str, tuple[str, str]],
    allow_new_theme: bool,
    sleep_s: float,
    max_retries: int,
    output_path: Path,
    save_every: int,
    resume: bool,
    concurrency: int,
    cache: SqliteCache | None,
) -> pd.DataFrame:
    if issue_fact_raw.empty:
        raise ValueError("issue_fact_raw is empty")

    allowed_need_theme_l2 = {l2 for _, l2, _ in taxonomy}
    need_theme_l2_to_l1 = {l2: l1 for l1, l2, _ in taxonomy}

    if sample_size > 0:
        work = issue_fact_raw.sample(n=min(sample_size, len(issue_fact_raw)), random_state=seed).copy()
    else:
        work = issue_fact_raw.copy()

    system_prompt = _build_system_prompt(taxonomy, allow_new_theme)

    processed_keys: set[tuple[str, str]] = set()
    output_columns: list[str] | None = None
    if resume and output_path.exists():
        existing = pd.read_csv(output_path, dtype="object")
        output_columns = list(existing.columns)
        missing_cols = [c for c in ["jtbd_l1", "scenario_l2"] if c not in existing.columns]
        if missing_cols:
            for c in ["jtbd_l1", "scenario_l2"]:
                if c not in output_columns:
                    output_columns.append(c)
            inferred = existing.apply(
                lambda rr: jtbd_scenario_map.get(_to_str(rr.get("need_theme_l2")) or _to_str(rr.get("need_theme")) or "其他", ("其他", "其他")),
                axis=1,
                result_type="expand",
            )
            inferred.columns = ["jtbd_l1", "scenario_l2"]
            if "jtbd_l1" not in existing.columns:
                existing["jtbd_l1"] = inferred["jtbd_l1"]
            if "scenario_l2" not in existing.columns:
                existing["scenario_l2"] = inferred["scenario_l2"]
            existing.to_csv(output_path, index=False, encoding="utf-8-sig")
        for _, r in existing.iterrows():
            processed_keys.add((_to_str(r.get("feedback_id")), _to_str(r.get("issue_no"))))

    if output_columns is None:
        output_columns = list(work.columns)
        for c in ["is_voc", "need_theme_l1", "need_theme_l2", "need_theme", "jtbd_l1", "scenario_l2"]:
            if c not in output_columns:
                output_columns.append(c)

    pending: list[dict] = []
    for _, row in work.iterrows():
        rule_row = row.to_dict()
        key = (_to_str(rule_row.get("feedback_id")), _to_str(rule_row.get("issue_no")))
        if key in processed_keys:
            continue
        pending.append(rule_row)

    total = len(work)
    done_count = len(processed_keys)
    if not resume:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=output_columns).to_csv(output_path, index=False, encoding="utf-8-sig")

    def worker(rule_row: dict) -> dict:
        llm_input = {
            "raw_text": _mask_pii(_to_str(rule_row.get("raw_text"))),
            "rule_domain": _to_str(rule_row.get("domain")),
            "rule_module": _to_str(rule_row.get("module")),
            "rule_scenario": _to_str(rule_row.get("scenario")),
            "car_series": _to_str(rule_row.get("car_series")),
            "channel": _to_str(rule_row.get("channel")),
        }

        ck = _cache_key(cfg.model, system_prompt, llm_input)
        llm_out = None
        if cache is not None:
            cached = cache.get(ck)
            if cached is not None:
                llm_out = cached

        if llm_out is None:
            token_budget = int(cfg.max_tokens)
            last_err = None
            for attempt in range(max_retries + 1):
                try:
                    llm_out = _deepseek_chat_json(cfg, system_prompt, llm_input, max_tokens=token_budget)
                    break
                except Exception as e:
                    last_err = e
                    if "Truncated" in str(e) or "finish_reason=length" in str(e) or "missing }" in str(e):
                        token_budget = min(token_budget * 2, 2048)
                    time.sleep(min(2.0 ** attempt, 8.0))
            if llm_out is None:
                raise RuntimeError(f"LLM call failed: {last_err}") from last_err
            if cache is not None:
                cache.set(ck, llm_out)

        refined = _sanitize_llm_output(
            rule_row=rule_row,
            llm=llm_out,
            allowed_need_theme_l2=allowed_need_theme_l2,
            allow_new_theme=allow_new_theme,
        )

        out_row = dict(rule_row)
        out_row["is_voc"] = refined["is_voc"]
        out_row["issue_type"] = refined["issue_type"] or _to_str(out_row.get("issue_type"))
        out_row["domain"] = refined["domain"] or _to_str(out_row.get("domain"))
        out_row["module"] = refined["module"] or _to_str(out_row.get("module"))
        out_row["scenario"] = refined["scenario"] or _to_str(out_row.get("scenario"))
        out_row["problem"] = refined["problem"] or _to_str(out_row.get("problem"))
        need_theme_l2 = refined["need_theme_l2"]
        if need_theme_l2 == "其他":
            _, kw_l2, kw_score = _match_taxonomy_by_keywords(taxonomy, _to_str(out_row.get("raw_text")))
            if kw_score > 0 and kw_l2 != "其他":
                need_theme_l2 = kw_l2
        if allow_new_theme and need_theme_l2.startswith("新主题候选:"):
            need_theme_l1 = "新主题候选"
        else:
            need_theme_l1 = need_theme_l2_to_l1.get(need_theme_l2, "其他")
        out_row["need_theme_l1"] = need_theme_l1
        out_row["need_theme_l2"] = need_theme_l2
        out_row["need_theme"] = need_theme_l2
        jtbd_l1, scenario_l2 = jtbd_scenario_map.get(need_theme_l2, ("其他", "其他"))
        out_row["jtbd_l1"] = jtbd_l1
        out_row["scenario_l2"] = scenario_l2
        if sleep_s > 0:
            time.sleep(sleep_s)
        return out_row

    write_lock = Lock()
    buffer: list[dict] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return
        batch = pd.DataFrame(buffer)
        batch = batch.reindex(columns=output_columns)
        with write_lock:
            batch.to_csv(output_path, index=False, encoding="utf-8-sig", mode="a", header=False)
        buffer = []

    max_workers = max(1, int(concurrency))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(worker, r) for r in pending]
        for fut in as_completed(futures):
            out_row = fut.result()
            buffer.append(out_row)
            done_count += 1
            if done_count % 10 == 0:
                print(f"refined {done_count}/{total}")
            if save_every > 0 and len(buffer) >= save_every:
                flush_buffer()

    flush_buffer()

    if output_path.exists():
        return pd.read_csv(output_path, dtype="object")
    return pd.DataFrame(columns=output_columns)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="issue_fact_raw.csv")
    parser.add_argument("--output", default="issue_fact_refined_v2.csv")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--taxonomy", default="need_theme_dict.csv")
    parser.add_argument("--allow-new-theme", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--cache-path", default=".llm_refiner_cache.sqlite")
    parser.add_argument("--no-cache", action="store_true", default=False)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--timeout-s", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep-s", type=float, default=0.1)
    parser.add_argument("--max-retries", type=int, default=6)
    args = parser.parse_args()

    _load_dotenv(Path(".env"))
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY in environment or .env")

    cfg = DeepSeekConfig(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    issue_raw = pd.read_csv(args.input, dtype="object")
    taxonomy, jtbd_scenario_map = _load_taxonomy(Path(args.taxonomy))
    output_path = Path(args.output)
    save_every = int(args.save_every)
    if args.checkpoint_every is not None:
        save_every = int(args.checkpoint_every)
    cache = None if args.no_cache else SqliteCache(Path(args.cache_path))
    refined = refine_issue_fact(
        issue_fact_raw=issue_raw,
        cfg=cfg,
        sample_size=args.sample_size,
        seed=args.seed,
        taxonomy=taxonomy,
        jtbd_scenario_map=jtbd_scenario_map,
        allow_new_theme=bool(args.allow_new_theme),
        sleep_s=args.sleep_s,
        max_retries=args.max_retries,
        output_path=output_path,
        save_every=save_every,
        resume=bool(args.resume),
        concurrency=int(args.concurrency),
        cache=cache,
    )
    print(f"Wrote {len(refined)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
