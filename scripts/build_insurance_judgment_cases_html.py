#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.domains.insurance.solver import InsuranceSolver
from afa_agent.io_utils import read_json
from afa_agent.models import Question, RetrievalHit


RUN_ROOT = ROOT / "artifacts" / "reproducible_runs" / "group_a_dynamic_confidence_20260630_180128"
ROUND0_ANSWERS = RUN_ROOT / "round0" / "runs" / "insurance_round0" / "outputs" / "debug" / "answers.json"
RESCUE_ANSWERS = RUN_ROOT / "rescue" / "runs" / "insurance_rescue" / "outputs" / "debug" / "answers.json"
INDEX_PATH = ROOT / "artifacts" / "preprocessed_loop_candidates" / "index" / "insurance" / "index.json"
DEFAULT_STRATEGY = ROOT / "configs" / "autoresearch" / "default_strategy.json"
RESCUE_STRATEGY = ROOT / "configs" / "autoresearch" / "evidence_gate_rescue_accuracy_first.json"
OUTPUT_PATH = ROOT / "artifacts" / "insurance_debug_cases" / "insurance_llm_judgment_cases.html"


CASES = [
    ("round0", "ins_a_001", "baseline 整题一次判断：多产品身故保险金排序"),
    ("round0", "ins_a_006", "baseline 整题一次判断：共享免赔额计算题"),
    ("rescue", "ins_a_007", "rescue 逐选项判断：保单贷款"),
    ("rescue", "ins_a_010", "rescue 逐选项判断：现金价值公式"),
    ("rescue", "ins_a_020", "规则直接覆盖：未调用大模型"),
]


def e(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def read_answers(path: Path) -> dict[str, dict[str, Any]]:
    return {row["qid"]: row for row in read_json(path)}


def load_questions() -> dict[str, Question]:
    manifest = read_json(ROOT / "artifacts" / "manifest" / "dataset_manifest.json")
    question_path = Path(manifest["domains"]["insurance"]["question_path"])
    rows = read_json(question_path)
    questions = {}
    for row in rows:
        if row.get("split") != "A":
            continue
        questions[row["qid"]] = Question(
            qid=row["qid"],
            domain=row["domain"],
            split=row["split"],
            question=row["question"],
            options=row["options"],
            answer_format=row["answer_format"],
            type=row["type"],
            doc_ids=row.get("doc_ids", []),
        )
    return questions


def load_units() -> dict[str, dict[str, Any]]:
    payload = read_json(INDEX_PATH)
    return {unit["unit_id"]: unit for unit in payload["units"]}


def set_strategy(path: Path) -> None:
    os.environ["AFA_STRATEGY_CONFIG"] = str(path.resolve())


def make_solver(strategy_path: Path) -> InsuranceSolver:
    set_strategy(strategy_path)
    return InsuranceSolver(client=None, retriever=None, strategy="insurance")


def hit_from_serialized(row: dict[str, Any], units_by_id: dict[str, dict[str, Any]]) -> RetrievalHit:
    unit = units_by_id.get(row.get("unit_id", ""), {})
    text = unit.get("text") or row.get("text") or row.get("text_preview", "")
    title_path = unit.get("title_path") or row.get("title_path", [])
    metadata = dict(unit.get("metadata", {}))
    metadata.update(row.get("metadata", {}) or {})
    metadata.setdefault("unit_type", unit.get("unit_type") or row.get("unit_type", ""))
    return RetrievalHit(
        unit_id=row.get("unit_id", ""),
        doc_id=row.get("doc_id", unit.get("doc_id", "")),
        score=float(row.get("score", 0.0) or 0.0),
        title_path=title_path,
        text=text,
        metadata=metadata,
    )


def hits_from_debug(rows: list[dict[str, Any]], units_by_id: dict[str, dict[str, Any]]) -> list[RetrievalHit]:
    return [hit_from_serialized(row, units_by_id) for row in rows]


def hits_from_evidence(rows: list[dict[str, Any]]) -> list[RetrievalHit]:
    hits = []
    for row in rows:
        metadata = dict(row.get("metadata", {}) or {})
        metadata.setdefault("unit_type", metadata.get("unit_type", ""))
        hits.append(
            RetrievalHit(
                unit_id=row.get("unit_id", ""),
                doc_id=row.get("doc_id", ""),
                score=float(row.get("score", 0.0) or 0.0),
                title_path=row.get("title_path", []),
                text=row.get("text", ""),
                metadata=metadata,
            )
        )
    return hits


def option_lines(question: Question) -> str:
    return "\n".join(f"{key}: {value}" for key, value in question.options.items())


def calc_hint(question: Question, option_text: str = "", include_sorting: bool = False) -> str:
    keywords = ["计算", "推理", "比较", "多少"]
    if include_sorting:
        keywords.append("排序")
    haystack = question.type + question.question + option_text
    if any(keyword in haystack for keyword in keywords):
        return "这可能是计算/比较题，请优先核对公式、适用条件、现金价值、账户价值、已交保费、基本保额。"
    return ""


def baseline_prompt(solver: InsuranceSolver, question: Question, hits: list[RetrievalHit]) -> tuple[str, str]:
    evidence_text = solver._format_prompt_hits(hits, default_max_items=4)
    hint = calc_hint(question)
    extra = solver.answering_settings.get("extra_context", "").strip()
    user = (
        f"题目：{question.question}\n题型：{question.answer_format}\n选项：\n{option_lines(question)}\n\n"
        f"{hint}\n{extra}\n\n证据：\n{evidence_text}\n\n"
        '请输出 JSON，格式为 {"answer": "A", "confidence": 0.0-1.0, "confidence_reason": "...", '
        '"reasoning_summary": "..."} 或 {"answer": "AC", "confidence": 0.0-1.0, "confidence_reason": "...", '
        '"reasoning_summary": "..."}。confidence 表示仅依据给定证据判断最终答案可靠性的置信度。'
    )
    return solver._system_prompt(), user


def option_prompt(
    solver: InsuranceSolver,
    question: Question,
    option_key: str,
    option_text: str,
    hits: list[RetrievalHit],
) -> tuple[str, str]:
    evidence_text = solver._format_prompt_hits(hits, default_max_items=6, option_text=option_text)
    hint = calc_hint(question, option_text, include_sorting=True)
    extra_context = solver._option_judgment_context(question, option_text, hint)
    user_parts = [
        f"题目：{question.question}",
        f"题型：{question.answer_format}",
        f"选项 {option_key}：{option_text}",
    ]
    if extra_context.strip():
        user_parts.append(extra_context.strip())
    user_parts.append(f"证据：\n{evidence_text}")
    user_parts.append(
        '请输出 JSON，格式为 {"label": true/false, "confidence": 0.0-1.0, '
        '"confidence_reason": "...", "reasoning_summary": "...", "used_evidence_ids": [1,2]}。'
        "confidence 表示仅基于给定证据判断该选项 label 是否可靠的置信度；证据缺关键指标、条款、公式或文档时必须降低。"
    )
    return solver._system_prompt(), "\n\n".join(user_parts)


def truncate(text: str, max_chars: int = 1600) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[display truncated]"


def render_tokens(row: dict[str, Any]) -> str:
    usage = row.get("token_usage", {}) or {}
    return (
        f"prompt={usage.get('prompt_tokens', 0)} / "
        f"completion={usage.get('completion_tokens', 0)} / "
        f"total={usage.get('total_tokens', 0)}"
    )


def render_query_variants(variants: list[str]) -> str:
    if not variants:
        return '<p class="muted">无 query 记录，通常表示规则直接覆盖或未进入检索。</p>'
    rows = []
    for idx, query in enumerate(variants, start=1):
        rows.append(f"<li><pre>{e(query)}</pre></li>")
    return f"<ol class=\"query-list\">{''.join(rows)}</ol>"


def render_top_hits(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="muted">无 topK 记录。</p>'
    items = []
    for idx, row in enumerate(rows, start=1):
        title = " > ".join(row.get("title_path", []))
        preview = row.get("text_preview", "")
        unit_type = row.get("unit_type") or (row.get("metadata", {}) or {}).get("unit_type", "")
        items.append(
            f"""
            <div class="hit">
              <div class="hit-head">#{idx} <b>{e(row.get('unit_id'))}</b> · doc={e(row.get('doc_id'))} · type={e(unit_type)} · score={float(row.get('score', 0) or 0):.2f}</div>
              <div class="hit-title">{e(title)}</div>
              <pre>{e(preview)}</pre>
            </div>
            """
        )
    return "\n".join(items)


def render_evidence(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="muted">无最终 evidence_items。</p>'
    cards = []
    for idx, row in enumerate(rows, start=1):
        title = " > ".join(row.get("title_path", []))
        unit_type = (row.get("metadata", {}) or {}).get("unit_type", "")
        cards.append(
            f"""
            <details class="evidence" open>
              <summary>证据 {idx}: {e(row.get('unit_id'))} · doc={e(row.get('doc_id'))} · type={e(unit_type)} · score={float(row.get('score', 0) or 0):.2f}</summary>
              <div class="hit-title">{e(title)}</div>
              <pre>{e(truncate(row.get('text', ''), 1800))}</pre>
            </details>
            """
        )
    return "\n".join(cards)


def split_reasoning_by_option(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in re.split(r"\s+\|\s+", text or ""):
        match = re.match(r"^([A-D])[:：]\s*(.*)$", part.strip(), re.S)
        if match:
            result[match.group(1)] = match.group(2).strip()
    return result


def render_baseline_case(
    case_title: str,
    question: Question,
    row: dict[str, Any],
    solver: InsuranceSolver,
    units_by_id: dict[str, dict[str, Any]],
) -> str:
    debug = row.get("debug_meta", {}) or {}
    hits = hits_from_debug(debug.get("retrieval_topk", []), units_by_id)
    system_prompt, user_prompt = baseline_prompt(solver, question, hits)
    stage_summary = "baseline / 整题一次 LLM 判断"
    return f"""
    <section class="case" id="{e(row['qid'])}-round0">
      <h2>{e(row['qid'])}: {e(case_title)}</h2>
      <div class="chips">
        <span>{e(stage_summary)}</span><span>answer={e(row.get('pred_answer'))}</span><span>{e(render_tokens(row))}</span>
      </div>
      {render_question(question)}
      <div class="grid two">
        <div>{panel('检索字段', render_retrieval_fields(debug, question))}</div>
        <div>{panel('大模型答案 / 解析结果', render_model_answer(row))}</div>
      </div>
      {panel('Prompt 拼接（按当前源码模板重建）', render_prompt(system_prompt, user_prompt))}
      {panel('检索 TopK', render_top_hits(debug.get('retrieval_topk', [])))}
      {panel('最终证据链 evidence_items', render_evidence(row.get('evidence_items', [])))}
    </section>
    """


def render_rescue_case(
    case_title: str,
    question: Question,
    row: dict[str, Any],
    solver: InsuranceSolver,
    units_by_id: dict[str, dict[str, Any]],
) -> str:
    debug = row.get("debug_meta", {}) or {}
    option_debug = debug.get("option_debug", []) or []
    option_reasoning = split_reasoning_by_option(row.get("reasoning_summary", ""))
    if debug.get("early_rule_answer"):
        prompt_html = '<p class="muted">该 case 在 rescue 路径被规则直接覆盖，token 为 0，未发起大模型调用。</p>'
        option_html = render_rule_case(debug)
    else:
        blocks = []
        for od in option_debug:
            option = od.get("option", "")
            option_text = question.options.get(option, "")
            hits = hits_from_debug(od.get("retrieval_topk", []), units_by_id)
            system_prompt, user_prompt = option_prompt(solver, question, option, option_text, hits)
            gate = ((od.get("evidence_gate", {}) or {}).get("final_gate", {}) or {})
            evidence_gate_json = json.dumps(od.get("evidence_gate", {}), ensure_ascii=False, indent=2)
            blocks.append(
                f"""
                <details class="option-block" open>
                  <summary>选项 {e(option)} · search_doc_ids={e(od.get('search_doc_ids'))} · model_confidence={e(od.get('model_confidence'))} · gate={e(gate.get('status'))}</summary>
                  <div class="option-answer"><b>该选项判断：</b>{e(question.options.get(option, ''))}<br><b>reasoning：</b>{e(option_reasoning.get(option, ''))}</div>
                  <div class="subgrid">
                    <div>{panel('该选项 query variants', render_query_variants(od.get('query_variants', [])))}</div>
                    <div>{panel('该选项 TopK', render_top_hits(od.get('retrieval_topk', [])))}</div>
                  </div>
                  {panel('该选项 Prompt（按当前源码模板重建）', render_prompt(system_prompt, user_prompt))}
                  {panel('该选项 evidence gate', f'<pre>{e(evidence_gate_json)}</pre>')}
                </details>
                """
            )
        prompt_html = "\n".join(blocks)
        option_html = render_option_labels(row, option_reasoning)
    return f"""
    <section class="case" id="{e(row['qid'])}-rescue">
      <h2>{e(row['qid'])}: {e(case_title)}</h2>
      <div class="chips">
        <span>rescue / {'规则覆盖' if debug.get('early_rule_answer') else '逐选项 LLM 判断'}</span>
        <span>answer={e(row.get('pred_answer'))}</span><span>{e(render_tokens(row))}</span>
      </div>
      {render_question(question)}
      <div class="grid two">
        <div>{panel('检索字段', render_retrieval_fields(debug, question))}</div>
        <div>{panel('大模型答案 / 解析结果', render_model_answer(row) + option_html)}</div>
      </div>
      {prompt_html}
      {panel('最终证据链 evidence_items', render_evidence(row.get('evidence_items', [])))}
    </section>
    """


def render_rule_case(debug: dict[str, Any]) -> str:
    return panel(
        "规则输出",
        f"<pre>{e(json.dumps(debug.get('rule_outputs', []), ensure_ascii=False, indent=2))}</pre>",
    )


def render_option_labels(row: dict[str, Any], option_reasoning: dict[str, str]) -> str:
    labels = row.get("option_labels", {}) or {}
    items = []
    for option in sorted(labels):
        items.append(
            f"<li><b>{e(option)}</b>: label={e(labels[option])}; reasoning={e(option_reasoning.get(option, ''))}</li>"
        )
    return f"<h4>逐选项解析</h4><ul>{''.join(items)}</ul>"


def render_model_answer(row: dict[str, Any]) -> str:
    debug = row.get("debug_meta", {}) or {}
    payload = {
        "pred_answer": row.get("pred_answer", ""),
        "option_labels": row.get("option_labels", {}),
        "reasoning_summary": row.get("reasoning_summary", ""),
        "model_confidence": debug.get("model_confidence"),
        "answer_finalization": debug.get("answer_finalization", {}),
        "final_consistency_check": debug.get("final_consistency_check", {}),
        "consistency_answers": debug.get("consistency_answers", []),
    }
    return f"<pre>{e(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>"


def render_retrieval_fields(debug: dict[str, Any], question: Question) -> str:
    rows = [
        ("doc_ids", question.doc_ids),
        ("prompt_template_id", debug.get("prompt_template_id")),
        ("query_variants_count", len(debug.get("query_variants", []) or [])),
        ("selected_evidence_ids", debug.get("selected_evidence_ids", [])),
        ("single_call_mcq", debug.get("single_call_mcq")),
        ("early_rule_answer", debug.get("early_rule_answer")),
    ]
    table = "".join(f"<tr><th>{e(k)}</th><td><code>{e(v)}</code></td></tr>" for k, v in rows)
    return f"<table class=\"meta\">{table}</table><h4>query variants</h4>{render_query_variants(debug.get('query_variants', []))}"


def render_question(question: Question) -> str:
    options = "".join(
        f"<tr><th>{e(key)}</th><td>{e(value)}</td></tr>"
        for key, value in question.options.items()
    )
    return f"""
    <div class="question">
      <div class="qmeta">type={e(question.type)} · format={e(question.answer_format)} · doc_ids={e(question.doc_ids)}</div>
      <p>{e(question.question)}</p>
      <table class="options">{options}</table>
    </div>
    """


def render_prompt(system_prompt: str, user_prompt: str) -> str:
    return f"""
    <div class="prompt">
      <h4>system</h4>
      <pre>{e(system_prompt)}</pre>
      <h4>user</h4>
      <pre>{e(user_prompt)}</pre>
    </div>
    """


def panel(title: str, body: str) -> str:
    return f"<div class=\"panel\"><h3>{e(title)}</h3>{body}</div>"


def render_page(sections: list[str]) -> str:
    nav = "".join(
        f'<a href="#{e(qid)}-{e(stage)}">{e(qid)} {e(stage)}</a>'
        for stage, qid, _title in CASES
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Insurance LLM Judgment Cases</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #202536;
      --muted: #667085;
      --line: #d9dee8;
      --accent: #3157d5;
      --accent-soft: #eef3ff;
      --warn: #8a5a00;
      --code: #101828;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text); background: var(--bg); }}
    header {{ position: sticky; top: 0; z-index: 3; padding: 16px 22px; background: rgba(255,255,255,.94); border-bottom: 1px solid var(--line); backdrop-filter: blur(10px); }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; }}
    h3 {{ margin: 0 0 10px; font-size: 15px; color: #344054; }}
    h4 {{ margin: 12px 0 6px; font-size: 13px; color: #475467; }}
    .note {{ color: var(--muted); font-size: 13px; line-height: 1.55; max-width: 1180px; }}
    nav {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    nav a {{ color: var(--accent); text-decoration: none; font-size: 13px; padding: 5px 8px; background: var(--accent-soft); border: 1px solid #d7e2ff; border-radius: 6px; }}
    main {{ padding: 22px; max-width: 1440px; margin: 0 auto; }}
    .case {{ margin-bottom: 28px; padding: 18px; background: #fff; border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 1px 2px rgba(16,24,40,.04); }}
    .chips {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }}
    .chips span {{ display: inline-flex; align-items: center; min-height: 28px; padding: 4px 9px; border-radius: 6px; background: #f2f4f7; border: 1px solid #eaecf0; font-size: 13px; color: #344054; }}
    .question {{ padding: 12px; background: #fbfcff; border: 1px solid var(--line); border-radius: 6px; margin-bottom: 14px; }}
    .qmeta {{ font-size: 13px; color: var(--muted); margin-bottom: 8px; }}
    .question p {{ margin: 0 0 10px; line-height: 1.6; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid var(--line); padding: 8px 9px; vertical-align: top; font-size: 13px; }}
    th {{ width: 150px; text-align: left; background: #f8fafc; color: #475467; }}
    .options th {{ width: 48px; }}
    .grid.two {{ display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap: 14px; }}
    .subgrid {{ display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap: 12px; }}
    .panel {{ border: 1px solid var(--line); border-radius: 8px; background: var(--panel); padding: 12px; margin: 12px 0; overflow: hidden; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; line-height: 1.48; color: var(--code); background: #f8fafc; border: 1px solid #eaecf0; border-radius: 6px; padding: 10px; max-height: 520px; overflow: auto; }}
    code {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
    .query-list {{ margin: 0; padding-left: 22px; }}
    .query-list li {{ margin-bottom: 8px; }}
    .hit {{ border: 1px solid #eaecf0; border-radius: 6px; padding: 10px; margin: 10px 0; background: #fcfcfd; }}
    .hit-head {{ font-size: 13px; color: #344054; margin-bottom: 6px; }}
    .hit-title {{ font-size: 12px; color: var(--muted); margin: 5px 0 8px; }}
    details.evidence, details.option-block {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px; margin: 12px 0; background: #fff; }}
    summary {{ cursor: pointer; font-weight: 600; color: #344054; }}
    .option-answer {{ margin: 10px 0; padding: 10px; background: #fff8eb; border: 1px solid #ffe3ad; border-radius: 6px; color: var(--warn); font-size: 13px; line-height: 1.55; }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    @media (max-width: 980px) {{
      main {{ padding: 12px; }}
      .grid.two, .subgrid {{ grid-template-columns: 1fr; }}
      header {{ position: static; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>保险题判断链路审计：检索字段 / Prompt / 模型答案 / 证据链</h1>
    <div class="note">
      数据来源：{e(RUN_ROOT)}。现有 debug 没保存 API raw response，因此“大模型答案”展示的是已解析的 pred_answer、option_labels、confidence、reasoning_summary；
      Prompt 为按当前源码模板和 debug topK 重建。最终证据链来自 answer debug 的 evidence_items，展示文本做了页面级截断。
    </div>
    <nav>{nav}</nav>
  </header>
  <main>
    {''.join(sections)}
  </main>
</body>
</html>
"""


def main() -> None:
    questions = load_questions()
    units_by_id = load_units()
    round0_answers = read_answers(ROUND0_ANSWERS)
    rescue_answers = read_answers(RESCUE_ANSWERS)
    baseline_solver = make_solver(DEFAULT_STRATEGY)
    rescue_solver = make_solver(RESCUE_STRATEGY)

    sections = []
    for stage, qid, title in CASES:
        question = questions[qid]
        if stage == "round0":
            sections.append(render_baseline_case(title, question, round0_answers[qid], baseline_solver, units_by_id))
        else:
            sections.append(render_rescue_case(title, question, rescue_answers[qid], rescue_solver, units_by_id))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_page(sections), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
