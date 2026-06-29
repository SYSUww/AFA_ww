#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


DEFAULT_CASES = [
    ("reg_a_015", "B"),
    ("reg_a_016", "C"),
    ("reg_a_018", "A"),
    ("reg_a_019", "B"),
    ("reg_a_019", "C"),
    ("reg_a_019", "D"),
]


def esc(value: object) -> str:
    return html.escape(str(value or ""))


def case_key(qid: str, option: str) -> str:
    return f"{qid}:{option}"


def token_pills(tokens: list[str], added: set[str], removed: set[str], mode: str) -> str:
    pills = []
    for token in tokens:
        cls = "token"
        if mode == "experiment" and token in added:
            cls += " add"
        if mode == "baseline" and token in removed:
            cls += " sub"
        pills.append(f'<span class="{cls}">{esc(token)}</span>')
    return "\n".join(pills)


def hit_card(hit: dict, peer_ids: set[str], top_peer: str) -> str:
    unit_id = hit.get("unit_id", "")
    cls = "hit"
    if unit_id == top_peer:
        cls += " same-top"
    elif unit_id in peer_ids:
        cls += " overlap"
    else:
        cls += " changed"
    matched = " ".join(f'<span class="match">{esc(token)}</span>' for token in hit.get("matched_tokens", [])[:28])
    title = " / ".join(hit.get("title_path") or [])
    return f"""
      <article class="{cls}">
        <header>
          <b>#{hit.get('rank')} {esc(hit.get('score'))}</b>
          <code>{esc(unit_id)}</code>
        </header>
        <div class="doc">{esc(hit.get('doc_id'))}</div>
        <h4>{esc(title)}</h4>
        <p>{esc(hit.get('text_preview'))}</p>
        <div class="matches">{matched}</div>
      </article>
    """


def render_case(question: dict, option: dict) -> str:
    baseline_ids = set(option["baseline_top_ids"])
    experiment_ids = set(option["experiment_top_ids"])
    baseline_top = option["baseline_top_ids"][0] if option["baseline_top_ids"] else ""
    experiment_top = option["experiment_top_ids"][0] if option["experiment_top_ids"] else ""
    added = set(option["added_query_tokens"])
    removed = set(option["removed_query_tokens"])
    baseline_hits = "\n".join(hit_card(hit, experiment_ids, experiment_top) for hit in option["baseline_hits"])
    experiment_hits = "\n".join(hit_card(hit, baseline_ids, baseline_top) for hit in option["experiment_hits"])
    return f"""
    <section class="case">
      <div class="case-head">
        <div>
          <h2>{esc(question['qid'])} / option {esc(option['option'])}</h2>
          <p>{esc(question['question'])}</p>
          <p class="query">{esc(option['query'].splitlines()[-1])}</p>
        </div>
        <dl>
          <div><dt>top1 changed</dt><dd>{esc(option['top1_changed'])}</dd></div>
          <div><dt>top-k overlap</dt><dd>{esc(option['topk_overlap'])}</dd></div>
          <div><dt>added tokens</dt><dd>{esc('、'.join(option['added_query_tokens']) or '-')}</dd></div>
        </dl>
      </div>
      <div class="tokens">
        <div>
          <h3>Baseline Tokens</h3>
          <div>{token_pills(option['baseline_query_tokens'], added, removed, 'baseline')}</div>
        </div>
        <div>
          <h3>Experiment Tokens</h3>
          <div>{token_pills(option['experiment_query_tokens'], added, removed, 'experiment')}</div>
        </div>
      </div>
      <div class="columns">
        <div>
          <h3>Baseline Top Hits</h3>
          {baseline_hits}
        </div>
        <div>
          <h3>Experiment Top Hits</h3>
          {experiment_hits}
        </div>
      </div>
    </section>
    """


def build_html(compare_path: Path, output_path: Path, selected_cases: list[tuple[str, str]]) -> None:
    data = json.loads(compare_path.read_text(encoding="utf-8"))
    wanted = {case_key(qid, option) for qid, option in selected_cases}
    sections = []
    for question in data["traces"]:
        for option in question["options"]:
            if case_key(question["qid"], option["option"]) in wanted:
                sections.append(render_case(question, option))

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Regulatory BM25 A/B Diff</title>
  <style>
    :root {{
      --ink: #182230;
      --muted: #667085;
      --line: #d8dee8;
      --bg: #f6f7f9;
      --panel: #fff;
      --add: #daf5df;
      --sub: #ffe1dd;
      --same: #e8f1ff;
      --changed: #fff3c4;
      --accent: #176b87;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header.page {{ padding: 28px 34px; background: #fff; border-bottom: 1px solid var(--line); position: sticky; top: 0; z-index: 2; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }}
    header.page p {{ margin: 0; color: var(--muted); }}
    main {{ padding: 22px 34px 42px; }}
    .summary {{ display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 10px; margin-bottom: 18px; }}
    .metric {{ background: #fff; border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; }}
    .metric b {{ display: block; font-size: 21px; }}
    .metric span {{ color: var(--muted); font-size: 12px; }}
    .case {{ background: #fff; border: 1px solid var(--line); border-radius: 8px; margin: 0 0 22px; overflow: hidden; }}
    .case-head {{ display: grid; grid-template-columns: 1fr 360px; gap: 16px; padding: 16px; border-bottom: 1px solid var(--line); }}
    .case h2 {{ margin: 0 0 8px; font-size: 18px; }}
    .case p {{ margin: 0 0 8px; line-height: 1.6; }}
    .query {{ color: var(--accent); font-weight: 700; }}
    dl {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 0; }}
    dl div {{ border: 1px solid var(--line); border-radius: 6px; padding: 8px; background: #fbfcfe; }}
    dt {{ color: var(--muted); font-size: 12px; }}
    dd {{ margin: 4px 0 0; font-weight: 700; word-break: break-word; }}
    .tokens {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; padding: 14px 16px; border-bottom: 1px solid var(--line); }}
    h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .token, .match {{ display: inline-block; border: 1px solid var(--line); border-radius: 5px; padding: 2px 6px; margin: 2px; font-size: 12px; background: #fff; }}
    .token.add {{ background: var(--add); border-color: #8bd49a; font-weight: 700; }}
    .token.sub {{ background: var(--sub); border-color: #ef9a91; font-weight: 700; }}
    .columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; }}
    .hit {{ border: 1px solid var(--line); border-left-width: 5px; border-radius: 6px; padding: 10px 12px; margin-bottom: 10px; background: #fff; }}
    .hit.overlap {{ border-left-color: var(--same); }}
    .hit.same-top {{ border-left-color: #7aa7e8; background: #f8fbff; }}
    .hit.changed {{ border-left-color: var(--changed); background: #fffdf3; }}
    .hit header {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; }}
    code {{ font-size: 12px; color: #475467; word-break: break-all; }}
    .doc {{ color: var(--muted); font-size: 12px; margin-top: 6px; word-break: break-all; }}
    h4 {{ margin: 8px 0; font-size: 13px; color: #344054; }}
    .hit p {{ margin: 0 0 8px; font-size: 13px; line-height: 1.65; }}
    .match {{ background: #eef6f8; }}
    @media (max-width: 1000px) {{
      header.page, main {{ padding-left: 16px; padding-right: 16px; }}
      .summary, .case-head, .tokens, .columns {{ grid-template-columns: 1fr; }}
      dl {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header class="page">
    <h1>Regulatory BM25 A/B Diff</h1>
    <p>Baseline vs selected regulatory terms. Green tokens are newly preserved whole terms; red tokens disappeared after whole-term segmentation.</p>
  </header>
  <main>
    <section class="summary">
      <div class="metric"><b>{data['summary']['question_count']}</b><span>questions</span></div>
      <div class="metric"><b>{data['summary']['option_count']}</b><span>options</span></div>
      <div class="metric"><b>{data['summary']['options_with_added_query_tokens']}</b><span>options with added tokens</span></div>
      <div class="metric"><b>{data['summary']['top1_changed_options']}</b><span>top1 changed</span></div>
      <div class="metric"><b>{data['summary']['average_topk_overlap']}</b><span>avg top-k overlap</span></div>
    </section>
    {''.join(sections)}
  </main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an HTML report for selected regulatory BM25 A/B cases.")
    parser.add_argument("--compare", type=Path, default=Path("artifacts/preprocessed/regulatory/terms/bm25_ab_compare.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/preprocessed/regulatory/terms/bm25_ab_diff_cases.html"))
    parser.add_argument("--case", action="append", default=[], help="Case as qid:option, e.g. reg_a_019:B")
    args = parser.parse_args()

    selected = []
    for item in args.case:
        qid, option = item.split(":", 1)
        selected.append((qid, option))
    build_html(args.compare, args.output, selected or DEFAULT_CASES)
    print(args.output)


if __name__ == "__main__":
    main()
