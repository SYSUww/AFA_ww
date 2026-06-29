#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import html
import json
from pathlib import Path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n\n... clipped {len(text) - limit} chars ..."


def _row(label: str, value: object) -> str:
    return f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(value or ''))}</td></tr>"


def _diff_rows(before: list[str], after: list[str], context: int) -> str:
    rows = []
    diff = difflib.unified_diff(before, after, fromfile="before", tofile="after", n=context, lineterm="")
    for line in diff:
        cls = "diff_ctx"
        if line.startswith("+") and not line.startswith("+++"):
            cls = "diff_add"
        elif line.startswith("-") and not line.startswith("---"):
            cls = "diff_sub"
        elif line.startswith("@@"):
            cls = "diff_hunk"
        rows.append(f'<div class="{cls}">{html.escape(line) or "&nbsp;"}</div>')
    return "\n".join(rows)


def _unit_cards(units: list[dict]) -> str:
    cards = []
    for unit in units:
        title_path = " / ".join(unit.get("title_path") or [])
        text = unit.get("text") or ""
        cards.append(
            """
            <article class="unit-card" data-type="{unit_type}" data-text="{search_text}">
              <header>
                <span>{unit_type}</span>
                <code>{unit_id}</code>
              </header>
              <h3>{title}</h3>
              <p>{text}</p>
            </article>
            """.format(
                unit_type=html.escape(unit.get("unit_type") or ""),
                unit_id=html.escape(unit.get("unit_id") or ""),
                title=html.escape(title_path),
                text=html.escape(text),
                search_text=html.escape((title_path + " " + text).lower()),
            )
        )
    return "\n".join(cards)


def build_html(
    *,
    doc_id: str,
    input_root: Path,
    preprocessed_root: Path,
    output_path: Path,
    raw_limit: int,
    processed_limit: int,
    context: int,
) -> None:
    documents = _read_json(preprocessed_root / "documents.json")
    units = _read_json(preprocessed_root / "units.json")
    summary = _read_json(preprocessed_root / "summary.json")

    doc_by_id = {doc["doc_id"]: doc for doc in documents}
    if doc_id not in doc_by_id:
        available = ", ".join(sorted(doc_by_id)[:20])
        raise SystemExit(f"Unknown doc_id {doc_id!r}. First available ids: {available}")

    doc = doc_by_id[doc_id]
    doc_units = [unit for unit in units if unit.get("doc_id") == doc_id]
    source_path = Path(doc["source_path"])
    if not source_path.exists() and input_root:
        source_path = input_root / (doc.get("source_bucket") or "") / Path(doc["source_path"]).name
    raw_text = source_path.read_text(encoding="utf-8")
    processed_text = "\n\n".join(unit.get("text") or "" for unit in doc_units)

    raw_lines = raw_text.splitlines()
    processed_lines = processed_text.splitlines()
    html_diff = difflib.HtmlDiff(wrapcolumn=100).make_table(
        raw_lines,
        processed_lines,
        fromdesc="Before: extracted md",
        todesc="After: preprocessed units",
        context=True,
        numlines=context,
    )
    unit_types = sorted({unit.get("unit_type") or "" for unit in doc_units})
    option_html = "\n".join(f'<option value="{html.escape(t)}">{html.escape(t)}</option>' for t in unit_types)

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Regulatory Compare - {html.escape(doc_id)}</title>
  <style>
    :root {{
      --ink: #18212f;
      --muted: #667085;
      --line: #d9e0ea;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --add: #d7f4df;
      --sub: #ffe0dc;
      --hunk: #fff1ba;
      --accent: #1d6f91;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }}
    header.hero {{ padding: 28px 34px 18px; border-bottom: 1px solid var(--line); background: #fff; position: sticky; top: 0; z-index: 5; }}
    h1 {{ margin: 0 0 12px; font-size: 24px; line-height: 1.2; letter-spacing: 0; }}
    .metrics {{ display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 10px; }}
    .metric {{ border: 1px solid var(--line); background: #fbfcfe; padding: 10px 12px; border-radius: 6px; }}
    .metric b {{ display: block; font-size: 20px; }}
    .metric span {{ color: var(--muted); font-size: 12px; }}
    main {{ padding: 18px 34px 40px; }}
    .tabs {{ display: flex; gap: 8px; margin: 0 0 14px; flex-wrap: wrap; }}
    .tabs button {{ border: 1px solid var(--line); background: #fff; border-radius: 6px; padding: 8px 12px; cursor: pointer; }}
    .tabs button.active {{ border-color: var(--accent); color: var(--accent); font-weight: 700; }}
    .panel {{ display: none; }}
    .panel.active {{ display: block; }}
    .grid {{ display: grid; grid-template-columns: 360px 1fr; gap: 16px; align-items: start; }}
    table.meta {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); }}
    table.meta th {{ width: 110px; color: var(--muted); text-align: right; background: #f3f6fa; }}
    table.meta th, table.meta td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); vertical-align: top; font-size: 13px; }}
    .diffbox, .text-pane, .units {{ border: 1px solid var(--line); background: #fff; border-radius: 6px; overflow: hidden; }}
    .diffbox {{ max-height: 72vh; overflow: auto; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; line-height: 1.55; }}
    .diffbox div {{ white-space: pre-wrap; padding: 1px 10px; border-bottom: 1px solid rgba(0,0,0,0.03); }}
    .diff_add {{ background: var(--add); font-weight: 700; }}
    .diff_sub {{ background: var(--sub); font-weight: 700; }}
    .diff_hunk {{ background: var(--hunk); color: #6b5300; font-weight: 700; }}
    .diff_ctx {{ color: #344054; }}
    .html-diff-wrap {{ overflow: auto; max-height: 72vh; border: 1px solid var(--line); background: #fff; border-radius: 6px; }}
    .html-diff-wrap table {{ border-collapse: collapse; width: 100%; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
    .html-diff-wrap td, .html-diff-wrap th {{ padding: 3px 6px; border: 1px solid #e6ebf2; vertical-align: top; }}
    .diff_add, .diff_sub, .diff_chg {{ border-radius: 2px; }}
    .diff_chg {{ background: #fff1ba; font-weight: 700; }}
    .tools {{ display: flex; gap: 10px; margin: 0 0 12px; }}
    .tools input, .tools select {{ padding: 8px 10px; border: 1px solid var(--line); border-radius: 6px; min-width: 180px; }}
    .unit-list {{ display: grid; gap: 10px; max-height: 72vh; overflow: auto; padding: 10px; }}
    .unit-card {{ border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; background: #fff; }}
    .unit-card header {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; color: var(--muted); font-size: 12px; }}
    .unit-card h3 {{ font-size: 14px; margin: 8px 0; line-height: 1.35; }}
    .unit-card p {{ margin: 0; line-height: 1.7; font-size: 14px; }}
    .twocol {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .text-pane h2 {{ font-size: 15px; margin: 0; padding: 10px 12px; border-bottom: 1px solid var(--line); background: #f3f6fa; }}
    pre {{ margin: 0; padding: 12px; white-space: pre-wrap; word-break: break-word; max-height: 72vh; overflow: auto; font-size: 12px; line-height: 1.65; }}
    @media (max-width: 900px) {{
      .metrics, .grid, .twocol {{ grid-template-columns: 1fr; }}
      header.hero, main {{ padding-left: 16px; padding-right: 16px; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <h1>{html.escape(doc.get("title") or doc_id)}</h1>
    <div class="metrics">
      <div class="metric"><b>{html.escape(doc_id)}</b><span>doc id</span></div>
      <div class="metric"><b>{html.escape(doc.get("doc_type") or "")}</b><span>doc type</span></div>
      <div class="metric"><b>{len(raw_lines)}</b><span>raw lines</span></div>
      <div class="metric"><b>{len(doc_units)}</b><span>processed units</span></div>
      <div class="metric"><b>{summary.get("unit_count", "")}</b><span>corpus units</span></div>
    </div>
  </header>
  <main>
    <nav class="tabs">
      <button class="active" data-tab="context">Context Diff</button>
      <button data-tab="html">Side-by-side Diff</button>
      <button data-tab="units">Processed Units Explorer</button>
      <button data-tab="raw">Raw vs Processed Text</button>
    </nav>

    <section id="context" class="panel active">
      <div class="grid">
        <table class="meta">
          {_row("doc_id", doc_id)}
          {_row("source_bucket", doc.get("source_bucket"))}
          {_row("doc_type", doc.get("doc_type"))}
          {_row("agency", doc.get("agency"))}
          {_row("doc_no", doc.get("doc_no"))}
          {_row("publish_date", doc.get("publish_date"))}
          {_row("effective_date", doc.get("effective_date"))}
          {_row("source_path", doc.get("source_path"))}
        </table>
        <div class="diffbox">{_diff_rows(raw_lines, processed_lines, context)}</div>
      </div>
    </section>

    <section id="html" class="panel">
      <div class="html-diff-wrap">{html_diff}</div>
    </section>

    <section id="units" class="panel">
      <div class="tools">
        <input id="unitSearch" type="search" placeholder="搜索处理后的单元">
        <select id="unitType"><option value="">全部类型</option>{option_html}</select>
      </div>
      <div class="units"><div id="unitList" class="unit-list">{_unit_cards(doc_units)}</div></div>
    </section>

    <section id="raw" class="panel">
      <div class="twocol">
        <div class="text-pane"><h2>Before</h2><pre>{html.escape(_clip(raw_text, raw_limit))}</pre></div>
        <div class="text-pane"><h2>After</h2><pre>{html.escape(_clip(processed_text, processed_limit))}</pre></div>
      </div>
    </section>
  </main>
  <script>
    const buttons = [...document.querySelectorAll('.tabs button')];
    const panels = [...document.querySelectorAll('.panel')];
    buttons.forEach(btn => btn.addEventListener('click', () => {{
      buttons.forEach(b => b.classList.toggle('active', b === btn));
      panels.forEach(p => p.classList.toggle('active', p.id === btn.dataset.tab));
    }}));
    const search = document.getElementById('unitSearch');
    const type = document.getElementById('unitType');
    const cards = [...document.querySelectorAll('.unit-card')];
    function filterUnits() {{
      const q = search.value.trim().toLowerCase();
      const t = type.value;
      cards.forEach(card => {{
        const okType = !t || card.dataset.type === t;
        const okText = !q || card.dataset.text.includes(q);
        card.style.display = okType && okText ? '' : 'none';
      }});
    }}
    search.addEventListener('input', filterUnits);
    type.addEventListener('change', filterUnits);
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a regulatory before/after comparison HTML page.")
    parser.add_argument("doc_id", help="Document id from preprocessed regulatory documents.json")
    parser.add_argument("--input-root", type=Path, default=Path("artifacts/extracted/regulatory"))
    parser.add_argument("--preprocessed-root", type=Path, default=Path("artifacts/preprocessed/regulatory"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-limit", type=int, default=90000)
    parser.add_argument("--processed-limit", type=int, default=90000)
    parser.add_argument("--context", type=int, default=4)
    args = parser.parse_args()

    output = args.output or args.preprocessed_root / f"compare_{args.doc_id}.html"
    build_html(
        doc_id=args.doc_id,
        input_root=args.input_root,
        preprocessed_root=args.preprocessed_root,
        output_path=output,
        raw_limit=args.raw_limit,
        processed_limit=args.processed_limit,
        context=args.context,
    )
    print(output)


if __name__ == "__main__":
    main()
