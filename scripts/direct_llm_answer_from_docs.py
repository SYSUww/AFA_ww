#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afa_agent.client import OpenAICompatibleClient, extract_json_object
from afa_agent.config import build_run_config
from afa_agent.models import TokenUsage


DOMAINS = ["regulatory", "financial_reports", "insurance", "research", "financial_contracts"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_questions(domain: str, split: str) -> list[dict[str, Any]]:
    manifest = read_json(ROOT / "artifacts" / "manifest" / "dataset_manifest.json")
    rows = read_json(Path(manifest["domains"][domain]["question_path"]))
    return [row for row in rows if row.get("split") == split]


def load_doc_map(domain: str, preprocessed_root: Path) -> dict[str, dict[str, str]]:
    rows = read_json(preprocessed_root / domain / "documents.json")
    doc_map: dict[str, dict[str, str]] = {}
    for row in rows:
        path = Path(row["cleaned_path"])
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        doc_map[row["doc_id"]] = {
            "path": str(path),
            "text": text,
            "source_relpath": row.get("source_relpath", ""),
        }
    return doc_map


def split_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    buffer = ""
    for part in re.split(r"(?<=[。；！？!?])\n*|\n(?=#|第[一二三四五六七八九十百零〇两\d]+[章节条])", text):
        part = part.strip()
        if not part:
            continue
        if len(part) > max_chars:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            chunks.extend(part[i : i + max_chars].strip() for i in range(0, len(part), max_chars))
            continue
        if len(buffer) + len(part) + 1 > max_chars and buffer:
            chunks.append(buffer.strip())
            buffer = part
        else:
            buffer = f"{buffer}\n{part}".strip() if buffer else part
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


def build_chunks(doc_ids: list[str], doc_map: dict[str, dict[str, str]], max_chars: int) -> list[dict[str, Any]]:
    chunks = []
    for doc_id in doc_ids:
        doc = doc_map.get(doc_id)
        if not doc:
            continue
        for idx, text in enumerate(split_text(doc["text"], max_chars=max_chars), start=1):
            chunks.append(
                {
                    "chunk_id": f"{doc_id}::chunk_{idx}",
                    "doc_id": doc_id,
                    "index": idx,
                    "text": text,
                    "chars": len(text),
                }
            )
    return chunks


STOP_TERMS = {
    "关于",
    "下列",
    "哪些",
    "说法",
    "正确",
    "错误",
    "内容",
    "结合",
    "文档",
    "陈述",
    "符合",
    "事实",
    "相比",
    "实现",
    "以及",
    "是否",
    "均为",
    "年度报告",
}


DOMAIN_FOCUS_TERMS = {
    "financial_reports": [
        "营业收入",
        "营业总收入",
        "净利润",
        "归属于上市公司股东的净利润",
        "经营活动产生的现金流量净额",
        "现金分红",
        "每10股",
        "研发投入",
        "股东回报",
        "利润分配",
    ],
    "insurance": ["保险金", "保险责任", "现金价值", "账户价值", "已交保费", "基本保额", "身故", "领取", "退保"],
    "research": ["预计", "同比", "市场规模", "渗透率", "增速", "投资建议", "风险提示", "结论"],
    "financial_contracts": ["发行人", "发行规模", "主体信用评级", "债项信用评级", "受托管理人", "违约", "回售", "赎回"],
    "regulatory": ["第", "条", "施行", "报告", "披露", "处罚", "监管", "客户尽职调查", "受益所有人"],
}


def focus_terms(question: dict[str, Any]) -> list[str]:
    text = question.get("question", "") + "\n" + "\n".join(question.get("options", {}).values())
    terms: list[str] = []
    for item in DOMAIN_FOCUS_TERMS.get(question.get("domain", ""), []):
        if item in text:
            terms.append(item)
    for item in re.findall(r"[A-Za-z0-9_.%％\-]+|[\u4e00-\u9fff]{2,12}", text):
        cleaned = item.strip()
        if not cleaned or cleaned in STOP_TERMS:
            continue
        if cleaned not in terms:
            terms.append(cleaned)
    return terms[:80]


def build_focused_chunks(
    question: dict[str, Any],
    doc_map: dict[str, dict[str, str]],
    *,
    max_chunks_per_doc: int,
    window_lines: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    terms = focus_terms(question)
    focused: list[dict[str, Any]] = []
    for doc_id in question.get("doc_ids", []):
        doc = doc_map.get(doc_id)
        if not doc:
            continue
        lines = [line.strip() for line in doc["text"].splitlines() if line.strip()]
        scored = []
        for idx, line in enumerate(lines):
            if len(line) < 3:
                continue
            score = sum(1 for term in terms if term and term in line)
            if re.search(r"20\d{2}", line):
                score += 1
            if re.search(r"\d", line) and any(unit in line for unit in ["元", "万元", "亿元", "%", "％", "级", "日"]):
                score += 1
            if score:
                scored.append((score, idx))
        scored.sort(key=lambda item: (-item[0], item[1]))
        used_spans: list[tuple[int, int]] = []
        chunk_index = 1
        for _, idx in scored:
            start = max(0, idx - window_lines)
            end = min(len(lines), idx + window_lines + 1)
            if any(not (end < a or start > b) for a, b in used_spans):
                continue
            text = "\n".join(lines[start:end]).strip()
            if len(text) > max_chars:
                text = text[:max_chars]
            focused.append(
                {
                    "chunk_id": f"{doc_id}::focus_{chunk_index}",
                    "doc_id": doc_id,
                    "index": chunk_index,
                    "text": text,
                    "chars": len(text),
                }
            )
            used_spans.append((start, end))
            chunk_index += 1
            if chunk_index > max_chunks_per_doc:
                break
        if chunk_index == 1 and lines:
            text = "\n".join(lines[: min(len(lines), window_lines * 2 + 1)])
            focused.append(
                {
                    "chunk_id": f"{doc_id}::focus_1",
                    "doc_id": doc_id,
                    "index": 1,
                    "text": text[:max_chars],
                    "chars": min(len(text), max_chars),
                }
            )
    return focused


def chunk_preview(chunk: dict[str, Any], max_preview_chars: int = 360) -> str:
    text = chunk["text"]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    headings = [line for line in lines[:80] if line.startswith("#") or re.match(r"^第[一二三四五六七八九十百零〇两\d]+[章节条]", line)]
    preview_parts = []
    if headings:
        preview_parts.append(" / ".join(headings[:6]))
    preview_parts.append(text[:max_preview_chars])
    if len(text) > max_preview_chars:
        preview_parts.append(text[-min(120, len(text)) :])
    return "\n".join(preview_parts)


def clean_answer(answer: str, question: dict[str, Any]) -> str:
    allowed = set(question.get("options", {}).keys())
    answer = "".join(ch for ch in str(answer).upper() if ch in allowed)
    if question["answer_format"] == "tf":
        return answer[:1] if answer[:1] in {"A", "B"} else "B"
    if question["answer_format"] == "mcq":
        return answer[:1] if answer[:1] in allowed else sorted(allowed)[0]
    selected = sorted(set(answer))
    for option in sorted(allowed):
        if len(selected) >= 2:
            break
        if option not in selected:
            selected.append(option)
    return "".join(sorted(selected))


def token_usage_add(total: TokenUsage, usage: TokenUsage) -> None:
    total.prompt_tokens += usage.prompt_tokens
    total.completion_tokens += usage.completion_tokens
    total.total_tokens += usage.total_tokens


class DirectDocAnswerer:
    def __init__(self, client: OpenAICompatibleClient, args: argparse.Namespace):
        self.client = client
        self.args = args

    def choose_chunks(self, question: dict[str, Any], chunks: list[dict[str, Any]], usage: TokenUsage) -> list[str]:
        if len(chunks) <= self.args.max_selected_chunks:
            return [chunk["chunk_id"] for chunk in chunks]
        if len(chunks) > self.args.chunk_select_batch_size:
            candidate_ids: list[str] = []
            for start in range(0, len(chunks), self.args.chunk_select_batch_size):
                batch = chunks[start : start + self.args.chunk_select_batch_size]
                selected = self._choose_chunks_once(
                    question,
                    batch,
                    usage,
                    max_select=min(4, self.args.max_selected_chunks),
                    instruction="这是分批粗选，请选出本批中最可能有证据的 chunk。",
                )
                for chunk_id in selected:
                    if chunk_id not in candidate_ids:
                        candidate_ids.append(chunk_id)
            chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
            candidate_chunks = [chunk_by_id[chunk_id] for chunk_id in candidate_ids if chunk_id in chunk_by_id]
            selected_ids = self._choose_chunks_once(
                question,
                candidate_chunks,
                usage,
                max_select=self.args.max_selected_chunks,
                instruction="这是粗选结果的二次精选，请保留最能回答题目的 chunk。",
            )
            return self._ensure_doc_coverage(selected_ids, chunks, question)
        selected_ids = self._choose_chunks_once(
            question,
            chunks,
            usage,
            max_select=self.args.max_selected_chunks,
            instruction="请选择最可能包含答案证据的 chunk。",
        )
        return self._ensure_doc_coverage(selected_ids, chunks, question)

    def _choose_chunks_once(
        self,
        question: dict[str, Any],
        chunks: list[dict[str, Any]],
        usage: TokenUsage,
        *,
        max_select: int,
        instruction: str,
    ) -> list[str]:
        preview_lines = []
        for chunk in chunks:
            preview_lines.append(
                f"[{chunk['chunk_id']}] doc={chunk['doc_id']} chars={chunk['chars']}\n{chunk_preview(chunk)}"
            )
        options_text = "\n".join(f"{key}. {value}" for key, value in question.get("options", {}).items())
        response = self.client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "你是长文档阅读助手。不能使用 BM25 或外部检索。"
                        "你会看到题目、选项和每个文档 chunk 的结构预览。"
                        "请选择最可能包含答案证据的 chunk_id，尽量覆盖题目涉及的每个 doc_id。只输出 JSON。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"题目ID：{question['qid']}\n题目：{question['question']}\n题型：{question['answer_format']}\n"
                        f"引用文档：{question.get('doc_ids', [])}\n选项：\n{options_text}\n\n"
                        f"{instruction}\n最多选择 {max_select} 个 chunk。"
                        "如果是多文档题，每个被引用文档至少选择一个看起来相关的 chunk。\n\n"
                        "chunk 预览：\n" + "\n\n".join(preview_lines)
                        + '\n\n输出 JSON：{"selected_chunk_ids": ["doc::chunk_1"], "selection_reason": "..."}'
                    ),
                },
            ]
        )
        token_usage_add(usage, response.token_usage)
        try:
            parsed = extract_json_object(response.content)
            selected = [str(item) for item in parsed.get("selected_chunk_ids", [])]
        except Exception:
            selected = []
        valid = {chunk["chunk_id"] for chunk in chunks}
        selected = [item for item in selected if item in valid]
        if not selected:
            selected = [chunk["chunk_id"] for chunk in chunks[:max_select]]
        return selected[:max_select]

    def _ensure_doc_coverage(
        self,
        selected_ids: list[str],
        chunks: list[dict[str, Any]],
        question: dict[str, Any],
    ) -> list[str]:
        selected = list(dict.fromkeys(selected_ids))
        selected_docs = {chunk_id.split("::")[0] for chunk_id in selected}
        by_doc: dict[str, list[str]] = {}
        for chunk in chunks:
            by_doc.setdefault(chunk["doc_id"], []).append(chunk["chunk_id"])
        for doc_id in question.get("doc_ids", []):
            if doc_id in selected_docs:
                continue
            for chunk_id in by_doc.get(doc_id, []):
                if chunk_id not in selected:
                    selected.append(chunk_id)
                    selected_docs.add(doc_id)
                    break
        if len(selected) <= self.args.max_selected_chunks:
            return selected
        required = []
        seen_docs = set()
        for chunk_id in selected:
            doc_id = chunk_id.split("::")[0]
            if doc_id in question.get("doc_ids", []) and doc_id not in seen_docs:
                required.append(chunk_id)
                seen_docs.add(doc_id)
        final = required[:]
        for chunk_id in selected:
            if len(final) >= self.args.max_selected_chunks:
                break
            if chunk_id not in final:
                final.append(chunk_id)
        return final[: self.args.max_selected_chunks]

    def extract_evidence(
        self,
        question: dict[str, Any],
        selected_chunks: list[dict[str, Any]],
        usage: TokenUsage,
    ) -> list[dict[str, Any]]:
        options_text = "\n".join(f"{key}. {value}" for key, value in question.get("options", {}).items())
        chunk_texts = []
        for chunk in selected_chunks:
            text = chunk["text"]
            if len(text) > self.args.chunk_answer_chars:
                text = text[: self.args.chunk_answer_chars] + "\n...[chunk truncated]..."
            chunk_texts.append(f"[{chunk['chunk_id']}] doc={chunk['doc_id']}\n{text}")
        response = self.client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "你是证据摘取助手。只能根据给定原文 chunk 摘取证据，不要直接作答。"
                        "证据应尽量短但足以支撑判断；如果证据来自多个文档，要分别列出。只输出 JSON。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"题目ID：{question['qid']}\n题目：{question['question']}\n题型：{question['answer_format']}\n"
                        f"选项：\n{options_text}\n\n原文 chunk：\n"
                        + "\n\n".join(chunk_texts)
                        + '\n\n输出 JSON：{"evidence": [{"doc_id": "...", "chunk_id": "...", "quote": "原文短摘录", "supports": "它能支持或反驳哪个选项/判断"}]}'
                    ),
                },
            ]
        )
        token_usage_add(usage, response.token_usage)
        try:
            parsed = extract_json_object(response.content)
            evidence = parsed.get("evidence", [])
            if isinstance(evidence, list):
                return [item for item in evidence if isinstance(item, dict)]
        except Exception:
            pass
        return []

    def answer_question(self, question: dict[str, Any], evidence: list[dict[str, Any]], usage: TokenUsage) -> dict[str, Any]:
        options_text = "\n".join(f"{key}. {value}" for key, value in question.get("options", {}).items())
        evidence_text = json.dumps(evidence, ensure_ascii=False, indent=2)
        format_hint = {
            "tf": "判断题只能回答 A 或 B，其中 A=正确，B=错误。",
            "mcq": "单选题只能回答一个选项字母。",
            "multi": "多选题必须回答两个或以上选项字母，按字母排序，不要重复。",
        }.get(question["answer_format"], "")
        response = self.client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "你是金融长文档问答专家。现在不使用 BM25，只根据题目引用文档中摘取的证据作答。"
                        "必须逐选项判断，证据不足时不要臆造。最终只输出 JSON。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"题目ID：{question['qid']}\n领域：{question['domain']}\n题目：{question['question']}\n"
                        f"题型：{question['answer_format']}。{format_hint}\n引用文档：{question.get('doc_ids', [])}\n"
                        f"选项：\n{options_text}\n\n证据：\n{evidence_text}\n\n"
                        '输出 JSON：{"answer": "AC", "reasoning_summary": "逐选项简要说明", '
                        '"evidence_used": [{"doc_id": "...", "quote": "...", "supports": "..."}]}'
                    ),
                },
            ]
        )
        token_usage_add(usage, response.token_usage)
        parsed = extract_json_object(response.content)
        answer = clean_answer(parsed.get("answer", ""), question)
        evidence_used = parsed.get("evidence_used")
        if not isinstance(evidence_used, list) or not evidence_used:
            evidence_used = evidence
        return {
            "qid": question["qid"],
            "domain": question["domain"],
            "question_type": question["answer_format"],
            "pred_answer": answer,
            "reasoning_summary": str(parsed.get("reasoning_summary", "")).strip(),
            "evidence_items": evidence_used,
            "token_usage": asdict(usage),
            "debug_meta": {
                "doc_ids": question.get("doc_ids", []),
                "direct_reading": True,
                "no_bm25": True,
            },
        }


def run_domain(domain: str, args: argparse.Namespace, client: OpenAICompatibleClient) -> list[dict[str, Any]]:
    output_dir = Path(args.output_root) / domain
    output_dir.mkdir(parents=True, exist_ok=True)
    answers_path = output_dir / "answers.json"
    questions = load_questions(domain, args.split)
    if args.limit:
        questions = questions[: args.limit]
    doc_map = load_doc_map(domain, Path(args.preprocessed_root))
    answerer = DirectDocAnswerer(client, args)
    results = read_json(answers_path) if answers_path.exists() else []
    completed = {row["qid"] for row in results}
    for question in questions:
        if question["qid"] in completed:
            continue
        raw_chunks = build_chunks(question.get("doc_ids", []), doc_map, max_chars=args.chunk_chars)
        chunks = raw_chunks
        if len(raw_chunks) > args.focused_threshold_chunks:
            focused = build_focused_chunks(
                question,
                doc_map,
                max_chunks_per_doc=args.focused_chunks_per_doc,
                window_lines=args.focused_window_lines,
                max_chars=args.focused_chunk_chars,
            )
            if focused:
                chunks = focused
        usage = TokenUsage()
        selected_ids = answerer.choose_chunks(question, chunks, usage)
        chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        selected_chunks = [chunk_by_id[item] for item in selected_ids if item in chunk_by_id]
        evidence = answerer.extract_evidence(question, selected_chunks, usage)
        result = answerer.answer_question(question, evidence, usage)
        result["debug_meta"]["selected_chunk_ids"] = selected_ids
        result["debug_meta"]["chunk_count"] = len(chunks)
        result["debug_meta"]["raw_chunk_count"] = len(raw_chunks)
        result["debug_meta"]["used_focused_chunks"] = len(chunks) != len(raw_chunks)
        results.append(result)
        write_json(answers_path, results)
    return results


def export_outputs(all_results: list[dict[str, Any]], output_root: Path) -> None:
    total_prompt = sum(row["token_usage"].get("prompt_tokens", 0) for row in all_results)
    total_completion = sum(row["token_usage"].get("completion_tokens", 0) for row in all_results)
    total_tokens = sum(row["token_usage"].get("total_tokens", 0) for row in all_results)
    with (output_root / "answer.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        writer.writeheader()
        writer.writerow(
            {
                "qid": "summary",
                "answer": "",
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_tokens,
            }
        )
        for row in all_results:
            writer.writerow(
                {
                    "qid": row["qid"],
                    "answer": row["pred_answer"],
                    "prompt_tokens": row["token_usage"].get("prompt_tokens", 0),
                    "completion_tokens": row["token_usage"].get("completion_tokens", 0),
                    "total_tokens": row["token_usage"].get("total_tokens", 0),
                }
            )
    with (output_root / "evidence.md").open("w", encoding="utf-8") as f:
        f.write("# Direct LLM Reading Evidence\n\n")
        for row in all_results:
            f.write(f"## {row['qid']} {row['pred_answer']}\n\n")
            f.write((row.get("reasoning_summary") or "").strip() + "\n\n")
            for item in row.get("evidence_items", [])[:8]:
                if not isinstance(item, dict):
                    continue
                f.write(f"- `{item.get('doc_id', '')}`: {item.get('quote', '')}  \n  {item.get('supports', '')}\n")
            f.write("\n")
    write_json(
        output_root / "summary.json",
        {
            "question_count": len(all_results),
            "token_usage": {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_tokens,
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", default=["all"])
    parser.add_argument("--split", default="A")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--preprocessed-root", default=str(ROOT / "artifacts" / "preprocessed_loop"))
    parser.add_argument("--output-root", default=str(ROOT / "artifacts" / "llm_direct_reading" / "group_a_20260628"))
    parser.add_argument("--chunk-chars", type=int, default=12000)
    parser.add_argument("--chunk-answer-chars", type=int, default=9000)
    parser.add_argument("--max-selected-chunks", type=int, default=8)
    parser.add_argument("--chunk-select-batch-size", type=int, default=12)
    parser.add_argument("--focused-threshold-chunks", type=int, default=28)
    parser.add_argument("--focused-chunks-per-doc", type=int, default=10)
    parser.add_argument("--focused-window-lines", type=int, default=4)
    parser.add_argument("--focused-chunk-chars", type=int, default=4500)
    args = parser.parse_args()
    domains = DOMAINS if args.domains == ["all"] else args.domains
    config = build_run_config()
    if not config.model:
        raise RuntimeError("Missing LLM model config")
    client = OpenAICompatibleClient(config.model)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_results = []
    for domain in domains:
        all_results.extend(run_domain(domain, args, client))
    all_results.sort(key=lambda row: row["qid"])
    write_json(output_root / "answers.json", all_results)
    export_outputs(all_results, output_root)
    print(output_root)


if __name__ == "__main__":
    main()
