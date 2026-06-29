#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


CORE_ALLOW = {
    "上市公司",
    "证券公司",
    "中国证监会",
    "中国证券监督管理委员会",
    "股东会",
    "董事会",
    "证券交易所",
    "上海证券交易所",
    "深圳证券交易所",
    "北京证券交易所",
    "投资者",
    "信息披露",
    "实际控制人",
    "控股股东",
    "行政处罚",
    "内幕信息",
    "重大资产重组",
    "基金管理人",
    "发行人",
    "处罚决定书",
    "注册会计师",
    "会计师事务所",
    "派出机构",
    "独立董事",
    "违法行为",
    "资产管理",
    "收购人",
    "信息披露义务人",
    "募集资金",
    "注册会计师审计",
    "会计师审计准则",
    "监管措施",
    "行政处罚决定书",
    "虚假记载",
    "私募基金",
    "财务报表",
    "募集说明书",
    "公众公司",
    "公募基金",
    "证券市场",
    "区域性股权市场",
    "关联交易",
    "挂牌公司",
    "公开发行证券",
    "证券发行",
    "登记结算",
    "公司法",
    "证券法",
    "中国人民银行",
    "证券投资基金",
    "证券账户",
    "定向发行",
    "可转债",
    "基金托管人",
    "私募基金管理人",
    "基金管理公司",
    "反洗钱",
    "证券期货业",
    "期货交易所",
    "期货交易",
    "交易规则",
    "股权激励计划",
    "发行对象",
    "股票期权",
    "收购报告书",
    "要约收购",
    "违法所得",
    "首次公开发行股票",
    "非公开发行",
}

ALLOW_PATTERNS = [
    r"(证券|期货|基金|债券|股票|股权|上市|挂牌|发行|收购|交易|登记|结算|账户)",
    r"(信息披露|财务报表|年度报告|募集说明书|审计|会计师|虚假记载)",
    r"(行政处罚|处罚决定书|违法|违规|内幕|操纵|监管措施|规范性文件)",
    r"(实际控制人|控股股东|独立董事|高级管理人员|董事会|股东会|表决权)",
    r"(反洗钱|客户尽职调查|受益所有人|中国人民银行|支付|账户)",
    r"(资产管理|资产支持|重大资产重组|募集资金|净资产|净资本)",
]

GENERIC_DENY = {
    "基金",
    "发行",
    "披露",
    "数据",
    "股份",
    "期货",
    "规范性",
    "会计师",
    "控制人",
    "管理人",
    "基金管理",
    "实际控制",
    "控股股东实际",
    "董事高级管理人员",
    "监事高级管理人员",
    "董事监事高级",
    "董事高级",
    "董事监事",
    "股东实际",
    "法律行政法规",
    "处罚决定书",
    "公开发行",
    "年度报告",
    "法律法规",
    "收购",
    "报告书",
    "股权",
    "要求",
    "对象",
    "方式",
    "影响",
    "公开",
    "市场",
    "措施",
    "准则",
    "转让",
    "计划",
    "权益",
    "统计",
    "数量",
    "结算",
    "登记",
    "评估",
    "分类",
    "履行",
    "负责人",
    "法律",
    "违法",
    "指标",
    "情形",
    "活动",
    "产品",
    "债券",
    "事务所",
    "内幕",
    "服务机构",
    "委员会",
    "管理公司",
    "监事高级",
    "控股",
    "准则第",
    "审计准则第",
    "对上市公司",
    "公司信息披露",
    "证券登记",
    "及其他",
    "公募基金管理",
    "资产支持",
    "期内",
    "本次",
    "证券投资",
    "发行证券",
    "证券交易",
    "证券期货",
    "证券服务机构",
    "证券服务",
    "北京证券",
    "中国证券",
    "公司债券",
    "公司信息披露内容",
    "发行股票",
    "本次发行",
    "交易行为",
    "公司股票",
    "根据当事人违法行为",
    "构成证券法",
    "规定构成证券法",
    "年证券法",
    "程度依据证券法",
    "日中国证券监督管理委员会",
    "对象发行",
    "上市公司向",
    "上市公司股份",
    "上市公司董事",
    "与上市公司",
    "公司董事会",
    "结算有限责任",
    "私募基金管理",
    "数据类型",
    "数据元",
    "支持证券",
    "年年度报告",
    "证券基金经营",
    "依据中华人民共和国证券法",
    "非上市公众",
    "当事人违法行为",
    "规范性文件深圳证券交易所",
    "基金经营",
    "号规范性文件",
    "文件深圳证券交易所",
    "行政处罚委员会",
    "中国证券登记",
    "简称证券法",
    "作出行政处罚",
    "了作出行政处罚",
    "收购公司",
    "本次交易",
    "履行信息披露",
    "虚假记载误导性",
    "证券基金",
    "会计师审计",
    "法律行政法规",
    "投资基金",
    "资产支持",
    "公开发行",
    "年度报告",
    "法律法规",
}

FRAGMENT_DENY = (
    "以下",
    "以上",
    "第",
    "的",
    "应当",
    "及其",
    "以及",
    "其他",
    "本处罚",
    "收到本",
    "之日起",
    "复印件",
    "寄送",
    "送中国",
    "内向",
    "日内",
    "名称中国",
    "称中国",
    "至中国",
    "申请",
    "法治",
    "办公室",
    "文件上海",
    "内容与",
    "依据证券法",
    "中华人民共和国证券法",
)

CATEGORY_RULES = [
    ("监管机构", ("证监会", "中国证券监督管理委员会", "中国人民银行", "交易所", "派出机构")),
    ("法律规则", ("证券法", "公司法", "规范性文件", "审计准则", "交易规则")),
    ("机构主体", ("公司", "交易所", "证监会", "中国人民银行", "投资者", "发行人", "收购人", "管理人", "托管人", "会计师", "董事", "股东", "控制人", "义务人")),
    ("披露审计", ("信息披露", "财务报表", "年度报告", "募集说明书", "审计", "虚假记载")),
    ("处罚执法", ("行政处罚", "处罚决定书", "违法", "违规", "内幕", "监管措施", "规范性文件")),
    ("发行交易", ("发行", "交易", "登记结算", "证券账户", "可转债", "债券", "股票", "股权")),
    ("基金资管", ("基金", "资产管理", "资产支持", "募集资金", "重大资产重组")),
    ("公司治理", ("董事会", "股东会", "表决权", "实际控制人", "控股股东", "独立董事", "高级管理人员")),
    ("反洗钱支付", ("反洗钱", "客户尽职调查", "受益所有人", "支付")),
]


def allowed(term: str) -> bool:
    if term in CORE_ALLOW:
        return True
    if term in GENERIC_DENY:
        return False
    if len(term) < 4:
        return False
    if any(fragment in term for fragment in FRAGMENT_DENY):
        return False
    if "中国证券监督管理委员会" in term and term != "中国证券监督管理委员会":
        return False
    if "上海证券交易所" in term and term != "上海证券交易所":
        return False
    if term.startswith(("和", "与", "对", "向", "日", "称", "送", "至")):
        return False
    if term.endswith(("与", "向", "实际", "高级", "有限责任")):
        return False
    if re.search(r"\d{4}年|^\d|[0-9]$", term):
        return False
    return any(re.search(pattern, term) for pattern in ALLOW_PATTERNS)


def category_for(term: str) -> str:
    for category, needles in CATEGORY_RULES:
        if any(needle in term for needle in needles):
            return category
    return "其他"


def main() -> None:
    parser = argparse.ArgumentParser(description="Select dictionary-ready regulatory terms from mined candidates.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/preprocessed/regulatory/terms/regulatory_terms_candidates.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/preprocessed/regulatory/terms"),
    )
    parser.add_argument("--top-k", type=int, default=220)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    selected = []
    seen = set()
    for row in rows:
        term = row["term"].strip()
        if term in seen or not allowed(term):
            continue
        seen.add(term)
        row = dict(row)
        row["category"] = category_for(term)
        selected.append(row)
        if len(selected) >= args.top_k:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "regulatory_terms_selected.csv"
    json_path = args.output_dir / "regulatory_terms_selected.json"
    txt_path = args.output_dir / "regulatory_terms_selected.txt"
    summary_path = args.output_dir / "selected_summary.json"

    fieldnames = [
        "rank",
        "term",
        "category",
        "score",
        "freq",
        "doc_freq",
        "unit_types",
        "doc_types",
        "already_in_domain_terms",
        "example",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(selected, start=1):
            out = {name: row.get(name, "") for name in fieldnames}
            out["rank"] = idx
            writer.writerow(out)

    json_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text("\n".join(row["term"] for row in selected) + "\n", encoding="utf-8")

    categories: dict[str, int] = {}
    for row in selected:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    summary_path.write_text(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "selected_count": len(selected),
                "top_k": args.top_k,
                "categories": categories,
                "outputs": {
                    "csv": str(csv_path.resolve()),
                    "json": str(json_path.resolve()),
                    "txt": str(txt_path.resolve()),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
