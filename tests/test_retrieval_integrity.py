from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from afa_agent.domains.generic_retriever import GenericBM25Retriever
from afa_agent.domains.financial_reports.solver import FinancialReportsSolver
from afa_agent.domains.insurance.plugin import InsurancePlugin
from afa_agent.domains.insurance.solver import InsuranceSolver
from afa_agent.domains.regulatory.retriever import RegulatoryRetriever
from afa_agent.domains.regulatory.solver import RegulatorySolver
from afa_agent.domains.research.plugin import ResearchPlugin
from afa_agent.io_utils import read_json, write_json
from afa_agent.models import Question


def make_unit(unit_id: str, doc_id: str, text: str, unit_type: str = "paragraph") -> dict[str, object]:
    return {
        "unit_id": unit_id,
        "doc_id": doc_id,
        "domain": "test",
        "unit_type": unit_type,
        "title_path": [doc_id],
        "text": text,
        "page_refs": [],
        "parent_unit_id": None,
        "metadata": {},
    }


def make_metric(unit_id: str, doc_id: str, text: str, metric_name: str) -> dict[str, object]:
    unit = make_unit(unit_id, doc_id, text, unit_type="metric_row")
    unit["metadata"] = {"metric_name": metric_name, "year": "2025"}
    return unit


class NeighborExpansionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.units = [
            make_unit("doc-a::sec-1", "doc-a", "unique-target"),
            make_unit("doc-b::sec-1", "doc-b", "unrelated-boundary"),
        ]

    def test_generic_neighbor_expansion_does_not_cross_document_boundary(self) -> None:
        hits = GenericBM25Retriever(self.units).search(
            ["doc-a", "doc-b"],
            "unique-target",
            top_k=2,
            expand_neighbors=True,
        )
        self.assertEqual([hit.unit_id for hit in hits], ["doc-a::sec-1"])

    def test_regulatory_neighbor_expansion_does_not_cross_document_boundary(self) -> None:
        hits = RegulatoryRetriever(self.units).search(
            ["doc-a", "doc-b"],
            "unique-target",
            top_k=2,
            expand_neighbors=True,
        )
        self.assertEqual([hit.unit_id for hit in hits], ["doc-a::sec-1"])


class UnitIdGuardTests(unittest.TestCase):
    def test_retrievers_reject_duplicate_unit_ids(self) -> None:
        duplicate_units = [
            make_unit("duplicate", "doc-a", "first"),
            make_unit("duplicate", "doc-a", "second"),
        ]
        for retriever_class in (GenericBM25Retriever, RegulatoryRetriever):
            with self.subTest(retriever=retriever_class.__name__):
                with self.assertRaisesRegex(ValueError, "duplicate unit_id"):
                    retriever_class(duplicate_units)

    def test_plugins_reject_duplicate_units_when_building_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parsed_path = root / "parsed.json"
            duplicate_units = [
                make_unit("duplicate", "doc-a", "first"),
                make_unit("duplicate", "doc-a", "second"),
            ]
            write_json(parsed_path, {"documents": [], "units": duplicate_units})
            for plugin in (ResearchPlugin(), InsurancePlugin()):
                with self.subTest(plugin=plugin.name):
                    with self.assertRaisesRegex(ValueError, "duplicate unit_id"):
                        plugin.build_index(parsed_path, root / f"{plugin.name}.json")


class InsuranceTargetedClauseTests(unittest.TestCase):
    def test_escape_wording_maps_to_traffic_hit_and_run_clause(self) -> None:
        units = [
            make_unit(
                "za::escape",
                "za",
                "责任免除：驾驶人或操作人员交通肇事逃逸，保险人不负责赔偿。",
                unit_type="clause_block",
            ),
            make_unit("za::other", "za", "特种车保险的其他条款。", unit_type="clause_block"),
        ]
        solver = InsuranceSolver.__new__(InsuranceSolver)
        solver.retriever = GenericBM25Retriever(units)
        question = Question(
            qid="ins_b_007",
            domain="insurance",
            split="B",
            question="同等交通事故逃逸情形是否属于责任免除？",
            options={"A": "众安特种车商业保险明确列为责任免除"},
            answer_format="mcq",
            type="单选题",
            doc_ids=["za"],
        )

        hits = solver._targeted_clause_hits(question, question.question)

        self.assertEqual([hit.unit_id for hit in hits], ["za::escape"])
        self.assertTrue(hits[0].metadata["targeted_clause"])
        self.assertIn("交通肇事逃逸", solver._focus_terms(question.question))


class FinancialReportMetricBundleTests(unittest.TestCase):
    @staticmethod
    def make_solver(units: list[dict[str, object]]) -> FinancialReportsSolver:
        solver = FinancialReportsSolver.__new__(FinancialReportsSolver)
        solver.units = units
        solver.metric_index = solver._build_metric_index(units)
        return solver

    def test_operating_metric_bundle_covers_revenue_cash_flow_and_eps_roles(self) -> None:
        units = [
            make_metric(
                "catl::revenue", "annual_catl_2025_report",
                "营业收入 | 423,701,834 | 362,012,554 | 17.04% | 400,917,045", "营业收入",
            ),
            make_metric(
                "catl::cash_eps", "annual_catl_2025_report",
                "经营活动产生的现金流量净额 | 133,219,982 | 96,990,345 | 37.35% | 92,826,124\n基本每股收益(元/股) | 16.14 | 11.58 | 39.38% | 10.06",
                "经营活动产生的现金流量净额",
            ),
            make_metric(
                "midea::revenue", "annual_midea_2025_report",
                "营业收入 | 456,451,731 | 407,149,600 | 12.11% | 372,037,280", "营业收入",
            ),
            make_metric(
                "midea::cash_eps", "annual_midea_2025_report",
                "经营活动产生的现金流量净额 | 53,345,930 | 60,511,572 | -11.84% | 57,902,611\n基本每股收益(元/股) | 5.80 | 5.44 | 6.62% | 4.93",
                "经营活动产生的现金流量净额",
            ),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "两家公司 2025 年营业收入均同比增长",
            "B": "宁德时代经营现金流率上升，而美的集团经营现金流率下降",
            "C": "2025 年宁德时代经营现金流率比美的集团高约 19.75 个百分点",
            "D": "宁德时代基本每股收益同比增幅比美的集团高约 32.76 个百分点",
        }
        question = Question(
            qid="fin_b_003", domain="financial_reports", split="B",
            question="根据宁德时代与美的集团年度报告中的营业收入、经营现金流和基本每股收益。",
            options=options, answer_format="multi", type="多选题",
            doc_ids=["annual_catl_2025_report", "annual_midea_2025_report"],
        )
        labels = {key: solver._choice_metric_bundle_rule(question, option)[0] for key, option in options.items()}
        self.assertEqual(labels, {"A": True, "B": True, "C": True, "D": True})

    def test_dividend_bundle_uses_full_year_and_normalizes_per_share(self) -> None:
        units = [
            make_metric(
                "catl::dividend", "annual_catl_2025_report",
                "2025年度利润分配预案：向全体股东每10股派发现金分红69.57元（含税）。", "现金分红",
            ),
            make_metric(
                "midea::dividend", "annual_midea_2025_report",
                "公司2025年度利润分配方案为：每10股派发现金43元；年末每10股派发现金分红38元。", "现金分红",
            ),
            make_metric(
                "cmb::dividend", "annual_cmb_2025_report",
                "2025年度现金股息，全年每股现金分红2.016元（含税）。", "现金分红",
            ),
            make_metric(
                "cscec::dividend", "annual_cscec_2025_report",
                "每10股派息数(元)(含税) | 2.718", "每10股派",
            ),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "按每 10 股全年现金分红由高到低排序为：宁德时代、美的集团、招商银行、中国建筑",
            "B": "美的集团 2025 年全年每 10 股现金分红为 38 元",
            "C": "招商银行全年每股现金分红 2.016 元，等价于每 10 股 20.16 元",
            "D": "宁德时代与美的集团的全年每 10 股现金分红相差 26.57 元",
        }
        question = Question(
            qid="fin_b_005", domain="financial_reports", split="B",
            question="根据四家公司2025年年度报告中的现金分红数据。",
            options=options, answer_format="multi", type="多选题",
            doc_ids=[
                "annual_catl_2025_report", "annual_midea_2025_report",
                "annual_cmb_2025_report", "annual_cscec_2025_report",
            ],
        )
        labels = {key: solver._choice_metric_bundle_rule(question, option)[0] for key, option in options.items()}
        self.assertEqual(labels, {"A": True, "B": False, "C": True, "D": True})

    def test_statement_scope_bundle_distinguishes_consolidated_and_parent_rows(self) -> None:
        units = [
            make_metric(
                "midea::scope_revenue", "annual_midea_2025_report",
                "项目 | 2025年度合并 | 2024年度合并 | 2025年度公司 | 2024年度公司\n其中:营业收入 | 456,451,731 | 407,149,600 | 936,519 | 946,607",
                "营业收入",
            ),
            make_unit(
                "midea::scope_cash", "annual_midea_2025_report",
                "经营活动产生/(使用)的现金流量净额 | 四(64)(h) | 53,345,930 | 60,511,572 | (11,628,058) | 4,645,875",
            ),
            make_metric(
                "midea::eps", "annual_midea_2025_report",
                "经营活动产生的现金流量净额 | 53,345,930 | 60,511,572 | -11.84%\n基本每股收益(元/股) | 5.80 | 5.44 | 6.62%",
                "经营活动产生的现金流量净额",
            ),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "合并口径营业收入同比增长，而母公司口径营业收入同比下降",
            "B": "合并口径经营活动现金流量净额为正，而母公司口径为负",
            "C": "5.80 元的基本每股收益为母公司单体财务报表指标",
            "D": "母公司 2025 年经营活动产生的现金流量净额为 53,345,930 千元",
        }
        question = Question(
            qid="fin_b_012", domain="financial_reports", split="B",
            question="根据美的集团2025年年度报告中的合并财务报表与母公司财务报表。",
            options=options, answer_format="multi", type="多选题", doc_ids=["annual_midea_2025_report"],
        )
        labels = {key: solver._choice_metric_bundle_rule(question, option)[0] for key, option in options.items()}
        self.assertEqual(labels, {"A": True, "B": True, "C": False, "D": False})

    def test_byd_regional_bundle_reconciles_cross_year_amounts(self) -> None:
        units = [
            make_unit(
                "byd::region", "annual_byd_2025_report",
                "地区信息\n营业收入2025年 | 2024年\n"
                "中国(包括港澳台地区) | 493,223,970 | 555,217,682\n"
                "境外 | 310,740,988 | 221,884,773\n"
                "合计 | 803,964,958 | 777,102,455",
            )
        ]
        solver = self.make_solver(units)
        options = {
            "A": "2025 年境外收入占营业收入的比重较 2024 年提高约 10.10 个百分点",
            "B": "2025 年境外收入占比相较 2024 年的相对增幅约为 10.10%",
            "C": "2025 年境外收入增加额大于中国（包括港澳台地区）收入减少额",
            "D": "境外收入增加额减去中国（包括港澳台地区）收入减少额，与公司营业收入增加额基本一致",
        }
        question = Question(
            qid="fin_b_001", domain="financial_reports", split="B",
            question="查阅比亚迪 2024 年和 2025 年年度报告中的分地区营业收入。",
            options=options, answer_format="multi", type="多选题",
            doc_ids=["annual_byd_2024_report", "annual_byd_2025_report"],
        )
        labels = {key: solver._choice_metric_bundle_rule(question, option)[0] for key, option in options.items()}
        self.assertEqual(labels, {"A": True, "B": False, "C": True, "D": True})

    def test_byd_cross_year_bundle_uses_raw_amounts_for_ratios(self) -> None:
        units = [
            make_metric(
                "byd::main", "annual_byd_2025_report",
                "营业收入(元) | 803,964,958,000.00 | 777,102,455,000.00 | 3.46%\n"
                "归属于上市公司股东的净利润(元) | 32,619,022,000.00 | 40,254,346,000.00 | -18.97%",
                "营业收入",
            ),
            make_metric(
                "byd::cash", "annual_byd_2025_report",
                "经营活动产生的现金流量净额(元) | 59,135,544,000.00 | 133,453,873,000.00 | -55.69%",
                "经营活动产生的现金流量净额",
            ),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "2025 年归母净利率较 2024 年的相对降幅约为 1.12%",
            "B": "2025 年归母净利润同比下降约 18.97%",
            "C": "2025 年经营活动现金流量净额同比下降约 55.69%",
            "D": "经营活动现金流量净额占营业收入的比例由约 17.17% 降至约 7.36%，下降约 9.82 个百分点",
        }
        question = Question(
            qid="fin_b_002", domain="financial_reports", split="B",
            question="结合比亚迪 2024 年和 2025 年年度报告中的营业收入、归属于上市公司股东的净利润及经营活动产生的现金流量净额。",
            options=options, answer_format="multi", type="多选题",
            doc_ids=["annual_byd_2024_report", "annual_byd_2025_report"],
        )
        labels = {key: solver._choice_metric_bundle_rule(question, option)[0] for key, option in options.items()}
        self.assertEqual(labels, {"A": False, "B": True, "C": True, "D": True})

    def test_cscec_bundle_uses_original_2024_disclosure_basis(self) -> None:
        units = [
            make_unit(
                "cscec::main", "annual_cscec_2025_report",
                "主要会计数据 | 2025年 | 2024年 | 调整后 | 调整前\n"
                "营业收入 | 2,082,141,811 | 2,187,334,286 | 2,187,147,839 | -4.8\n"
                "归属于上市公司股东的净利润 | 39,069,002 | 46,193,694 | 46,187,099 | -15.4\n"
                "经营活动产生的现金流量净额 | 20,537,132 | 15,825,793 | 15,773,535 | 29.8",
            ),
            make_unit(
                "cscec::eps", "annual_cscec_2025_report",
                "主要财务指标 | 2025年 | 2024年 | 调整后 | 调整前\n"
                "基本每股收益(元/股) | 0.94 | 1.11 | 1.11 | -15.3",
            ),
            make_unit(
                "cscec::dividend-2025", "annual_cscec_2025_report",
                "本年度公司现金分红占合并报表归属于上市公司股东净利润的比例为28.75%。",
            ),
            make_unit(
                "cscec::dividend-2024", "annual_cscec_2024_report",
                "本年度公司现金分红占合并报表归属于上市公司股东净利润的比例为24.29%。",
            ),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "2025 年现金分红占归母净利润比例较 2024 年提高 4.46 个百分点",
            "B": "2025 年经营活动现金流量净额占营业收入的比例超过 1%",
            "C": "2025 年基本每股收益降幅约为 15.32%，与归母净利润约 15.41% 的降幅接近",
            "D": "2025 年归母净利润减少额小于经营活动现金流量净额增加额",
        }
        question = Question(
            qid="fin_b_008", domain="financial_reports", split="B",
            question="根据中国建筑 2024 年和 2025 年年度报告（2024年数据采用2024年年报原始披露值）。",
            options=options, answer_format="multi", type="多选题",
            doc_ids=["annual_cscec_2025_report", "annual_cscec_2024_report"],
        )
        labels = {key: solver._choice_metric_bundle_rule(question, option)[0] for key, option in options.items()}
        self.assertEqual(labels, {"A": True, "B": False, "C": True, "D": False})

    def test_solvency_bundle_compares_three_companies(self) -> None:
        units = [
            make_unit(
                "byd::solvency", "annual_byd_2025_report",
                "流动比率 | 0.79 | 0.75 | 5.33%\n资产负债率 | 70.74% | 74.64% | -3.90%\n"
                "速动比率 | 0.42 | 0.47 | -10.64%",
            ),
            make_unit(
                "catl::solvency", "annual_catl_2025_report",
                "流动比率 | 1.60 | 1.61 | -0.62%\n资产负债率 | 61.94% | 65.24% | -3.30%\n"
                "速动比率 | 1.36 | 1.42 | -4.23%",
            ),
            make_unit(
                "midea::solvency", "annual_midea_2025_report",
                "流动比率 | 121.34% | 110.59% | 10.75%\n资产负债率 | 61.17% | 62.33% | -1.16%\n"
                "速动比率 | 94.60% | 85.94% | 8.66%",
            ),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "三家公司 2025 年资产负债率均较 2024 年下降",
            "B": "按 2025 年资产负债率由低到高排序为：美的集团、宁德时代、比亚迪",
            "C": "比亚迪 2025 年流动比率和速动比率均较 2024 年上升",
            "D": "宁德时代 2025 年流动比率和速动比率均较 2024 年上升",
        }
        question = Question(
            qid="fin_b_004", domain="financial_reports", split="B",
            question="比较比亚迪、宁德时代和美的集团 2024 年、2025 年年度报告中的资产负债率、流动比率和速动比率。",
            options=options, answer_format="multi", type="多选题",
            doc_ids=["annual_byd_2025_report", "annual_catl_2025_report", "annual_midea_2025_report"],
        )
        labels = {key: solver._choice_metric_bundle_rule(question, option)[0] for key, option in options.items()}
        self.assertEqual(labels, {"A": True, "B": True, "C": False, "D": False})

    def test_solvency_bundle_preserves_ratio_units_across_three_years(self) -> None:
        units = [
            make_unit(
                "byd::2025-balance", "annual_byd_2025_report",
                "流动比率 | 0.79 | 0.75\n资产负债率 | 70.74% | 74.64%\n速动比率 | 0.42 | 0.47",
            ),
            make_unit(
                "byd::2025-interest", "annual_byd_2025_report",
                "利息保障倍数 | 16.58 | 24.73\n现金利息保障倍数 | 33.38 | 88.70",
            ),
            make_unit(
                "byd::2024-interest", "annual_byd_2024_report",
                "利息保障倍数 | 24.73 | 21.39\n现金利息保障倍数 | 88.70 | 128.98",
            ),
            make_unit(
                "catl::2025-balance", "annual_catl_2025_report",
                "流动比率 | 1.60 | 1.61\n资产负债率 | 61.94% | 65.24%\n速动比率 | 1.36 | 1.42",
            ),
            make_unit(
                "catl::2025-interest", "annual_catl_2025_report",
                "利息保障倍数 | 31.95 | 16.16\n现金利息保障倍数 | 54.51 | 28.61",
            ),
            make_unit(
                "catl::2024-interest", "annual_catl_2024_report",
                "流动比率 | 1.61 | 1.57\n资产负债率 | 65.24% | 69.34%\n速动比率 | 1.42 | 1.41\n"
                "利息保障倍数 | 16.16 | 15.35\n现金利息保障倍数 | 28.61 | 27.47",
            ),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "宁德时代资产负债率连续下降，利息保障倍数连续上升",
            "B": "比亚迪现金利息保障倍数由 2023 年的 128.98 降至 2025 年的 33.38，共下降约 74.12 个百分点",
            "C": "宁德时代利息保障倍数 2025 年较 2023 年提高约 108.14 个百分点",
            "D": "比亚迪 2025 年资产负债率下降，但利息保障倍数和现金利息保障倍数均较 2024 年下降",
        }
        question = Question(
            qid="fin_b_010", domain="financial_reports", split="B",
            question="根据比亚迪与宁德时代 2023—2025 年偿债指标。",
            options=options, answer_format="multi", type="多选题",
            doc_ids=[
                "annual_byd_2024_report", "annual_byd_2025_report",
                "annual_catl_2024_report", "annual_catl_2025_report",
            ],
        )
        labels = {key: solver._choice_metric_bundle_rule(question, option)[0] for key, option in options.items()}
        self.assertEqual(labels, {"A": True, "B": False, "C": False, "D": True})

    def test_research_expense_rate_bundle_replays_amount_and_rate_changes(self) -> None:
        units = [
            make_unit(
                "catl::research", "annual_catl_2025_report",
                "项目 | 2025年 | 2024年 | 2023年\n"
                "研发投入金额(千元) | 22,146,581 | 18,606,756 | 18,356,108\n"
                "研发投入占营业收入比例 | 5.23% | 5.14% | 4.58%",
            ),
            make_unit(
                "midea::research", "annual_midea_2025_report",
                "研发费用金额(千元) | 17,787,624 | 16,232,771 | 9.58%\n"
                "研发费用占营业收入比例 | 3.90% | 3.99% | -0.09%",
            ),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "两家公司 2025 年研发费用占营业收入比例均较 2024 年上升",
            "B": "宁德时代 2025 年研发费用增幅高于营业收入增幅，因此研发费用率小幅上升",
            "C": "美的集团 2025 年研发费用金额增长约 9.58%，但研发费用率下降 0.09 个百分点",
            "D": "2025 年宁德时代研发费用率比美的集团高约 1.33 个百分点",
        }
        question = Question(
            qid="fin_b_009", domain="financial_reports", split="B",
            question="根据宁德时代与美的集团 2024 年、2025 年年度报告中的研发费用及研发费用占营业收入比例。",
            options=options, answer_format="multi", type="多选题",
            doc_ids=["annual_catl_2025_report", "annual_midea_2025_report"],
        )
        labels = {key: solver._choice_metric_bundle_rule(question, option)[0] for key, option in options.items()}
        self.assertEqual(labels, {"A": False, "B": True, "C": True, "D": True})


class RegulatoryCompositeClauseTests(unittest.TestCase):
    @staticmethod
    def make_solver(units: list[dict[str, object]]) -> RegulatorySolver:
        solver = RegulatorySolver.__new__(RegulatorySolver)
        solver.retriever = type("Retriever", (), {"units": units})()
        solver.supplemental_units = []
        return solver

    def test_low_risk_simplification_requires_institutional_assessment(self) -> None:
        solver = self.make_solver(
            [
                make_unit(
                    "cdd::29",
                    "cdd",
                    "金融机构经过风险评估且具有充足理由判断为低风险时，可以采取简化客户尽职调查措施。简化尽职调查不等于豁免。",
                )
            ]
        )
        question = Question(
            qid="reg_b_021",
            domain="regulatory",
            split="B",
            question="无法准确判断是否符合简化条件。",
            options={"B": "无法准确判断时不得简化或豁免"},
            answer_format="mcq",
            type="单选题",
            doc_ids=["cdd"],
        )
        hits = solver._targeted_literal_hits(question, question.options["B"])
        payload = solver._targeted_rule_payload(question.options["B"], hits)
        self.assertEqual([hit.unit_id for hit in hits], ["cdd::29"])
        self.assertTrue(payload["label"])

    def test_classification_chain_can_use_exact_corpus_wide_restructuring_rule(self) -> None:
        solver = self.make_solver(
            [
                make_unit(
                    "classification::score",
                    "classification",
                    "证券公司分类评价中，因违法违规被实施行政处罚、行政监管措施的，进行相应扣分。",
                ),
                make_unit(
                    "restructuring::duty",
                    "restructuring",
                    "为重大资产重组出具专业文件的证券服务机构未履行诚实守信、勤勉尽责义务，可以采取监管措施并依法追究法律责任。",
                ),
            ]
        )
        option = "可能承担中介责任，也可能因处罚或监管措施产生分类评价扣分"
        question = Question(
            qid="reg_b_024",
            domain="regulatory",
            split="B",
            question="重组中介未勤勉尽责。",
            options={"B": option},
            answer_format="mcq",
            type="单选题",
            doc_ids=["classification"],
        )
        hits = solver._targeted_literal_hits(question, option)
        payload = solver._targeted_rule_payload(option, hits)
        self.assertEqual({hit.unit_id for hit in hits}, {"classification::score", "restructuring::duty"})
        self.assertTrue(payload["label"])

    def test_non_trading_disclosure_combines_deadline_and_timely_definition(self) -> None:
        solver = self.make_solver(
            [
                make_unit(
                    "disclosure::8",
                    "disclosure",
                    "确有需要的，可以在非交易时段对外发布重大信息，但应当在下一交易时段开始前披露相关公告。",
                ),
                make_unit(
                    "disclosure::63",
                    "disclosure",
                    "及时，是指自起算日起或者触及披露时点的两个交易日内。",
                ),
            ]
        )
        solver.supplemental_units = [
            make_unit(
                "disclosure::8",
                "disclosure",
                "确有需要的，可以在非交易时段对外发布重大信息，但应当在下一交易时段开始前披露相关公告。",
            )
        ]
        option = "应在下一交易时段开始前披露，并遵守两个交易日内及时定义"
        question = Question(
            qid="reg_b_026",
            domain="regulatory",
            split="B",
            question="非交易时段发布重大信息。",
            options={"B": option},
            answer_format="mcq",
            type="单选题",
            doc_ids=["disclosure"],
        )
        hits = solver._targeted_literal_hits(question, option)
        payload = solver._targeted_rule_payload(option, hits)
        self.assertEqual({hit.unit_id for hit in hits}, {"disclosure::8", "disclosure::63"})
        self.assertTrue(payload["label"])

    def test_bankcard_statistics_reporting_clause_is_supported(self) -> None:
        solver = self.make_solver(
            [
                make_unit(
                    "bankcard::48",
                    "bankcard",
                    "银行卡清算机构应当按规定向中国人民银行报送业务统计数据、业务发展情况、业务管理情况等必要信息。",
                )
            ]
        )
        option = "业务统计应按办法报人民银行"
        question = Question(
            qid="reg_b_025",
            domain="regulatory",
            split="B",
            question="银行卡清算机构的业务统计。",
            options={"A": option},
            answer_format="multi",
            type="多选题",
            doc_ids=["bankcard"],
        )
        hits = solver._targeted_literal_hits(question, option)
        payload = solver._targeted_rule_payload(option, hits)
        self.assertEqual([hit.unit_id for hit in hits], ["bankcard::48"])
        self.assertTrue(payload["label"])

    def test_payment_fee_clause_supports_user_confirmation_and_continuous_notice(self) -> None:
        solver = self.make_solver(
            [
                make_unit(
                    "payment::62",
                    "payment",
                    "非银行支付机构调整支付业务的收费项目或者收费标准的，原则上应当至少于调整施行前30个自然日，在经营场所、官方网站、公众号等醒目位置，业务办理途径的关键节点持续公示，在办理相关业务前确认用户知悉、接受调整后的收费项目或者收费标准。",
                )
            ]
        )
        options = {
            "A": "官网公示30个自然日就足够",
            "B": "应在办理前确认用户知悉、接受",
            "C": "只需通知监管",
            "D": "调整施行前应持续公示",
        }
        question = Question(
            qid="reg_b_027",
            domain="regulatory",
            split="B",
            question="支付收费调整。",
            options=options,
            answer_format="multi",
            type="多选题",
            doc_ids=["payment"],
        )
        labels = {}
        for key, option in options.items():
            hits = solver._targeted_literal_hits(question, option)
            self.assertEqual([hit.unit_id for hit in hits], ["payment::62"])
            labels[key] = solver._targeted_rule_payload(option, hits)["label"]
        self.assertEqual(labels, {"A": False, "B": True, "C": False, "D": True})


class PluginParseUnitTests(unittest.TestCase):
    def _parse_with_sections(self, plugin, sections: list[dict[str, object]]) -> list[dict[str, object]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "manifest.json"
            output_path = root / "parsed.json"
            write_json(
                manifest_path,
                {
                    "domains": {
                        plugin.name: {
                            "referenced_doc_ids": ["doc-1"],
                            "documents": {
                                "doc-1": {"source_path": str(root / "source.txt"), "source_type": "txt"}
                            },
                        }
                    }
                },
            )
            module_name = f"afa_agent.domains.{plugin.name}.plugin"
            with (
                patch(f"{module_name}.get_stage_settings", return_value={}),
                patch(f"{module_name}.load_text_by_source", return_value=("source text", {})),
                patch(f"{module_name}.detect_title", return_value="Test title"),
                patch(f"{module_name}.split_text_into_sections", return_value=sections),
            ):
                plugin.parse(manifest_path, output_path)
            return read_json(output_path)["units"]

    def test_research_marks_conclusion_without_duplicate_section(self) -> None:
        units = self._parse_with_sections(
            ResearchPlugin(),
            [
                {
                    "section_id": "sec_1",
                    "title_path": ["Test title"],
                    "text": "预计市场规模同比增长",
                    "unit_type": "paragraph",
                    "page_refs": [],
                }
            ],
        )
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["unit_type"], "conclusion_block")
        self.assertEqual(len({unit["unit_id"] for unit in units}), len(units))

    def test_insurance_marks_formula_without_duplicate_section(self) -> None:
        units = self._parse_with_sections(
            InsurancePlugin(),
            [
                {
                    "section_id": "sec_1",
                    "title_path": ["Test title"],
                    "text": "身故保险金取账户价值与已交保费较大者",
                    "unit_type": "clause_block",
                    "page_refs": [],
                }
            ],
        )
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["unit_type"], "formula_block")
        self.assertEqual(len({unit["unit_id"] for unit in units}), len(units))


if __name__ == "__main__":
    unittest.main()
