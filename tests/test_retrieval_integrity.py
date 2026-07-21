from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from afa_agent.domains.generic_retriever import GenericBM25Retriever
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
