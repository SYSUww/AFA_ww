from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from afa_agent.domains.generic_retriever import GenericBM25Retriever
from afa_agent.domains.financial_contracts.solver import FinancialContractsSolver
from afa_agent.domains.financial_reports.solver import FinancialReportsSolver
from afa_agent.domains.insurance.plugin import InsurancePlugin
from afa_agent.domains.insurance.solver import InsuranceSolver
from afa_agent.domains.regulatory.retriever import RegulatoryRetriever
from afa_agent.domains.regulatory.solver import RegulatorySolver
from afa_agent.domains.research.plugin import ResearchPlugin
from afa_agent.domains.research.solver import ResearchSolver
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


class InsuranceProductIdentityClauseBundleTests(unittest.TestCase):
    @staticmethod
    def make_solver(units: list[dict[str, object]]) -> InsuranceSolver:
        solver = InsuranceSolver.__new__(InsuranceSolver)
        solver.retriever = GenericBM25Retriever(units)
        solver.answering_settings = {
            "prompt_template_id": "test",
            "max_hits": 5,
            "max_hit_chars": 600,
        }
        return solver

    def test_mental_damage_bundle_keeps_four_insurer_product_pairs(self) -> None:
        units = [
            make_unit("9::identity", "9", "中国平安财产保险股份有限公司 特种车商业保险示范条款"),
            make_unit(
                "9::clause", "9",
                "附加精神损害抚慰金责任险：投保特种车主险的特种车可投保本附加险，保险人依据法院判决负责赔偿精神损害抚慰金。",
            ),
            make_unit("10::identity", "10", "众安在线财产保险股份有限公司 特种车商业保险示范条款"),
            make_unit(
                "10::clause", "10",
                "附加精神损害抚慰金责任险：投保特种车主险的特种车可投保本附加险，保险人依据法院判决负责赔偿精神损害抚慰金。",
            ),
            make_unit("13::identity", "13", "众安在线财产保险股份有限公司 食品安全责任保险"),
            make_unit(
                "13::clause", "13",
                "依照人民法院判决应承担的精神损害赔偿责任属于可选责任。",
            ),
            make_unit("14::identity", "14", "中国平安财产保险股份有限公司 平安产险食品安全责任保险"),
            make_unit(
                "14::clause", "14",
                "依照人民法院判决应由被保险人承担的精神损害赔偿责任，保险人负责赔偿。",
            ),
        ]
        solver = self.make_solver(units)
        question = Question(
            qid="unseen_mental_damage",
            domain="insurance",
            split="B",
            question="关于精神损害赔偿或精神损害抚慰金，下列说法正确的是？",
            options={
                "A": "平安特种车商业保险可通过附加险承担",
                "B": "众安特种车商业保险可通过附加险承担",
                "C": "众安食品安全责任保险纳入法院判决责任",
                "D": "平安食品安全责任保险纳入法院判决责任",
            },
            answer_format="multi",
            type="多选题",
            doc_ids=["9", "10", "13"],
        )

        answer = solver._solve_product_identity_clause_bundle(question)

        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertEqual(answer.pred_answer, "ABCD")
        self.assertEqual(answer.token_usage.total_tokens, 0)
        self.assertEqual({item["doc_id"] for item in answer.evidence_items}, {"9", "10", "13", "14"})
        self.assertFalse(answer.debug_meta["answer_finalization"]["format_forced"])

    def test_administrative_exclusion_bundle_requires_literal_product_scoped_clause(self) -> None:
        units = [
            make_unit("2::identity", "2", "中国人寿保险股份有限公司 国寿增益宝终身寿险"),
            make_unit(
                "2::exclusion-a", "2",
                "第七条责任免除：被保险人故意犯罪或者抗拒依法采取的刑事强制措施。",
            ),
            make_unit(
                "2::exclusion-b", "2",
                "被保险人自合同成立之日起2年内自杀、酒后驾驶，或因战争、核爆炸导致身故。",
            ),
            make_unit("11::identity", "11", "中国平安财产保险股份有限公司 家庭财产保险 家庭版"),
            make_unit("11::clause", "11", "下列原因造成的损失不负责赔偿：行政行为或司法行为。"),
            make_unit("8::identity", "8", "众安在线财产保险股份有限公司 营运交通工具团体意外伤害保险"),
            make_unit(
                "8::clause", "8",
                "第七条责任免除：保险人不承担保险金给付责任，包括被保险人被依法拘留、服刑期间。",
            ),
            make_unit("14::identity", "14", "中国平安财产保险股份有限公司 平安产险食品安全责任保险"),
            make_unit("14::clause", "14", "下列原因造成的损失不负责赔偿：行政行为或司法行为。"),
        ]
        solver = self.make_solver(units)
        question = Question(
            qid="unseen_administrative_exclusion",
            domain="insurance",
            split="B",
            question="关于行政行为或司法行为导致损失的免责，下列产品明确列明的是？",
            options={"A": "国寿增益宝", "B": "平安家庭财产保险", "C": "众安营运交通工具团体意外", "D": "平安食品安全"},
            answer_format="multi",
            type="多选题",
            doc_ids=["2", "8", "11", "14"],
        )

        answer = solver._solve_product_identity_clause_bundle(question)

        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertEqual(answer.pred_answer, "BD")
        self.assertEqual(answer.option_labels, {"A": False, "B": True, "C": False, "D": True})
        self.assertEqual(answer.debug_meta["final_consistency_check"], {"issues": []})

    def test_limitation_period_bundle_uses_two_year_support_and_counterevidence(self) -> None:
        units = [
            make_unit("3::identity", "3", "众安在线财产保险股份有限公司 个人急性白血病复发医疗保险"),
            make_unit("3::clause", "3", "保险金申请人请求给付保险金的诉讼时效期间为二年。"),
            make_unit("5::identity", "5", "平安健康保险股份有限公司 平安e生保住院医疗保险"),
            make_unit("5::clause", "5", "诉讼时效适用现行有效法律规定。"),
            make_unit("6::identity", "6", "太平洋健康保险股份有限公司 太保团体百万医疗保险"),
            make_unit("6::clause", "6", "受益人请求给付保险金的诉讼时效期间为2年。"),
            make_unit("16::identity", "16", "平安养老保险股份有限公司 平安富鸿金生养老年金保险"),
            make_unit("16::clause", "16", "受益人请求给付保险金的诉讼时效期间为5年。"),
        ]
        solver = self.make_solver(units)
        question = Question(
            qid="unseen_limitation_period",
            domain="insurance",
            split="B",
            question="关于诉讼时效期间，下列产品明确约定保险金请求权诉讼时效期间为2年的是？",
            options={"A": "众安白血病", "B": "平安e生保", "C": "太保团体百万医疗", "D": "平安富鸿金生"},
            answer_format="multi",
            type="多选题",
            doc_ids=["3", "5", "6", "16"],
        )

        answer = solver._solve_product_identity_clause_bundle(question)

        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertEqual(answer.pred_answer, "AC")
        self.assertEqual(answer.token_usage.total_tokens, 0)
        self.assertIn("5年", answer.reasoning_summary)


class FinancialContractSubjectClauseTests(unittest.TestCase):
    @staticmethod
    def make_solver(units: list[dict[str, object]]) -> FinancialContractsSolver:
        solver = FinancialContractsSolver.__new__(FinancialContractsSolver)
        solver.retriever = GenericBM25Retriever(units)
        return solver

    def assert_bundle_labels(
        self,
        solver: FinancialContractsSolver,
        question: Question,
        expected: dict[str, bool],
        target_doc_id: str | set[str],
    ) -> None:
        target_doc_ids = {target_doc_id} if isinstance(target_doc_id, str) else target_doc_id
        labels: dict[str, bool] = {}
        for option_key, option_text in question.options.items():
            hits = solver._targeted_literal_hits(question, option_key, option_text)
            self.assertTrue(hits, option_key)
            self.assertTrue(all(hit.doc_id in target_doc_ids for hit in hits), option_key)
            result = solver._rule_override(question, option_key, option_text, hits)
            self.assertIsNotNone(result, option_key)
            labels[option_key] = bool(result["label"])
        self.assertEqual(labels, expected)

    def test_performance_reward_bundle_covers_limits_scope_payment_and_formula(self) -> None:
        units = [
            make_unit(
                "text08::reward-terms",
                "text08",
                "苏州华亚智能科技股份有限公司。本次交易中，业绩奖励总额不超过标的公司超额业绩部分的100%，且不超过交易作价的20%。"
                "上述超额业绩奖励的50%由标的公司以现金形式向奖励对象直接发放，50%通过设立专项资管计划，用于二级市场购买持有上市公司股票。"
                "超额业绩奖励金额=（业绩承诺期内累积实现净利润数-业绩承诺期内累积承诺净利润数）*50%。",
            ),
            make_unit(
                "text08::reward-subjects",
                "text08",
                "本次超额业绩奖励对象为届时仍在标的公司任职的管理团队及核心人员。",
            ),
            make_unit("text10::reward", "text10", "其他公司的业绩奖励对象为所有员工。"),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "超额业绩奖励总额不超过超额业绩部分的100%，且不超过交易作价的20%",
            "B": "业绩奖励对象为标的公司所有员工",
            "C": "超额业绩奖励的50%以现金形式直接发放，50%通过设立专项资管计划用于购买上市公司股票",
            "D": "业绩奖励金额=(业绩承诺期内累积实现净利润数-业绩承诺期内累积承诺净利润数)*50%",
        }
        question = Question(
            qid="unseen_reward_question", domain="financial_contracts", split="B",
            question="根据《苏州华亚智能科技股份有限公司交易报告书》，关于本次交易设置的业绩奖励机制。",
            options=options, answer_format="multi", type="多选题", doc_ids=["text08", "text10"],
        )
        self.assert_bundle_labels(solver, question, {"A": True, "B": False, "C": True, "D": True}, "text08")

    def test_downward_revision_bundle_binds_issuer_before_clause_judgment(self) -> None:
        units = [
            make_unit(
                "text09::revision",
                "text09",
                "金达威。在本次可转债存续期间，当公司股票在任意连续三十个交易日中至少有十五个交易日的收盘价低于当期转股价格的85%时触发。"
                "股东大会进行表决时，持有本次可转债的股东应当回避。修正后的转股价格应不低于该次股东大会召开日前二十个交易日公司股票交易均价和前一个交易日公司股票交易均价。"
                "同时，修正后的转股价格不得低于最近一期经审计的每股净资产值和股票面值。",
            ),
            make_unit("text11::revision", "text11", "其他发行人以80%为触发比例。"),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "触发条件为连续三十个交易日中至少十五个交易日收盘价低于当期转股价格的80%",
            "B": "修正后的转股价格应不低于股东大会召开日前二十个交易日公司股票交易均价和前一个交易日交易均价",
            "C": "修正后的转股价格不得低于最近一期经审计的每股净资产和股票面值",
            "D": "股东大会表决时，持有本次可转债的股东应当回避",
        }
        question = Question(
            qid="unseen_revision_question", domain="financial_contracts", split="B",
            question="关于金达威可转债转股价格向下修正条款。",
            options=options, answer_format="multi", type="多选题", doc_ids=["text11", "text09"],
        )
        self.assert_bundle_labels(solver, question, {"A": False, "B": True, "C": True, "D": True}, "text09")

    def test_reorganization_bundle_uses_negative_clauses_and_price_comparison(self) -> None:
        units = [
            make_unit(
                "text12::reorg-status",
                "text12",
                "陕国投。本次交易标的为海航旅游集团持有的长安银行股份，占长安银行股份的5.92%，以流拍价76,799.69万元抵偿债务。"
                "公司及公司控股股东与海航旅游集团不存在关联关系，本次交易不构成关联交易。本次交易不构成重组上市。",
            ),
            make_unit(
                "text12::reorg-valuation",
                "text12",
                "实际控制人仍为陕西省国资委，不会导致上市公司控制权发生变更。标的资产的定价依据为标的资产的流拍价格，"
                "评估机构采用市场法对长安银行股权进行评估，该5.92%股权价值为76,799.69万元。",
            ),
            make_unit("text07::reorg", "text07", "其他重大资产重组构成关联交易。"),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "本次交易构成关联交易，因为交易对方海航旅游集团与陕国投存在关联关系",
            "B": "本次交易不构成重组上市，因为实际控制人未变更",
            "C": "交易标的为长安银行5.92%股权，评估采用市场法",
            "D": "交易价格以流拍价确定，该价格低于评估值",
        }
        question = Question(
            qid="unseen_reorg_question", domain="financial_contracts", split="B",
            question="关于陕国投重大资产重组。",
            options=options, answer_format="multi", type="多选题", doc_ids=["text12", "text07"],
        )
        self.assert_bundle_labels(solver, question, {"A": False, "B": True, "C": True, "D": False}, "text12")

    def test_investor_protection_bundle_covers_covenant_default_and_dispute_clauses(self) -> None:
        units = [
            make_unit(
                "text03::protection",
                "text03",
                "厦门金圆投资集团有限公司。出现交叉保护承诺情形的，发行人将及时采取措施以在10个交易日内恢复承诺相关要求。"
                "发行人违反交叉保护条款且未在上述第（2）条约定期限内恢复承诺的，持有人有权要求发行人按照负面事项救济措施的约定采取负面事项救济措施。",
            ),
            make_unit(
                "text03::default-dispute",
                "text03",
                "发行人无法按时还本付息时，债券持有人同意给予发行人自原约定各给付日起90个自然日的宽限期。"
                "其他违约事项还包括提前偿付未足额、违反交叉保护承诺及其他承诺。"
                "协商不成的，双方约定向位于发行人住所所在地有管辖权的法院提请诉讼。",
            ),
            make_unit("text14::other", "text14", "其他发行人的投资者保护条款。"),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "发生交叉保护情形时，发行人应在10个交易日内采取措施恢复承诺相关要求",
            "B": "发行人发生违约时，债券持有人同意给予发行人自原约定给付日起90个自然日的宽限期",
            "C": "争议解决方式约定为向位于发行人住所所在地有管辖权的法院提请诉讼",
            "D": "发行人违反交叉保护条款且未在约定期限内恢复承诺的，持有人有权要求发行人按照负面事项救济措施采取行动",
        }
        question = Question(
            qid="unseen_protection_question", domain="financial_contracts", split="B",
            question="关于《厦门金圆投资集团有限公司募集说明书》中投资者保护条款与违约事项。",
            options=options, answer_format="multi", type="多选题", doc_ids=["text14", "text03"],
        )
        self.assert_bundle_labels(
            solver,
            question,
            {"A": True, "B": True, "C": True, "D": True},
            "text03",
        )

        universal_option = "发行人发生任何违约情形时，持有人均给予自原约定给付日起90个自然日的宽限期"
        universal_hits = solver._targeted_literal_hits(question, "B", universal_option)
        universal_result = solver._rule_override(question, "B", universal_option, universal_hits)
        self.assertIsNotNone(universal_result)
        self.assertFalse(universal_result["label"])
        self.assertEqual(
            universal_result["rule"],
            "contract_default_grace_explicit_universal_scope",
        )

        scoped_option = "发行人无法按时还本付息时，债券持有人同意给予自原约定给付日起90个自然日的宽限期"
        scoped_hits = solver._targeted_literal_hits(question, "B", scoped_option)
        scoped_result = solver._rule_override(question, "B", scoped_option, scoped_hits)
        self.assertIsNotNone(scoped_result)
        self.assertTrue(scoped_result["label"])
        self.assertEqual(scoped_result["rule"], "contract_default_payment_grace_period")

    def test_subscription_commitments_bind_each_option_to_its_issuer(self) -> None:
        units = [
            make_unit(
                "text04::independent", "text04",
                "安克创新。发行人独立董事承诺本人及本人配偶、父母、子女不参与本次可转债的发行认购，亦不会委托其他主体参与本次可转债的发行认购。",
            ),
            make_unit(
                "text04::controller", "text04",
                "若发行日与最后一次减持公司股票的日期间隔不满六个月，本人及配偶、父母、子女将不参与认购公司本次发行的可转债。",
            ),
            make_unit(
                "text11::subscription", "text11",
                "普联软件。公司独立董事关于不参与本次可转债发行认购的承诺：本人及本人配偶、父母、子女将不参与本次可转债发行认购，亦不会委托其他主体参与。",
            ),
            make_unit(
                "text05::subscription", "text05",
                "本川智能。发行人独立董事出具承诺：本人及本人关系密切的家庭成员承诺不认购本次发行可转债，亦不会委托其他主体参与本次发行可转债发行认购。",
            ),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "安克创新的独立董事承诺不参与本次可转债的发行认购，亦不会委托其他主体参与",
            "B": "普联软件的独立董事承诺不参与本次可转债发行认购，但未明确其配偶、父母、子女是否参与",
            "C": "本川智能的独立董事承诺本人及本人关系密切的家庭成员不认购本次发行可转债",
            "D": "安克创新的控股股东承诺若发行日与最后一次减持公司股票的日期间隔不满六个月，将不参与认购公司本次发行的可转债",
        }
        question = Question(
            qid="unseen_subscription_question", domain="financial_contracts", split="B",
            question="关于上市公司相关主体针对本次可转债发行认购的承诺，以下说法错误的是？",
            options=options, answer_format="mcq", type="单选题", doc_ids=["text11", "text04"],
        )
        expected = {"A": False, "B": True, "C": False, "D": False}
        labels = {}
        payloads = []
        expected_docs = {"A": "text04", "B": "text11", "C": "text05", "D": "text04"}
        for option_key, option_text in options.items():
            hits = solver._targeted_literal_hits(question, option_key, option_text)
            self.assertTrue(hits, option_key)
            self.assertTrue(all(hit.doc_id == expected_docs[option_key] for hit in hits), option_key)
            result = solver._rule_override(question, option_key, option_text, hits)
            self.assertIsNotNone(result, option_key)
            labels[option_key] = bool(result["label"])
            focused_hits = solver._prioritize_rule_hits(result["rule"], hits, question.doc_ids)
            self.assertEqual(sum(bool(hit.metadata.get("targeted_literal")) for hit in focused_hits), 1)
            payloads.append(
                {
                    "option": option_key,
                    "label": bool(result["label"]),
                    "rule_override": result,
                    "evidence_items": [hit.to_dict() for hit in focused_hits],
                }
            )
        self.assertEqual(labels, expected)
        focused_evidence = solver._complete_rule_evidence_items(payloads)
        self.assertEqual(len(focused_evidence), 4)
        self.assertEqual(
            {item["metadata"]["option_key"]: item["metadata"]["document_subject"] for item in focused_evidence},
            {"A": "安克创新", "B": "普联软件", "C": "本川智能", "D": "安克创新"},
        )

    def test_convertible_rights_bundle_covers_price_vote_redemption_and_put(self) -> None:
        units = [
            make_unit(
                "text06::identity", "text06",
                "公司、本公司、发行人、鼎捷数智，均指鼎捷数智股份有限公司。",
            ),
            make_unit(
                "text06::conversion", "text06",
                "本次发行可转换公司债券的初始转股价格不低于募集说明书公告日前二十个交易日公司股票交易均价和前一个交易日公司股票交易均价。"
                "向下修正方案须经出席会议的股东所持表决权的三分之二以上通过。",
            ),
            make_unit(
                "text06::redemption-put", "text06",
                "到期赎回条款：具体赎回价格将提请股东大会授权董事会根据市场情况与保荐机构（主承销商）协商确定。"
                "本次发行的可转债最后两个计息年度，可转债持有人在每年回售条件首次满足后可行使回售权一次，不能多次行使部分回售权。",
            ),
            make_unit(
                "text11::other", "text11",
                "普联软件。初始转股价格不低于募集说明书公告日前二十个交易日公司股票交易均价和前一个交易日公司股票交易均价。"
                "向下修正方案须经出席会议的股东所持表决权的三分之二以上通过。"
                "到期赎回条款：具体赎回价格将提请股东大会授权董事会根据市场情况与保荐机构协商确定。"
                "最后两个计息年度内，每年回售条件首次满足后可行使回售权一次，不能多次行使部分回售权。",
            ),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "初始转股价格不低于募集说明书公告日前二十个交易日公司股票交易均价和前一个交易日交易均价",
            "B": "转股价格向下修正方案须经出席会议的股东所持表决权的三分之二以上通过",
            "C": "到期赎回价格由董事会根据市场情况直接确定，无需股东大会授权",
            "D": "有条件回售条款仅在最后两个计息年度内触发，且每年只能行使一次",
        }
        question = Question(
            qid="unseen_convertible_rights", domain="financial_contracts", split="B",
            question="关于鼎捷数智可转换公司债券的发行条款。",
            options=options, answer_format="multi", type="多选题", doc_ids=["text11", "text06"],
        )
        self.assertEqual(solver._question_subject_terms(question.question), ["鼎捷数智"])
        self.assertEqual(solver._subject_bound_doc_ids(question), ["text06"])
        self.assert_bundle_labels(solver, question, {"A": True, "B": True, "C": False, "D": True}, "text06")

    def test_concentration_bundle_covers_table_rating_eligibility_and_transition(self) -> None:
        units = [
            make_unit(
                "text02::concentration-table", "text02",
                "深圳市融资租赁（集团）有限公司。对单一集团的全部融资租赁业务余额占净资产的比例 | ≤50% | "
                "90.79% | 100.42% | 107.81% | 119.28%。监管指标整改原则上有不超过3年的过渡期。",
            ),
            make_unit(
                "text02::concentration-eligibility", "text02",
                "发行人两年行业监管评级均为A级。相关行业租赁资产占租赁资产总额84.17%，超过80%，"
                "适用《广东省融资租赁公司监督管理实施细则》关于适当放宽集中度关联度要求的条款。",
            ),
            make_unit("text14::other", "text14", "其他融资租赁公司的集中度指标。"),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "发行人近三年及一期对单一集团的全部融资租赁业务余额均超过净资产50%的要求",
            "B": "发行人已连续两年获得深圳市地方金融管理局A级行业监管评级",
            "C": "发行人满足《广东省融资租赁公司监督管理实施细则》中关于放宽集中度关联度要求的适用条件",
            "D": "根据《融资租赁公司监督管理暂行办法》，过渡期原则上不超过5年",
        }
        question = Question(
            qid="unseen_concentration_bundle", domain="financial_contracts", split="B",
            question="根据深圳市融资租赁（集团）有限公司募集说明书，关于发行人集中度指标不符合监管要求的情况。",
            options=options, answer_format="multi", type="多选题", doc_ids=["text14", "text02"],
        )
        self.assert_bundle_labels(solver, question, {"A": True, "B": True, "C": True, "D": False}, "text02")

    def test_depreciation_bundle_recovers_full_tables_for_each_issuer(self) -> None:
        units = [
            make_unit(
                "text11::depreciation", "text11",
                "普联软件。新增折旧摊销合计 | 募投项目预计营业收入合计 | 募投项目预计净利润合计 | "
                "折旧摊销占营业收入比重 | 折旧摊销占净利润比重。T+2 | 854.84 | 5,398.00 | 991.91。"
                "T+10 | 137.50 | 23,302.00 | 6,290.32。",
            ),
            make_unit(
                "text05::depreciation", "text05",
                "本川智能。本次募投项目在完全达产（T+5年）前，新增折旧摊销占营业收入最高比例为3.49%，"
                "占净利润最高比例为77.09%。",
            ),
            make_unit(
                "text04::depreciation", "text04",
                "安克创新。募投项目年新增折旧摊销费用预计最高金额为11,376.57万元，"
                "新增营业收入预计可以覆盖项目折旧摊销费用。",
            ),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "普联软件详细测算了本次募投项目T+2年至T+10年每年新增折旧摊销、营业收入和净利润，并计算了折旧摊销占营业收入、净利润的比重",
            "B": "本川智能测算了本次募投项目在完全达产（T+5年）前，新增折旧摊销占营业收入最高比例为3.49%，占净利润最高比例为77.09%",
            "C": "安克创新在募投项目新增资产折旧摊销的风险中仅定性描述了风险，未量化披露测算数据",
            "D": "普联软件和本川智能均在募集说明书中对募投项目新增折旧摊销进行了定量测算并披露",
        }
        question = Question(
            qid="unseen_depreciation_bundle", domain="financial_contracts", split="B",
            question="关于募集资金投资项目新增折旧摊销对未来经营业绩的影响，以下说法一致的是？",
            options=options, answer_format="multi", type="多选题", doc_ids=["text04", "text05", "text11"],
        )
        expected_docs = {
            "A": {"text11"}, "B": {"text05"}, "C": {"text04"}, "D": {"text05", "text11"},
        }
        labels: dict[str, bool] = {}
        for option_key, option_text in options.items():
            hits = solver._targeted_literal_hits(question, option_key, option_text)
            self.assertTrue(hits, option_key)
            self.assertEqual({hit.doc_id for hit in hits}, expected_docs[option_key], option_key)
            result = solver._rule_override(question, option_key, option_text, hits)
            self.assertIsNotNone(result, option_key)
            labels[option_key] = bool(result["label"])
        self.assertEqual(labels, {"A": True, "B": True, "C": False, "D": True})

    def test_compensation_bundle_recovers_income_formula_and_payment_priority(self) -> None:
        units = [
            make_unit(
                "text10::compensation", "text10",
                "山东科源制药股份有限公司与补偿义务人签署《业绩预测补偿及减值补偿协议》。"
                "各业绩承诺方用于补偿的股份数最高不超过其因本次交易获得的上市公司股份。"
                "当各业绩承诺方因本次交易获得的上市公司股份不足以支付其业绩补偿金额时，补偿义务人应以现金进行补偿。"
                "麝香酮资产当期补偿金额=（截至当期期末的累积承诺收入－截至当期期末累积实际收入）÷"
                "承诺期间各年的承诺收入总和×麝香酮资产交易作价×补偿义务人本次交易前持有宏济堂股份比例39.61%－累积已补偿金额。",
            ),
            make_unit("text08::other", "text08", "其他交易的补偿条款。"),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "补偿义务人应优先以现金方式进行补偿",
            "B": "补偿义务人以股份补偿为主，股份不足以支付时以现金补足",
            "C": "收入承诺未达成时，补偿金额按公式计算，涉及累积承诺收入与实际收入的差额",
            "D": "补偿金额的计算中考虑了交易作价和补偿义务人本次交易前持有宏济堂的股份比例",
        }
        question = Question(
            qid="unseen_compensation_bundle", domain="financial_contracts", split="B",
            question="根据《业绩预测补偿及减值补偿协议》，关于科源制药重组中补偿义务人的补偿方式。",
            options=options, answer_format="multi", type="多选题", doc_ids=["text08", "text10"],
        )
        self.assert_bundle_labels(solver, question, {"A": False, "B": True, "C": True, "D": True}, "text10")

    def test_solvency_bundle_keeps_all_year_rows_together(self) -> None:
        units = [
            make_unit(
                "text14::solvency", "text14",
                "西部证券。项目 | 2025年12月31日 | 2024年12月31日 | 2023年12月31日。"
                "资产负债率（扣除代理款） | 64.00 | 62.47 | 66.27。流动比率 | 1.83 | 1.95 | 1.91。",
            ),
            make_unit("text02::other", "text02", "其他发行人的资产负债率。"),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "2025年末资产负债率（扣除代理款）为64.00%，流动比率为1.83",
            "B": "2024年末资产负债率（扣除代理款）为62.47%，流动比率为1.95",
            "C": "2023年末资产负债率（扣除代理款）为66.27%，流动比率为1.91",
            "D": "2023年末资产负债率（扣除代理款）为67.27%，流动比率为1.89",
        }
        question = Question(
            qid="unseen_solvency_bundle", domain="financial_contracts", split="B",
            question="根据西部证券债券募集说明书，关于发行人资产负债率（扣除代理款）和流动比率的描述。",
            options=options, answer_format="multi", type="多选题", doc_ids=["text02", "text14"],
        )
        self.assert_bundle_labels(solver, question, {"A": True, "B": True, "C": True, "D": False}, "text14")

    def test_lockup_bundle_recovers_all_counterparty_groups(self) -> None:
        units = [
            make_unit(
                "text10::lockup-primary", "text10",
                "科源制药重组锁定期安排。交易对方力诺投资、力诺集团承诺，自本次股份发行结束之日起36个月内不得转让。"
                "交易对方济南财投新动能、济南财金投资、济南鑫控承诺，自本次股份发行结束之日起36个月内不得转让。",
            ),
            make_unit(
                "text10::lockup-other", "text10",
                "除力诺投资、力诺集团、济南财投新动能、济南财金投资、济南鑫控外的交易对方承诺，"
                "自本次股份发行结束之日起12个月内不得转让；但持续拥有权益的时间不足12个月的，36个月内不得转让。",
            ),
            make_unit("text08::other", "text08", "其他重组的锁定期安排。"),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "力诺投资、力诺集团承诺股份锁定期为36个月",
            "B": "济南财投新动能、济南财金投资、济南鑫控承诺锁定期为36个月",
            "C": "除上述主体外的其他交易对方，若对用于认购股份的资产持续拥有权益时间不足12个月，锁定期为12个月",
            "D": "所有交易对方均承诺锁定期为36个月",
        }
        question = Question(
            qid="unseen_lockup_bundle", domain="financial_contracts", split="B",
            question="关于科源制药重组交易对方的锁定期安排，下列说法错误的是？",
            options=options, answer_format="multi", type="多选题", doc_ids=["text08", "text10"],
        )
        self.assert_bundle_labels(solver, question, {"A": False, "B": False, "C": True, "D": True}, "text10")

    def test_capacity_risk_matrix_distinguishes_production_capacity_from_software_and_warehousing(self) -> None:
        units = [
            make_unit("text04::identity", "text04", "股票简称：安克创新 股票代码：300866"),
            make_unit(
                "text04::risk-list", "text04",
                "募投项目拟研发产品产业化落地风险。募集资金投资项目效益不及预期的风险。"
                "项目收入基于预计销量和预计单价测算。",
            ),
            make_unit(
                "text04::depreciation", "text04",
                "募投项目新增资产折旧摊销的风险，预计最高金额为11,376.57万元。",
            ),
            make_unit(
                "text04::warehouse", "text04",
                "仓储智能化升级项目将实现仓储、配送、管理等环节的自动化和智能化，提高仓储运营效率和服务质量。",
            ),
            make_unit("text05::identity", "text05", "股票简称 | 本川智能 股票代码 | 300964.SZ"),
            make_unit(
                "text05::customer-pipeline", "text05",
                "2024年下半年以来，公司开拓的新客户合作后预计年销售额合计约40,500万元，"
                "上述40,500万元客户采购意向涉及领域与本次募投项目产品主要面向领域的相关性较高。",
            ),
            make_unit(
                "text05::capacity-risk", "text05",
                "募投项目新增产能消化风险。项目建成投产后，将新增合计55万平方米的年产能，"
                "报告期内公司产能利用率处于较高水平。",
            ),
            make_unit(
                "text11::identity", "text11",
                "公司名称 | 普联软件股份有限公司 股票代码 | 300996",
            ),
            make_unit(
                "text11::software-projects", "text11",
                "公司所处行业为软件和信息技术服务业（I65），募集资金投向国产ERP功能扩展建设项目、"
                "数智化金融风险管控系列产品建设项目及云湖平台研发升级项目，均主要投向现有研发方向，不属于落后产能。",
            ),
            make_unit(
                "text11::research-risk", "text11",
                "募集资金投资项目研发风险：ERP产品适配性研发升级、XBRL产品改造和云湖平台技术底座升级。",
            ),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "安克创新在风险因素中提到了募投项目新增产能消化风险，但未量化具体数据",
            "B": "本川智能详细披露了2024年下半年以来开拓的新客户合作后预计年销售额合计约40,500万元，并指出该采购意向与本次募投项目产品主要面向领域相关性较高",
            "C": "普联软件未在其募投项目风险中提及新增产能消化问题，因其募投项目为软件研发类，不涉及传统产能",
            "D": "安克创新的募投项目包括仓储智能化升级，因此涉及新增产能消化风险",
        }
        question = Question(
            qid="unseen_capacity_risk_matrix", domain="financial_contracts", split="B",
            question="关于募投项目新增产能消化风险，以下表述与各文件原文一致的是？",
            options=options, answer_format="multi", type="多选题", doc_ids=["text04", "text05", "text11"],
        )
        expected_docs = {"A": {"text04"}, "B": {"text05"}, "C": {"text11"}, "D": {"text04"}}
        labels: dict[str, bool] = {}
        for option_key, option_text in options.items():
            hits = solver._targeted_literal_hits(question, option_key, option_text)
            self.assertTrue(hits, option_key)
            self.assertEqual({hit.doc_id for hit in hits}, expected_docs[option_key], option_key)
            self.assertTrue(all(hit.metadata.get("document_subject") for hit in hits), option_key)
            result = solver._rule_override(question, option_key, option_text, hits)
            self.assertIsNotNone(result, option_key)
            labels[option_key] = bool(result["label"])
        self.assertEqual(labels, {"A": False, "B": True, "C": True, "D": False})

    def test_holder_meeting_matrix_preserves_issuer_wording_and_excludes_false_equivalence(self) -> None:
        units = [
            make_unit("text04::identity", "text04", "股票简称：安克创新 股票代码：300866"),
            make_unit(
                "text04::meeting", "text04",
                "债券持有人会议：公司发生减资（因员工持股计划、股权激励或公司为维护公司价值及股东权益"
                "所必需回购股份导致的减资除外）。除约定之外，不得要求公司提前偿付可转债的本金和利息。",
            ),
            make_unit("text05::identity", "text05", "股票简称 | 本川智能 股票代码 | 300964.SZ"),
            make_unit(
                "text05::meeting", "text05",
                "债券持有人会议的召开情形包括公司发生减资（因公司实施员工持股计划、股权激励、"
                "用于转换公司发行的本次可转债或为维护公司价值及股东权益而进行股份回购导致的减资除外）。"
                "本次可转债持有人不得因此要求公司提前清偿或者提供相应的担保。",
            ),
            make_unit(
                "text11::identity", "text11",
                "公司名称 | 普联软件股份有限公司 股票代码 | 300996",
            ),
            make_unit(
                "text11::meeting", "text11",
                "债券持有人会议的召开情形包括公司发生减资（因员工持股计划、股权激励或公司为维护公司价值及"
                "股东权益所必须回购股份导致的减资除外）。若公司发生因持股计划、股权激励或为维护公司价值及"
                "股东权益回购股份而导致减资，本次可转换债券持有人不得因此要求公司提前清偿或者提供相应的担保。",
            ),
        ]
        solver = self.make_solver(units)
        options = {
            "A": "安克创新列举的召开情形包括公司发生减资，因员工持股计划、股权激励或公司为维护公司价值及股东权益所必需回购股份导致的减资除外",
            "B": "普联软件列举的召开情形包括公司发生减资，因员工持股计划、股权激励或公司为维护公司价值及股东权益所必须回购股份导致的减资除外",
            "C": "本川智能列举的召开情形包括公司发生减资，因公司实施员工持股计划、股权激励、用于转换公司发行的本次可转债或为维护公司价值及股东权益而进行股份回购导致的减资除外",
            "D": "三份募集说明书中，关于因股份回购导致减资时债券持有人是否享有提前清偿或担保请求权的规定完全一致",
        }
        question = Question(
            qid="unseen_holder_meeting_matrix", domain="financial_contracts", split="B",
            question="关于债券持有人会议召开的情形，以下说法符合三份募集说明书规定的是？",
            options=options, answer_format="multi", type="多选题", doc_ids=["text04", "text05", "text11"],
        )
        expected_docs = {"A": {"text04"}, "B": {"text11"}, "C": {"text05"}, "D": {"text04", "text05", "text11"}}
        labels: dict[str, bool] = {}
        for option_key, option_text in options.items():
            hits = solver._targeted_literal_hits(question, option_key, option_text)
            self.assertTrue(hits, option_key)
            self.assertEqual({hit.doc_id for hit in hits}, expected_docs[option_key], option_key)
            self.assertTrue(all(hit.metadata.get("document_subject") for hit in hits), option_key)
            result = solver._rule_override(question, option_key, option_text, hits)
            self.assertIsNotNone(result, option_key)
            labels[option_key] = bool(result["label"])
        self.assertEqual(labels, {"A": True, "B": True, "C": True, "D": False})

    def test_complete_rule_evidence_keeps_support_and_counterevidence_without_locator_backfill(self) -> None:
        support = make_unit("target::a", "target", "A项直接证据")
        support["metadata"] = {"targeted_literal": True}
        counterevidence = make_unit("target::b", "target", "B项反证")
        counterevidence["metadata"] = {"targeted_literal": True}
        payloads = [
            {
                "option": "A",
                "label": True,
                "rule_override": {"rule": "contract_exact_a"},
                "evidence_items": [
                    support,
                    make_unit("noise::a", "noise", "无关locator文档"),
                ],
            },
            {
                "option": "B",
                "label": False,
                "rule_override": {"rule": "contract_exact_b"},
                "evidence_items": [counterevidence],
            },
        ]

        evidence = FinancialContractsSolver._complete_rule_evidence_items(payloads)

        self.assertEqual([item["unit_id"] for item in evidence], ["target::a", "target::b"])
        self.assertEqual(evidence[0]["metadata"]["option_key"], "A")
        self.assertTrue(evidence[0]["metadata"]["rule_label"])
        self.assertEqual(evidence[1]["metadata"]["option_key"], "B")
        self.assertFalse(evidence[1]["metadata"]["rule_label"])


class ResearchFinancialClauseBundleTests(unittest.TestCase):
    @staticmethod
    def make_solver(units: list[dict[str, object]]) -> ResearchSolver:
        solver = ResearchSolver.__new__(ResearchSolver)
        solver.retriever = GenericBM25Retriever(units)
        return solver

    def assert_labels(
        self,
        solver: ResearchSolver,
        question: Question,
        expected: dict[str, bool],
    ) -> None:
        labels: dict[str, bool] = {}
        payloads: list[dict[str, object]] = []
        for option_key, option_text in question.options.items():
            result = solver._rule_evaluate(question, option_text, [])
            self.assertIsNotNone(result[0], option_key)
            self.assertTrue(result[2], option_key)
            labels[option_key] = bool(result[0])
            payloads.append(
                {
                    "option": option_key,
                    "label": bool(result[0]),
                    "rule_backed": True,
                    "evidence_items": [hit.to_dict() for hit in result[2]],
                }
            )
        self.assertEqual(labels, expected)
        focused = solver._complete_rule_evidence_items(payloads)
        self.assertTrue(focused)
        self.assertTrue(all(item["metadata"].get("targeted_research") for item in focused))

    def test_deposit_migration_bundle_supports_insurance_and_low_volatility_wealth_management(self) -> None:
        units = [
            make_unit(
                "banca::migration", "banca",
                "居民存款逐步向理财、基金及保险等替代资产转移。保险产品凭借相对稳定的收益特征及长期锁定收益能力，"
                "成为承接存款搬家的重要方向，相对银行存款具备比较优势。",
            ),
            make_unit(
                "allocation::wealth", "allocation",
                "理财资金大幅增加公募基金和存款配置，基金配置以债基和货基为主，对权益类基金配置很少；"
                "其负债端对波动的低容忍度决定了稳健优先。",
            ),
            make_unit(
                "funds::redemption", "funds",
                "赎回费规则对债券基金和指数型基金显著宽松，较此前条款明显宽松。",
            ),
        ]
        question = Question(
            qid="unseen_deposit_migration", domain="research", split="B",
            question="存款搬家现象正在发生，以下关于不同金融机构影响的判断中哪些得到支持？",
            options={
                "A": "存款搬家的主要去向是高风险权益市场，因此推升股市估值中枢",
                "B": "保险产品凭借长期锁定收益和相对存款的利率优势成为重要载体",
                "C": "银行理财因负债端对净值波动敏感，主要配置固收和公募基金",
                "D": "公募基金赎回费新规抑制存款搬家并使基金规模增长停滞",
            },
            answer_format="multi", type="多选题", doc_ids=["banca", "allocation", "funds"],
        )
        self.assert_labels(solver=self.make_solver(units), question=question, expected={"A": False, "B": True, "C": True, "D": False})

    def test_bancassurance_bundle_links_participating_products_to_deep_bank_cooperation(self) -> None:
        units = [
            make_unit(
                "banca::value", "banca",
                "报行合一实行之后银保渠道价值提升，头部保险公司积极发展银保。",
            ),
            make_unit(
                "insurer::participating", "insurer",
                "分红险成为行业转型核心战略，依靠风险共担、收益共享机制；行业转型共识形成，"
                "分红险在新单结构中已占据主导地位。",
            ),
            make_unit(
                "allocation::balance", "allocation",
                "投资端为匹配负债要求，需要解决高波动资产与偿付能力之间的平衡难题，并配置高分红资产。",
            ),
            make_unit(
                "banca::binding", "banca",
                "分红险依赖长期稳定的银保战略合作关系以及产品、客户和资产配置的深度协同；"
                "合作关系将从协议代理向长期战略合作转型并深化银保一体化合作。",
            ),
        ]
        question = Question(
            qid="unseen_bancassurance_strategy", domain="research", split="B",
            question="在预定利率持续下调和报行合一深化的背景下，哪些最可能成为行业共识？",
            options={
                "A": "个险仍是唯一重要渠道，银保战略地位无需提升",
                "B": "分红险转型要求投资端提供稳定的分红收益来源",
                "C": "应单边扩大高波动高收益成长股",
                "D": "银保价值持续提升取决于深度绑定而非简单协议代理",
            },
            answer_format="multi", type="多选题", doc_ids=["banca", "insurer", "allocation"],
        )
        self.assert_labels(solver=self.make_solver(units), question=question, expected={"A": False, "B": True, "C": False, "D": True})

    def test_asset_liability_bundle_combines_duration_fvoci_and_risk_constraints(self) -> None:
        units = [
            make_unit(
                "allocation::duration", "allocation",
                "有效久期缺口综合考虑资产和负债的到期现金流。增配长久期低风险债和发展分红险后，"
                "资产负债久期缺口持续收窄。",
            ),
            make_unit(
                "allocation::fvoci", "allocation",
                "提高FVOCI账户占比可以平滑利润波动；高分红低波动资产计入其他综合收益，有助于报表稳定性。",
            ),
            make_unit(
                "insurer_report::government", "insurer_report",
                "加大长久期国债配置可拉长资产久期，锁定长期稳定票息；政府债配置有效缩窄资产负债久期缺口。",
            ),
            make_unit(
                "broker::capital", "broker",
                "两融和股票质押属于券商资本中介业务的扩表场景。",
            ),
            make_unit(
                "allocation::risk", "allocation",
                "资产配置为匹配负债要求，需要处理高波动资产、偿付能力与收益之间的平衡难题。",
            ),
        ]
        question = Question(
            qid="unseen_asset_liability", domain="research", split="B",
            question="关于金融机构资产负债管理能力的讨论，哪些符合当前主流观点？",
            options={
                "A": "优化资产负债久期匹配并增配FVOCI资产，实现更好的资产负债联动",
                "B": "增配长久期政府债券有助于拉长资产久期并匹配稳定负债",
                "C": "两融和股票质押是平滑券商收入的资产负债管理工具",
                "D": "所有机构都应追求收益最大化而非风险最小化",
            },
            answer_format="multi", type="多选题", doc_ids=["allocation", "broker"],
        )
        self.assert_labels(solver=self.make_solver(units), question=question, expected={"A": True, "B": True, "C": False, "D": False})

    def test_cross_industry_innovation_bundle_supports_reverse_export_and_precision_capability(self) -> None:
        units = [
            make_unit(
                "auto::export", "auto",
                "中国方案主导合资转型，大众平台首次实现中方主导定义，研发主导权从外资向中方转移，"
                "中国开始反向输出技术标准。",
            ),
            make_unit(
                "manufacturing::precision", "manufacturing",
                "HANS M410拥有微米级精密制造体系，支持3C零部件复杂结构一体化成型，并推动3D打印设备向高精度、"
                "多功能集成方向迭代，深度参与头部客户研发。",
            ),
            make_unit(
                "asic::self-developed", "asic",
                "自研 ASIC 成为 CSP 投资重心。OpenAI正在推进自研AI芯片，自研ASIC可为大模型提供更高度的定制化支持。",
            ),
        ]
        question = Question(
            qid="unseen_cross_industry_innovation", domain="research", split="B",
            question="合资品牌导入中国智驾，海外云厂商发展自研ASIC，消费电子制造采用3D打印技术，这些事件指向什么？",
            options={
                "A": "中国企业在部分高端制造和核心技术领域已具备反向输出能力，全球产业链分工正在变化",
                "B": "跨国企业正在全面采用中国供应商并放弃自研",
                "C": "中国企业精密制造和算法能力提升，使其参与甚至主导部分全球产业链创新环节",
                "D": "合资导入中国智驾只是权宜之计，之后会重新切换",
            },
            answer_format="multi", type="多选题", doc_ids=["auto", "manufacturing", "asic"],
        )
        self.assert_labels(solver=self.make_solver(units), question=question, expected={"A": True, "B": False, "C": True, "D": False})

    def test_cross_industry_autonomy_bundle_links_staged_paths_and_rejects_absolute_claims(self) -> None:
        units = [
            make_unit(
                "auto::autonomy", "auto",
                "自主品牌高阶智驾与自研芯片并进，集中呈现智能驾驶算法；新势力竞逐自研芯片与全域智驾，"
                "并搭载5nm智驾芯片。",
            ),
            make_unit(
                "auto::progression", "auto",
                "当前辅助驾驶系统加速落地，2026年进入L3级自动驾驶规模化商用阶段，部分车型预埋L4级智驾。",
            ),
            make_unit(
                "asic::automotive", "asic",
                "公司依托自主半导体IP提供芯片定制服务，应用覆盖汽车电子；软硬件芯片定制平台解决方案覆盖智慧汽车。",
            ),
            make_unit(
                "equipment::substitution", "equipment",
                "国产光模块测试仪器龙头有望受益国产替代，相关设备仍存在国产替代空间；老旧进口设备替换正在推进，"
                "自主研发工艺已产品化并批量生产。",
            ),
            make_unit(
                "bank::progression", "bank",
                "银行信创从办公到一般业务，再到核心系统，遵循由外到内、由易及难；具体从非关键外围业务起步，"
                "经办公系统和一般业务系统，分三个阶段攻坚核心系统。",
            ),
            make_unit(
                "asic::reuse", "asic",
                "半导体IP提供预先验证、可重复使用的功能模块，应对SoC设计复杂度；SiPaaS依靠可复用性缩短设计周期并降低设计风险。",
            ),
        ]
        question = Question(
            qid="unseen_cross_industry_autonomy", domain="research", split="B",
            question="汽车行业加速智能化、芯片ASIC定制、激光设备国产替代、银行IT推进信创，哪些推进路径分析正确？",
            options={
                "A": "汽车自主可控主要体现为智驾芯片和算法自研，与ASIC定制服务直接相关",
                "B": "激光设备国产化率已接近100%，因此国产替代空间有限",
                "C": "银行IT从外围系统到核心系统，与汽车从辅助驾驶到完全自动驾驶的渐进路线相似",
                "D": "芯片IP授权模式与银行IT完全相同，都是购买现成软件快速替代",
            },
            answer_format="multi", type="多选题", doc_ids=["auto", "asic", "equipment", "bank"],
        )
        self.assert_labels(solver=self.make_solver(units), question=question, expected={"A": True, "B": False, "C": True, "D": False})

    def test_staged_autonomy_direct_entailment_survives_runtime_chunk_boundaries(self) -> None:
        units = [
            make_unit(
                "auto::assist_l3", "auto",
                "自主品牌高阶智驾与自研芯片并进，集中呈现智能驾驶算法；G-ASD辅助驾驶系统已落地，"
                "行业进入L3级自动驾驶规模化商用阶段，并搭载5nm智驾芯片。",
            ),
            make_unit(
                "auto::l4", "auto",
                "新车型搭载高算力平台与全域线控转向，并预埋L4级智驾。",
            ),
            make_unit(
                "asic::adas", "asic",
                "公司将半导体IP、芯片定制服务和软件支持服务有机结合，提供高性能车规ADAS系统平台解决方案。",
            ),
            make_unit(
                "asic::smart_car", "asic",
                "面向AI应用的软硬件芯片定制平台解决方案覆盖智慧汽车等高效率端侧计算设备。",
            ),
            make_unit(
                "equipment::share", "equipment",
                "光模块贴片设备全球市占率数据显示外国企业与其他企业占比79%，仍存在较大国产替代空间。",
            ),
            make_unit(
                "testing::share", "equipment",
                "海外企业合计占据84%主导份额，国产替代空间广阔。",
            ),
            make_unit(
                "bank::stages", "bank",
                "银行信创从办公系统到一般业务系统，最后攻坚核心系统，分三个阶段推进。",
            ),
            make_unit(
                "ip::module", "asic",
                "半导体IP提供预先验证、可重复使用的功能模块，可降低昂贵研发成本。",
            ),
            make_unit(
                "ip::custom", "asic",
                "芯片设计将预先验证的IP核与定制设计的电路组合，从而构建复杂的芯片。",
            ),
            make_unit(
                "bank::full_stack", "bank",
                "银行信创实现从底层硬件到上层应用软件的全面自主可控，推进路径从办公到一般业务。",
            ),
        ]
        question = Question(
            qid="res_b_017", domain="research", split="B",
            question="汽车行业加速智能化、芯片设计走向ASIC定制、激光设备受益扩产、银行IT推进信创，哪些路径正确？",
            options={
                "A": "汽车自主可控主要体现为智驾芯片和算法自研，与ASIC定制服务直接相关",
                "B": "激光设备国产化率已接近100%，因此国产替代空间有限",
                "C": "银行IT从外围系统到核心系统，与汽车从辅助驾驶到完全自动驾驶的渐进路线相似",
                "D": "芯片IP授权模式与银行IT完全相同，都是购买现成软件快速替代",
            },
            answer_format="multi", type="多选题", doc_ids=["auto", "asic", "equipment", "bank"],
        )
        solver = self.make_solver(units)
        self.assert_labels(solver=solver, question=question, expected={"A": True, "B": False, "C": True, "D": False})

        evidence_by_option = {
            option_key: {hit.unit_id for hit in solver._rule_evaluate(question, option_text, [])[2]}
            for option_key, option_text in question.options.items()
        }
        self.assertEqual(evidence_by_option["A"], {"auto::assist_l3", "asic::adas", "asic::smart_car"})
        self.assertEqual(evidence_by_option["B"], {"equipment::share", "testing::share"})
        self.assertEqual(evidence_by_option["C"], {"auto::assist_l3", "auto::l4", "bank::stages"})
        self.assertEqual(evidence_by_option["D"], {"ip::module", "ip::custom", "bank::full_stack"})

    def test_source_control_bundle_distinguishes_operational_control_and_asset_ownership(self) -> None:
        units = [
            make_unit(
                "chicken::chain", "chicken",
                "公司构建覆盖种源育种、食品深加工至终端销售的全产业生态闭环；育种、饲料、养殖到屠宰、"
                "食品深加工各环节均为自有。下游订单需求反向指导上游养殖出栏节奏，实现供需精准匹配，"
                "并通过食品加工延伸下游增值链条。",
            ),
            make_unit(
                "commerce::quality", "commerce",
                "公司推出透明工厂并建立严格的质量检验机制，通过用户调研持续优化产品配方，确保原材料到终端产品"
                "全链条品质可控；供应链选品和提高品控能力支撑自营品发展。",
            ),
            make_unit(
                "commerce::store", "commerce",
                "线下旗舰店成为直播电商机构强化消费者品牌心智、打造长期品牌的重要载体。",
            ),
        ]
        question = Question(
            qid="res_b_002", domain="research", split="B",
            question="一家白羽肉鸡全产业链龙头与一家直播电商公司推行一体化或自营战略。",
            options={
                "A": "前者向上游延伸至种源育种，后者向上游延伸至产品配方和透明工厂，两者都通过控制源头建立品质壁垒",
                "B": "前者重资产、后者轻资产，因此后者不具备供应链控制力",
                "C": "线下旗舰店与深加工厂都向下游延伸，前者偏品牌体验、后者偏产品增值",
                "D": "原材料低迷时前者扩大出栏量，后者要求供应商降价维持毛利率",
            },
            answer_format="multi", type="多选题", doc_ids=["chicken", "commerce"],
        )
        self.assert_labels(solver=self.make_solver(units), question=question, expected={"A": True, "B": False, "C": True, "D": False})

    def test_brand_building_bundle_rejects_absolute_rankings_and_binds_customer_recognition(self) -> None:
        units = [
            make_unit(
                "chicken::brand", "chicken",
                "品牌+渠道双轮驱动形成品牌溢价，C端零售渠道快速增长，品牌矩阵持续完善且品牌价值不断提升。",
            ),
            make_unit(
                "pet::risk", "pet",
                "公司布局智能养宠硬件，但跨界布局缺乏成熟运营经验；宠物经济赛道竞争激烈、头部集中，"
                "存在产品同质化、市场教育及供应链壁垒。",
            ),
            make_unit(
                "commerce::transition", "commerce",
                "公司从流量驱动迈向产品驱动，长期信任关系要求优质内容供给与供应链选品共同支撑。",
            ),
            make_unit(
                "equipment::recognition", "equipment",
                "针对特征参数小的产品推出设备组合并获得国内外客户一致认可；高可靠加工方案具备产能与交付区位优势，"
                "已获得行业龙头认可。焊线机凭借稳定性上的优势获得行业龙头企业意向订单。",
            ),
        ]
        question = Question(
            qid="res_b_018", domain="research", split="B",
            question="四家分属不同行业的企业都面临品牌化挑战，哪些品牌化难度判断符合实际？",
            options={
                "A": "从B端向C端延伸的品牌化难度最大，因为农产品差异化小且鸡肉品牌认知低",
                "B": "宠物经济品牌化难度最小，因为宠物智能硬件仍是蓝海且已有全球渠道",
                "C": "从渠道品牌向产品品牌转型，需要维持内容热度与产品品质双轮驱动",
                "D": "设备品牌需要以技术参数、稳定性和持续交付能力建立客户认可",
            },
            answer_format="multi", type="多选题", doc_ids=["chicken", "pet", "commerce", "equipment"],
        )
        self.assert_labels(solver=self.make_solver(units), question=question, expected={"A": False, "B": False, "C": True, "D": True})

    def test_globalization_bundle_combines_multi_region_capacity_with_solution_output(self) -> None:
        units = [
            make_unit(
                "capacity::regions", "capacity",
                "头部企业形成产能落地与全球协同模式，同时布局欧洲市场和东南亚市场；欧洲项目契合欧盟本土化率要求，"
                "用于规避贸易壁垒；东南亚市场则利用劳动力成本优势和政策激励。",
            ),
            make_unit(
                "auto::standards", "auto",
                "合资平台由中方主导定义，中国开始反向输出技术标准。",
            ),
            make_unit(
                "rfid::solutions", "rfid",
                "企业通过定制化解决方案输出，为海外客户提供硬件+软件+实施+运维一体化服务，获取持续性收入；"
                "全球营销网络和国际收入体现定制化服务能力。",
            ),
            make_unit(
                "battery::risk", "battery",
                "为应对地缘政治风险并规避贸易风险，企业陆续在泰国建设工厂；美国关税变化也推动越南工厂和海外布局。",
            ),
        ]
        question = Question(
            qid="res_b_020", domain="research", split="B",
            question="多个行业提到全球化布局或出海战略，哪些中国企业出海逻辑有充分依据？",
            options={
                "A": "出海主要目的地集中在东南亚，因为劳动力成本低且贸易壁垒少",
                "B": "出海不仅是产能转移，更是技术标准和服务能力输出",
                "C": "服务消费通过跨境旅游和免税实现，与制造业出海模式完全不同",
                "D": "为对冲地缘风险和关税壁垒，在海外多区域布局产能已成为确定性趋势",
            },
            answer_format="multi", type="多选题", doc_ids=["capacity", "auto", "rfid", "battery"],
        )
        solver = self.make_solver(units)
        self.assert_labels(solver=solver, question=question, expected={"A": False, "B": True, "C": False, "D": True})

        b_result = solver._remaining_choice_evidence_bundle_rule(question, question.options["B"])
        d_result = solver._remaining_choice_evidence_bundle_rule(question, question.options["D"])
        self.assertIsNotNone(b_result)
        self.assertIsNotNone(d_result)
        self.assertEqual({hit.doc_id for hit in b_result[2]}, {"auto", "rfid"})
        self.assertEqual({hit.doc_id for hit in d_result[2]}, {"capacity", "battery"})

    def test_supply_constraint_bundle_binds_downstream_impact_and_duration_counterevidence(self) -> None:
        units = [
            make_unit(
                "chemical::event",
                "chemical",
                "伊朗已暂停所有石化产品出口，以确保国内供应减少情况下的内需保障。若油价上涨，油气及替代路线企业有望受益；"
                "风险包括地缘风险演化导致原材料价格波动以及行业产能发生重大变化。",
            ),
            make_unit(
                "optical::bottleneck",
                "optical",
                "关键的EML与CW-LD等光电芯片因产能配置问题陷入供应紧张，光学对准等高精度制程能力限制产能，"
                "供应商通过策略性长约锁定关键物料。",
            ),
            make_unit(
                "optical::duration",
                "optical",
                "LightCounting预计EML和CW激光器芯片的短缺将制约市场增长直至2026年底。",
            ),
            make_unit(
                "optical::advantage",
                "optical",
                "光模块头部厂商技术领先、客户关系稳固、具备规模化交付能力，优势将进一步凸显。"
                "光芯片研发和扩产周期长，具有较高的技术、人才、客户验证和资金壁垒，部分光芯片供需缺口持续扩大。",
            ),
            make_unit(
                "optical::substitution",
                "optical",
                "基于InP的EML的短缺正在加速向硅光的转型，但仍需要CW激光器；DR4和DR8可使产能提升30-50%，"
                "能够供应相关光源且具有供应能力的厂商数量增加。",
            ),
        ]
        question = Question(
            qid="res_b_003",
            domain="research",
            split="B",
            question="伊朗暂停石化产品出口，同期EML和CW激光器芯片短缺。这两类供给约束有何共同规律？",
            options={
                "A": "两者都存在供给收缩，且替代越困难，对下游成本与供货的影响越明显",
                "B": "供给约束使得拥有自主供应能力的企业获得竞争优势",
                "C": "光模块芯片短缺可以通过国产替代快速解决，而石化产品短缺则完全依赖地缘政治走向",
                "D": "两类供给约束的持续时间都将非常短暂，因为新增产能可在半年内快速释放",
            },
            answer_format="multi",
            type="多选题",
            doc_ids=["chemical", "optical"],
        )
        self.assert_labels(
            solver=self.make_solver(units),
            question=question,
            expected={"A": True, "B": True, "C": False, "D": False},
        )

    def test_market_fund_flow_bundle_binds_percentiles_and_investor_constraints(self) -> None:
        units = [
            make_unit(
                "flows::percentiles",
                "flows",
                "杠杆资金&股票型ETF分化加剧。两融资金净流入554亿元，处近三年95%分位；"
                "股票型ETF净申购-506亿元，处近三年3%分位。",
            ),
            make_unit(
                "flows::retail",
                "flows",
                "融资融券业务个人投资者数量达到811.1万名，其中平均每日参与交易的投资者数量达到46.8万名，"
                "较前值上升6.5万名，散户参与度上升。",
            ),
            make_unit(
                "flows::etf",
                "flows",
                "股票型ETF：净流入-506.4亿，处近三年2.6%分位。股票型ETF整体上周净流入-506.4亿，"
                "净流向整体处近三年2.6%分位。",
            ),
            make_unit(
                "wealth::stability",
                "wealth",
                "理财资金大幅增加了对公募基金和存款的配置，基金配置以债基和货基为主，对权益类基金配置很少；"
                "其负债端对波动的低容忍度决定了稳健为首要目标。",
            ),
            make_unit(
                "insurer::balance",
                "insurer",
                "新华保险满足资产负债匹配要求，通过配置长久期利率债收窄资产负债久期缺口，并增配高股息OCI类权益；"
                "高分红、低波动资产兼顾长期投资收益率与报表稳定性。",
            ),
        ]
        question = Question(
            qid="res_b_004",
            domain="research",
            split="B",
            question="近期两融净流入处于历史高位而ETF大幅净流出，同时银行理财增配债基、险资增配高股息股票。这些资金行为反映了当前市场的哪些深层特征？",
            options={
                "A": "不同资金方的风险偏好完全趋同，差异仅源于监管约束",
                "B": "个人投资者两融参与度上升，理财仍以稳健为纲，险资平衡长期收益与稳定回报",
                "C": "ETF净流出处于近三年极低分位，两融净流入处于近三年极高分位，两者分化程度达到极端水平",
                "D": "保险资金增配权益的同时也加强了资产负债久期匹配管理，并非放弃风险管理",
            },
            answer_format="multi",
            type="多选题",
            doc_ids=["flows", "wealth", "insurer"],
        )
        self.assert_labels(
            solver=self.make_solver(units),
            question=question,
            expected={"A": False, "B": True, "C": True, "D": True},
        )

    def test_risk_reallocation_bundle_separates_local_derisking_from_systemwide_appetite(self) -> None:
        units = [
            make_unit(
                "insurer::property_equity",
                "insurer",
                "公司持续增配低估值、高股息权益资产。不动产风险拨备充分、风险敞口实质收敛，"
                "不动产相关投资在总投资资产中的占比仅为3.1%。",
            ),
            make_unit(
                "bank::defensive",
                "bank",
                "金市配置减少基金投资，增配政府债券；金融投资的主要功能仍以流动性管理为主，"
                "结构变化中非标占比下降。",
            ),
            make_unit(
                "insurer::equity_high",
                "insurer",
                "保险公司的资产配置策略由其负债特性驱动，权益配置达到历史高位；高分红、低波动权益"
                "兼顾投资收益与报表稳定性的平衡。",
            ),
            make_unit(
                "margin::scope",
                "margin",
                "两融资金整体上周净流入约553.7亿元，处近三年95%分位，参与度处近三年82%分位；"
                "个人投资者数量达到811.1万名，散户参与度上升。",
            ),
            make_unit(
                "wealth::constraint",
                "wealth",
                "理财负债端对波动的低容忍度决定了稳健为首要目标，基金配置以债基和货基为主。",
            ),
            make_unit(
                "consumer::risk",
                "consumer",
                "2023年以来在居民风险偏好较低、预定利率多次下调、保险公司积极销售等因素推动下，"
                "银保渠道持续高增。",
            ),
        ]
        question = Question(
            qid="res_b_006",
            domain="research",
            split="B",
            question="一家综合金融集团的不动产投资占比已降至较低水平，同时保险行业整体大幅增配了高股息权益资产。结合当前银行业普遍增配政府债券、压缩主动负债的背景，以下哪些最能反映金融机构对风险资产态度的微妙变化？",
            options={
                "A": "压缩不动产敞口与银行增配政府债券的行为方向一致，都体现了风险偏好下降",
                "B": "保险资金在缩窄不动产的同时大幅增配权益，说明其整体风险偏好并未实质下降，只是在资产间进行风险置换",
                "C": "券商两融规模攀升表明整个金融体系的风险偏好已全面回升",
                "D": "银保渠道分红险的热销意味着消费者风险偏好已完全修复",
            },
            answer_format="multi",
            type="多选题",
            doc_ids=["insurer", "bank", "margin", "wealth", "consumer"],
        )
        self.assert_labels(
            solver=self.make_solver(units),
            question=question,
            expected={"A": True, "B": True, "C": False, "D": False},
        )

    def test_institutional_change_bundle_links_cost_exit_barriers_and_time_horizon(self) -> None:
        units = [
            make_unit(
                "life::compliance",
                "life",
                "在代理人规模扩张红利趋弱、渠道合规成本抬升与低利率常态化的背景下，传统人海战术面临约束；"
                "公司率先完成渠道清虚与产品结构调整。",
            ),
            make_unit(
                "bancassurance::exit",
                "bancassurance",
                "报行合一加速供给侧出清，头部集中度显著提升。费率严监管彻底击碎了中小险企依赖高费用换规模的竞争模型，"
                "头部公司在合规高压下成为银行首选并实现市场份额的逆势扩张。",
            ),
            make_unit(
                "service::subsidy",
                "service",
                "文旅等领域的消费补贴持续，消费乘数效应达到1:8.3，预计有望带动消费增长。",
            ),
            make_unit(
                "service::crowding",
                "service",
                "服务消费占比偏低主要由住房等刚性支出挤出，高居住成本挤压了我国服务消费支出空间。",
            ),
            make_unit(
                "bancassurance::network",
                "bancassurance",
                "受益于监管放开银保合作网点限制，公司利用品牌优势与国有大行及头部股份制银行建立合作，"
                "承接了外部渠道中小险企退出后的网点真空。",
            ),
            make_unit(
                "pet::network",
                "pet",
                "公司建设全国一体化医院网络，满足中国各地宠物主人的各种需求；遍布中国各地的社区宠物医院"
                "通过转介把客户引导至综合性或专科医院。",
            ),
            make_unit(
                "bancassurance::horizon",
                "bancassurance",
                "我国银保渠道自2000年启动以来，经历由政策驱动、产品驱动到结构转型的多阶段演进。",
            ),
            make_unit(
                "pet::horizon",
                "pet",
                "我国目前处于连锁化扩张的关键阶段，成熟模式走过40-80年的发展历程，国内企业转向谋求稳健发展。",
            ),
        ]
        question = Question(
            qid="res_b_008",
            domain="research",
            split="B",
            question="服务消费、宠物医疗、寿险、银保等行业均在经历制度变革。以下哪些对制度变革效果的判断是这些行业共同支持的？",
            options={
                "A": "监管政策的标准化通常会增加中小企业的合规成本，加速行业集中",
                "B": "财政补贴政策能够直接提升相关服务消费占比，且不存在挤出效应",
                "C": "银保渠道放开网点合作限制与宠物医院连锁化都是通过打破地域或渠道壁垒来提升头部企业份额",
                "D": "所有制度变革的效果都是立竿见影的，能在一年内完成行业格局重塑",
            },
            answer_format="multi",
            type="多选题",
            doc_ids=["life", "bancassurance", "service", "pet"],
        )
        self.assert_labels(
            solver=self.make_solver(units),
            question=question,
            expected={"A": True, "B": False, "C": True, "D": False},
        )

    def test_structural_cost_bundle_binds_cost_quality_and_durable_barriers_by_industry(self) -> None:
        units = [
            make_unit(
                "breeding::cost_quality", "breeding",
                "料肉比下降0.05，每年可节约饲料成本约3亿元；成活率提升并带来防疫成本下降。",
            ),
            make_unit(
                "breeding::control", "breeding",
                "育种、饲料、养殖到加工的产业链各环节均为自有，实现全产业链价值内部留存，"
                "并依托自主育种技术优势降低养殖成本。",
            ),
            make_unit(
                "breeding::quality", "breeding",
                "种源性能指标达到全球领先水平，并以自主育种技术优势降低上游养殖成本，"
                "以规模化养殖提升中游生产效率。",
            ),
            make_unit(
                "breeding::barrier", "breeding",
                "公司培育出拥有完全自主知识产权的种源，一举打破国外的垄断，成为世界第三大白羽肉鸡育种企业。",
            ),
            make_unit(
                "consumer::cost_quality", "consumer",
                "钛金属3D打印让表壳减少一半的原物料使用量，接口组件更轻薄、更坚固，"
                "比传统锻造制程节省33%的材料用量并大大降低了成本。",
            ),
            make_unit(
                "consumer::process", "consumer",
                "3D打印无需模具、节约原材料、制造周期短，因而具有显著的成本和效率优势。",
            ),
            make_unit(
                "consumer::quality", "consumer",
                "设备支持一体化成型，提高了结构强度和可靠性，并通过六振镜让吞吐量数倍提升。",
            ),
            make_unit(
                "consumer::barrier", "consumer",
                "企业在激光焊接关键工艺上形成不可复制、不可替代的优势，并满足材料机械性能提升要求。",
            ),
            make_unit(
                "optical::cost_quality", "optical",
                "全光交换架构的省电与省钱来自MEMS反射镜，功耗仅约100瓦，较传统交换机耗电量大幅减少约95%，并用于高速互连。",
            ),
            make_unit(
                "optical::upgrade", "optical",
                "若将带宽从800G提升至1.6T，只需更换高速光模块，升级成本将更具竞争力。",
            ),
            make_unit(
                "optical::quality", "optical",
                "硅光子具有低功耗、低延迟、高带宽、高集成度，可使产能提升，且性能和可靠性的提升也是优势。",
            ),
            make_unit(
                "optical::barrier", "optical",
                "关键的EML与CW-LD等光电芯片供应紧张，光学对准等高精度制程能力也是限制产能放大的因素。",
            ),
        ]
        question = Question(
            qid="res_b_009",
            domain="research",
            split="B",
            question="在养殖、消费电子制造和光通信领域，都出现了通过核心技术突破实现结构性降本的案例。以下关于这些案例的分析中，哪些是正确的？",
            options={
                "A": "都说明在产业链的关键环节进行技术突破能够带来显著的成本优势",
                "B": "都是通过掌控核心环节实现结构性降本，而非单纯压缩费用",
                "C": "降本的同时都带来了产品性能的提升，从而形成“降本+提质”双重优势",
                "D": "这些降本路径均可被竞争对手快速模仿，因此无法形成持续壁垒",
            },
            answer_format="multi",
            type="多选题",
            doc_ids=["breeding", "consumer", "optical"],
        )
        self.assert_labels(
            solver=self.make_solver(units),
            question=question,
            expected={"A": True, "B": True, "C": True, "D": False},
        )

    def test_service_consumption_bundle_aligns_dual_side_policy_and_risk_sharing(self) -> None:
        units = [
            make_unit(
                "service::long_term", "service",
                "服务消费政策体系正从“短期刺激”转向“长效制度建设”：需求端以长期制度安排使居民有闲、敢于消费；"
                "供给端扩围、提质、融合，并通过标准与品牌建设驱动产业升级和供给提质。",
            ),
            make_unit(
                "service::capacity", "service",
                "春秋假期等政策红利深挖全时段消费潜力，叠加景区业态扩容提质、文旅百业跨界融合，形成长期供给能力。",
            ),
            make_unit(
                "service::public_finance", "service",
                "提高中央财政在公共服务供给中的承担比例，提升民生领域公共支出，并在文旅等领域发放消费补贴。",
            ),
            make_unit(
                "service::public_policy", "service",
                "规划要求提高公共服务支出占财政支出比重、增加民生保障支出、健全公共卫生体系并完善医疗服务。",
            ),
            make_unit(
                "service::dual_side", "service",
                "需求端“增收、减负、清障”，以春秋假期和长护险释放潜能；供给端“扩围、提质、融合”提升服务品质。",
            ),
            make_unit(
                "pet::quality_network", "pet",
                "全国一体化医院网络由社区、综合和专业宠物医院构成，社区医院实行标准化的诊断及操作流程，"
                "并通过转介让客户获得高质量医疗及其他服务。",
            ),
            make_unit(
                "pet::scale_specialty", "pet",
                "连锁化是必然趋势，并购后必须重视服务标准化；专科化也是未来发展趋势，引入新技术提升诊疗效率。",
            ),
            make_unit(
                "insurance::risk_hedge", "insurance",
                "社会保障是风险对冲工具，通过分散养老、生育、医疗和长期照护风险，削弱预防性储蓄动机、抬升边际消费倾向。",
            ),
            make_unit(
                "insurance::care_release", "insurance",
                "长护险正重塑养老产业支付底座，补齐医疗—养老—护理短板，全面释放养老服务消费需求并成为支付引擎。",
            ),
        ]
        question = Question(
            qid="res_b_014",
            domain="research",
            split="B",
            question="推广春秋假和发放消费券、宠物医院连锁化和专科化、保险与养老健康服务结合，这些政策与商业实践背后体现了怎样的共同思路？以下哪些最准确？",
            options={
                "A": "通过时间和金钱的再分配来刺激短期消费，而非关注长期供给能力建设",
                "B": "将公共服务完全市场化，以减少财政负担",
                "C": "从供给侧和需求侧同时发力，以提升服务消费的质量和规模",
                "D": "利用金融工具来分担居民在养老、医疗等方面的支出风险，从而释放即期消费潜力",
            },
            answer_format="multi",
            type="多选题",
            doc_ids=["service", "pet", "insurance"],
        )
        self.assert_labels(
            solver=self.make_solver(units),
            question=question,
            expected={"A": False, "B": False, "C": True, "D": True},
        )


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
        results = {key: solver._choice_metric_bundle_rule(question, option) for key, option in options.items()}
        labels = {key: result[0] for key, result in results.items()}
        self.assertEqual(labels, {"A": True, "B": False, "C": False, "D": True})
        self.assertEqual(
            [item["unit_id"] for item in results["A"][2]][:3],
            ["catl::2025-balance", "catl::2025-interest", "catl::2024-interest"],
        )
        self.assertEqual(
            [item["unit_id"] for item in results["D"][2]][:3],
            ["byd::2025-balance", "byd::2025-interest", "byd::2024-interest"],
        )

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

    def test_reference_date_matrix_binds_commencement_and_transition_clauses(self) -> None:
        solver = self.make_solver(
            [
                make_unit(
                    "cdd::51",
                    "customer-diligence",
                    "第五十一条 对本办法施行前已经建立业务关系的存量客户，未满足本办法有关客户尽职调查要求的，金融机构应当自本办法施行之日起半年内完成较高风险以上存量客户的尽职调查，自本办法施行之日起2年内完成全部存量客户的尽职调查。",
                ),
                make_unit(
                    "bo::preamble",
                    "beneficial-owner",
                    "金融机构客户受益所有人识别管理办法现予公布，自2026年1月20日起施行。2025年12月19日。",
                ),
                make_unit(
                    "cdd::preamble",
                    "customer-diligence",
                    "金融机构客户尽职调查和客户身份资料及交易记录保存管理办法现予公布，自2026年1月1日起施行。2025年10月31日。",
                ),
                make_unit(
                    "aml::34",
                    "anti-money-laundering-law",
                    "第三十四条 客户身份资料在业务关系结束后、客户交易信息在交易结束后，应当至少保存十年。",
                ),
            ]
        )
        options = {
            "A": "较高风险以上存量客户应在半年内完成尽调",
            "B": "受益所有人识别新办法已于2026年1月15日生效",
            "C": "客户尽调新办法已于2026年1月15日生效",
            "D": "反洗钱法下客户身份资料至少保存十年",
        }
        question = Question(
            qid="reg_b_001",
            domain="regulatory",
            split="B",
            question="2026年1月15日，机构同时处理存量高风险客户尽调、受益所有人识别和客户资料保存。",
            options=options,
            answer_format="multi",
            type="多选题",
            doc_ids=["beneficial-owner", "customer-diligence", "anti-money-laundering-law"],
        )
        labels = {}
        evidence_ids = {}
        for key, option in options.items():
            hits = solver._targeted_literal_hits(question, option)
            evidence_ids[key] = {hit.unit_id for hit in hits}
            labels[key] = solver._targeted_rule_payload(option, hits)["label"]
        self.assertEqual(labels, {"A": True, "B": False, "C": True, "D": True})
        self.assertEqual(evidence_ids["A"], {"cdd::51", "cdd::preamble"})
        self.assertEqual(evidence_ids["B"], {"bo::preamble"})
        self.assertEqual(evidence_ids["C"], {"cdd::preamble"})
        self.assertEqual(evidence_ids["D"], {"aml::34"})


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
