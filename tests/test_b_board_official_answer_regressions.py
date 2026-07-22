from __future__ import annotations

import unittest

from afa_agent.domains.financial_contracts.solver import FinancialContractsSolver
from afa_agent.domains.generic_retriever import GenericBM25Retriever
from afa_agent.domains.insurance.solver import InsuranceSolver
from afa_agent.models import Question


def make_unit(unit_id: str, doc_id: str, text: str) -> dict[str, object]:
    return {
        "unit_id": unit_id,
        "doc_id": doc_id,
        "domain": "test",
        "unit_type": "paragraph",
        "title_path": [doc_id],
        "text": text,
        "page_refs": [],
        "parent_unit_id": None,
        "metadata": {},
    }


class ConfirmedOfficialAnswerRegressionTests(unittest.TestCase):
    def test_changan_bank_risk_management_retrieves_and_supports_all_options(self) -> None:
        units = [
            make_unit(
                "text12::credit",
                "text12",
                "目前，长安银行信用风险主要集中在贷款业务、债券投资业务及承诺与担保等表内外业务。",
            ),
            make_unit(
                "text12::liquidity",
                "text12",
                "长安银行的流动性风险主要来自存款人提前或集中提款、保本理财产品到期兑付、借款人延期偿还贷款、资产负债的金额与到期日错配等。",
            ),
            make_unit(
                "text12::board",
                "text12",
                "董事会为长安银行风险管理的最高决策机构，下设董事会风险管理委员会，形成全面风险管理组织架构。",
            ),
            make_unit(
                "text12::defenses",
                "text12",
                "公司建立以业务管理、风险合规及审计监督为三道防线的内部控制体系：各业务部门是第一道防线，各类风险主管部门是第二道防线，审计部门是第三道防线。",
            ),
            make_unit("text14::noise", "text14", "其他发行人的风险管理制度。"),
        ]
        solver = FinancialContractsSolver.__new__(FinancialContractsSolver)
        solver.retriever = GenericBM25Retriever(units)
        question = Question(
            qid="fc_b_019",
            domain="financial_contracts",
            split="B",
            question="关于长安银行的风险管理，以下说法正确的有？",
            options={
                "A": "信用风险主要集中在贷款业务、债券投资及表外承诺与担保等业务",
                "B": "流动性风险主要来自存款人提前或集中提款、资产负债期限错配等",
                "C": "董事会是风险管理的最高决策机构，下设风险管理委员会",
                "D": "操作风险管理体系采用三道防线架构，即业务部门、风险主管部门和审计部门",
            },
            answer_format="multi",
            type="多选题",
            doc_ids=["text12", "text14"],
        )

        labels: dict[str, bool] = {}
        for option, text in question.options.items():
            hits = solver._targeted_literal_hits(question, option, text)
            self.assertTrue(hits, option)
            self.assertTrue(all(hit.doc_id == "text12" for hit in hits), option)
            rule = solver._rule_override(question, option, text, hits)
            self.assertIsNotNone(rule, option)
            assert rule is not None
            labels[option] = bool(rule["label"])

        self.assertEqual(labels, {"A": True, "B": True, "C": True, "D": True})


class InsuranceOfficialClauseRegressionTests(unittest.TestCase):
    @staticmethod
    def solve(question: str, options: dict[str, str], units: list[dict[str, object]]) -> str:
        solver = InsuranceSolver.__new__(InsuranceSolver)
        solver.retriever = GenericBM25Retriever(units)
        solver.answering_settings = {"prompt_template_id": "test", "max_hits": 16}
        item = Question(
            qid="regression",
            domain="insurance",
            split="B",
            question=question,
            options=options,
            answer_format="multi",
            type="多选题",
            doc_ids=sorted({str(unit["doc_id"]) for unit in units}),
        )
        answer = solver._solve_product_identity_clause_bundle(item)
        assert answer is not None
        if answer.token_usage.total_tokens != 0:
            raise AssertionError("deterministic clause bundle unexpectedly used model tokens")
        return answer.pred_answer

    def test_terrorism_exclusion_is_bcd(self) -> None:
        units = [
            make_unit(
                "1::all",
                "1",
                "中国人寿保险股份有限公司 国寿增益宝 第七条责任免除：投保人故意杀害；2年内自杀；酒后驾驶；核爆炸。",
            ),
            make_unit("8::all", "8", "众安在线财产保险股份有限公司 营运交通工具团体意外伤害保险 责任免除包括恐怖袭击。"),
            make_unit("10::all", "10", "众安在线财产保险股份有限公司 特种车商业保险示范条款 责任免除包括恐怖活动。"),
            make_unit(
                "12::all",
                "12",
                "众安在线财产保险股份有限公司 家庭财产综合保险 下列原因造成的损失，保险人不负责赔偿：恐怖活动。",
            ),
        ]
        answer = self.solve(
            "关于恐怖活动/恐怖袭击责任免除，下列产品明确列明该项的是？",
            {"A": "国寿增益宝", "B": "众安营运交通工具团体意外伤害保险", "C": "众安特种车商业保险", "D": "众安家庭财产综合保险"},
            units,
        )
        self.assertEqual(answer, "BCD")

    def test_earthquake_exclusion_is_bcd(self) -> None:
        units = [
            make_unit("2::all", "2", "平安安佑福 重大疾病保险 责任免除包括核爆炸、酒后驾驶。"),
            make_unit("11::all", "11", "中国平安财产保险股份有限公司 家庭财产保险 家庭版 责任免除包括地震、海啸。"),
            make_unit("12::all", "12", "众安在线财产保险股份有限公司 家庭财产综合保险 责任免除包括地震、海啸及次生灾害。"),
            make_unit("14::all", "14", "中国平安财产保险股份有限公司 食品安全责任保险 责任免除包括地震等自然灾害。"),
        ]
        answer = self.solve(
            "关于地震相关免责或除外责任，下列产品明确列明的是？",
            {"A": "平安安佑福重疾险", "B": "平安家庭财产保险", "C": "众安家庭财产综合保险", "D": "平安食品安全责任保险"},
            units,
        )
        self.assertEqual(answer, "BCD")

    def test_two_year_suicide_exclusion_is_abd(self) -> None:
        units = [
            make_unit("1::all", "1", "中国人寿保险股份有限公司 国寿增益宝 合同成立或效力恢复起2年内自杀免责，无民事行为能力人除外。"),
            make_unit("2::all", "2", "平安安佑福 重大疾病保险 合同成立或效力恢复起2年内自杀免责，无民事行为能力人除外。"),
            make_unit("8::all", "8", "众安在线财产保险股份有限公司 营运交通工具团体意外伤害保险 责任免除包括恐怖袭击、依法拘留。"),
            make_unit("16::all", "16", "平安养老保险股份有限公司 平安富鸿金生 合同成立或效力恢复起2年内自杀免责，无民事行为能力人除外。"),
        ]
        answer = self.solve(
            "关于自本合同成立或复效起2年内自杀，但无民事行为能力人除外的责任免除表述，下列产品明确包含该规则的是？",
            {"A": "国寿增益宝", "B": "平安安佑福重疾险", "C": "众安营运交通工具团体意外伤害保险", "D": "平安富鸿金生养老年金保险"},
            units,
        )
        self.assertEqual(answer, "ABD")

    def test_nuclear_exclusion_is_abcd(self) -> None:
        units = [
            make_unit("2::all", "2", "平安安佑福 重大疾病保险 责任免除包括核爆炸、核辐射或核污染。"),
            make_unit("5::all", "5", "平安e生保 医疗保险 责任免除包括核爆炸、核辐射与核污染。"),
            make_unit("6::all", "6", "太平洋健康保险股份有限公司 团体百万医疗保险 责任免除包括核爆炸、核辐射或核污染。"),
            make_unit("9::all", "9", "中国平安财产保险股份有限公司 特种车商业保险示范条款 责任免除包括核反应、核辐射及放射性污染。"),
        ]
        answer = self.solve(
            "关于核爆炸、核辐射或核污染等核风险免责，下列产品明确列明的是？",
            {"A": "平安安佑福重疾险", "B": "平安e生保", "C": "太保团体百万医疗", "D": "平安特种车商业保险"},
            units,
        )
        self.assertEqual(answer, "ABCD")


if __name__ == "__main__":
    unittest.main()
