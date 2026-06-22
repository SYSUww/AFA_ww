from __future__ import annotations

from afa_agent.domains.financial_contracts import FinancialContractsPlugin
from afa_agent.domains.financial_reports import FinancialReportsPlugin
from afa_agent.domains.insurance import InsurancePlugin
from afa_agent.domains.regulatory import RegulatoryPlugin
from afa_agent.domains.research import ResearchPlugin


PLUGIN_REGISTRY = {
    "regulatory": RegulatoryPlugin,
    "financial_reports": FinancialReportsPlugin,
    "insurance": InsurancePlugin,
    "research": ResearchPlugin,
    "financial_contracts": FinancialContractsPlugin,
}


def get_plugin(domain: str):
    if domain not in PLUGIN_REGISTRY:
        raise ValueError(f"Unsupported domain: {domain}")
    return PLUGIN_REGISTRY[domain]()
