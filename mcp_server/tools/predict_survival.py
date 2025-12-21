"""Survival prediction tool handler."""

from typing import Dict, Any
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent if 'mcp_server' in str(Path(__file__)) else Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_core.inference import InferenceEngine


async def predict_survival_handler(data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle survival prediction requests."""
    try:
        engine = InferenceEngine()
        result = engine.predict_survival(data)

        return {
            "success": True,
            "model": "survival_month_v1",
            "company": data["company_name"],
            "predictions": {
                "survival_probability": {
                    "6_months": result.get("prob_6m", 0.0),
                    "12_months": result.get("prob_12m", 0.0),
                    "24_months": result.get("prob_24m", 0.0),
                    "36_months": result.get("prob_36m", 0.0)
                },
                "overall_risk": result.get("risk_level", "Medium")
            },
            "critical_factors": {
                "runway_adequate": data["runway_months"] > 12,
                "has_revenue": data["has_revenue"],
                "funding_level": "Good" if data["funding_raised"] > 500000 else "Limited"
            },
            "milestones": result.get("milestones", [
                "Achieve product-market fit",
                "Reach breakeven",
                "Secure Series A funding"
            ]),
            "warnings": result.get("warnings", []),
            "recommendations": result.get("recommendations", [
                "Extend runway immediately",
                "Focus on revenue generation",
                "Consider pivot if traction is low"
            ])
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model": "survival_month_v1"
        }
