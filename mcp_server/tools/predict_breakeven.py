"""Breakeven prediction tool handler."""

from typing import Dict, Any
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent if 'mcp_server' in str(Path(__file__)) else Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_core.inference import InferenceEngine


async def predict_breakeven_handler(data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle breakeven prediction requests."""
    try:
        engine = InferenceEngine()
        result = engine.predict_breakeven(data)

        return {
            "success": True,
            "model": "breakeven_time_v1",
            "company": data["company_name"],
            "predictions": {
                "breakeven_months": result.get("months", 0),
                "breakeven_date": result.get("date", "Unknown"),
                "confidence_score": result.get("confidence", 0.0)
            },
            "financial_metrics": {
                "current_runway_months": data["funding_raised"] / data["monthly_burn_rate"] if data[
                                                                                                   "monthly_burn_rate"] > 0 else 0,
                "monthly_gap": data["monthly_burn_rate"] - data["monthly_revenue"],
                "funding_needed": result.get("additional_funding_needed", 0)
            },
            "projections": {
                "revenue_at_breakeven": result.get("breakeven_revenue", 0),
                "team_size_at_breakeven": result.get("projected_team_size", data["team_size"])
            },
            "recommendations": result.get("recommendations", [
                "Reduce burn rate",
                "Accelerate revenue growth",
                "Consider additional funding round"
            ])
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model": "breakeven_time_v1"
        }
