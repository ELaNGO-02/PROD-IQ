"""Traction prediction tool handler."""

from typing import Dict, Any
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent if 'mcp_server' in str(Path(__file__)) else Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_core.inference import InferenceEngine


async def predict_traction_handler(data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle traction prediction requests."""
    try:
        engine = InferenceEngine()
        result = engine.predict_traction(data)

        return {
            "success": True,
            "model": "traction_time_v1",
            "company": data["company_name"],
            "predictions": {
                "months_to_traction": result.get("months", 0),
                "traction_date": result.get("date", "Unknown"),
                "confidence_score": result.get("confidence", 0.0)
            },
            "growth_metrics": {
                "user_growth_trajectory": result.get("user_trajectory", "Linear"),
                "revenue_growth_trajectory": result.get("revenue_trajectory", "Linear"),
                "market_penetration": result.get("market_penetration", "Low")
            },
            "traction_indicators": {
                "customer_acquisition_cost": result.get("cac", 0),
                "lifetime_value": result.get("ltv", 0),
                "viral_coefficient": result.get("viral_coef", 0.0)
            },
            "recommendations": result.get("recommendations", [
                "Optimize customer acquisition channels",
                "Improve product-market fit",
                "Accelerate growth tactics"
            ])
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model": "traction_time_v1"
        }
