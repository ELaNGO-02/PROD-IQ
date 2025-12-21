"""Success prediction tool handler."""

from typing import Dict, Any
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent if 'mcp_server' in str(Path(__file__)) else Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_core.inference import InferenceEngine


async def predict_success_handler(data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle success prediction requests."""
    try:
        engine = InferenceEngine()
        result = engine.predict_success(data)

        return {
            "success": True,
            "model": "success_label_v1",
            "company": data["company_name"],
            "predictions": {
                "success_probability": result.get("probability", 0.0),
                "success_category": result.get("category", "Unknown"),
                "confidence_score": result.get("confidence", 0.0)
            },
            "risk_factors": {
                "market_risk": result.get("market_risk", "Medium"),
                "competition_risk": result.get("competition_risk", "Medium"),
                "execution_risk": result.get("execution_risk", "Medium"),
                "financial_risk": result.get("financial_risk", "Medium")
            },
            "strengths": result.get("strengths", [
                "Strong team",
                "Good market timing",
                "Adequate funding"
            ]),
            "challenges": result.get("challenges", [
                "High competition",
                "Market saturation"
            ]),
            "recommendations": result.get("recommendations", [
                "Focus on differentiation",
                "Build strong customer relationships",
                "Monitor burn rate carefully"
            ])
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model": "success_label_v1"
        }
