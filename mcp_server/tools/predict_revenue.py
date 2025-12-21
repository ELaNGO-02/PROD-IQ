"""Revenue prediction tool handler."""

import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path (3 levels up: tools -> mcp_server -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_core.inference import InferenceEngine


async def predict_revenue_handler(data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle revenue prediction requests."""
    try:
        engine = InferenceEngine()
        result = engine.predict_revenue(data)

        return {
            "success": True,
            "model": "revenue_estimator_v1",
            "company": data.get("company_name", "Unknown"),
            "predictions": {
                "estimated_revenue": result.get("revenue", 0),
                "confidence_score": result.get("confidence", 0.0),
            },
            "input_data": {
                "funding_raised": data.get("funding_raised", 0),
                "team_size": data.get("team_size", 0),
                "category": data.get("category", "Unknown")
            }
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "model": "revenue_estimator_v1"
        }

