"""
Maps MCP tool inputs → FeatureEngine raw_input format
"""

from typing import Dict, Any

def adapt_mcp_input(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": data.get("company_name", "Unknown Startup"),
        "description": data.get("description", "No description provided"),
        "main_category": data.get("category", "other").lower(),
        "total_funding": float(data.get("funding_raised", 0)),
        "team_size": int(data.get("team_size", 2)),
        "price": 0,  # default, unless user provides
        "country": "unknown",
        "business_model": "subscription",
        "revenue_monthly": float(data.get("monthly_revenue", 0)),
        "launch_date": None,
    }
