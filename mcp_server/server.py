"""
MCP Server for Startup Analyzer using FastMCP
"""

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP
from typing import Dict, Any, Optional

# Import from ml_core (with underscore) and tools
from ml_core.inference import InferenceEngine
from tools.predict_revenue import predict_revenue_handler
from tools.predict_success import predict_success_handler
from tools.predict_breakeven import predict_breakeven_handler
from tools.predict_survival import predict_survival_handler
from tools.predict_traction import predict_traction_handler
from tools.market_scout import market_scout_handler
from tools.journey_simulator import simulate_journey

# Initialize FastMCP server
mcp = FastMCP("prod-iq")


# ============================================================================
# PREDICTION TOOLS
# ============================================================================

@mcp.tool()
async def predict_revenue(
        company_name: str,
        category: str,
        funding_raised: float,
        team_size: int,
        founding_year: int,
        monthly_active_users: Optional[int] = None,
        description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Predict the estimated revenue for a startup.

    Args:
        company_name: Name of the startup
        category: Business category (e.g., SaaS, E-commerce, FinTech)
        funding_raised: Total funding raised in USD
        team_size: Number of team members
        founding_year: Year the company was founded
        monthly_active_users: Number of monthly active users (optional)
        description: Company description (optional)

    Returns:
        Dictionary with predicted revenue and confidence metrics
    """
    return await predict_revenue_handler({
        "company_name": company_name,
        "category": category,
        "funding_raised": funding_raised,
        "team_size": team_size,
        "founding_year": founding_year,
        "monthly_active_users": monthly_active_users,
        "description": description
    })


@mcp.tool()
async def predict_success(
        company_name: str,
        category: str,
        funding_raised: float,
        team_size: int,
        founding_year: int,
        market_size: str,
        competition_level: str,
        description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Predict the success probability of a startup.

    Args:
        company_name: Name of the startup
        category: Business category
        funding_raised: Total funding raised in USD
        team_size: Number of team members
        founding_year: Year the company was founded
        market_size: Market size category (Small, Medium, Large)
        competition_level: Competition level (Low, Medium, High)
        description: Company description (optional)

    Returns:
        Dictionary with success probability and risk factors
    """
    return await predict_success_handler({
        "company_name": company_name,
        "category": category,
        "funding_raised": funding_raised,
        "team_size": team_size,
        "founding_year": founding_year,
        "market_size": market_size,
        "competition_level": competition_level,
        "description": description
    })


@mcp.tool()
async def predict_breakeven(
        company_name: str,
        category: str,
        funding_raised: float,
        monthly_burn_rate: float,
        monthly_revenue: float,
        team_size: int,
        description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Predict when a startup will reach breakeven point.

    Args:
        company_name: Name of the startup
        category: Business category
        funding_raised: Total funding raised in USD
        monthly_burn_rate: Monthly cash burn rate in USD
        monthly_revenue: Current monthly revenue in USD
        team_size: Number of team members
        description: Company description (optional)

    Returns:
        Dictionary with breakeven timeline and financial projections
    """
    return await predict_breakeven_handler({
        "company_name": company_name,
        "category": category,
        "funding_raised": funding_raised,
        "monthly_burn_rate": monthly_burn_rate,
        "monthly_revenue": monthly_revenue,
        "team_size": team_size,
        "description": description
    })


@mcp.tool()
async def predict_survival(
        company_name: str,
        category: str,
        funding_raised: float,
        months_operating: int,
        runway_months: int,
        team_size: int,
        has_revenue: bool,
        description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Predict the survival probability of a startup over different time periods.

    Args:
        company_name: Name of the startup
        category: Business category
        funding_raised: Total funding raised in USD
        months_operating: Number of months the company has been operating
        runway_months: Number of months of runway remaining
        team_size: Number of team members
        has_revenue: Whether the company has generated revenue
        description: Company description (optional)

    Returns:
        Dictionary with survival probabilities and risk assessment
    """
    return await predict_survival_handler({
        "company_name": company_name,
        "category": category,
        "funding_raised": funding_raised,
        "months_operating": months_operating,
        "runway_months": runway_months,
        "team_size": team_size,
        "has_revenue": has_revenue,
        "description": description
    })


@mcp.tool()
async def predict_traction(
        company_name: str,
        category: str,
        founding_year: int,
        funding_raised: float,
        team_size: int,
        user_growth_rate: Optional[float] = None,
        revenue_growth_rate: Optional[float] = None,
        description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Predict the time to achieve significant traction.

    Args:
        company_name: Name of the startup
        category: Business category
        founding_year: Year the company was founded
        funding_raised: Total funding raised in USD
        team_size: Number of team members
        user_growth_rate: Monthly user growth rate as percentage (optional)
        revenue_growth_rate: Monthly revenue growth rate as percentage (optional)
        description: Company description (optional)

    Returns:
        Dictionary with traction timeline and growth projections
    """
    return await predict_traction_handler({
        "company_name": company_name,
        "category": category,
        "founding_year": founding_year,
        "funding_raised": funding_raised,
        "team_size": team_size,
        "user_growth_rate": user_growth_rate,
        "revenue_growth_rate": revenue_growth_rate,
        "description": description
    })


# ============================================================================
# ANALYSIS TOOLS
# ============================================================================

@mcp.tool()
async def market_scout(
        company_name: str,
        category: str,
        target_market: str,
        description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze market opportunities and competitive landscape.

    Args:
        company_name: Name of the startup
        category: Business category
        target_market: Target market or geography
        description: Company description (optional)

    Returns:
        Dictionary with market analysis, opportunities, and competitor insights
    """
    return await market_scout_handler({
        "company_name": company_name,
        "category": category,
        "target_market": target_market,
        "description": description
    })


@mcp.tool()
async def story_weaver(
        company_name: str,
        category: str,
        funding_raised: float,
        team_size: int,
        achievements: Optional[str] = None,
        description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a compelling narrative and insights about a startup.

    Args:
        company_name: Name of the startup
        category: Business category
        funding_raised: Total funding raised in USD
        team_size: Number of team members
        achievements: Key achievements or milestones (optional)
        description: Company description (optional)

    Returns:
        Dictionary with narrative summary, key highlights, and recommendations
    """
    return await simulate_journey({
        "company_name": company_name,
        "category": category,
        "funding_raised": funding_raised,
        "team_size": team_size,
        "achievements": achievements,
        "description": description
    })


# ============================================================================
# COMBINED ANALYSIS
# ============================================================================

@mcp.tool()
async def analyze_startup_complete(
        company_name: str,
        category: str,
        funding_raised: float,
        team_size: int,
        founding_year: int,
        monthly_burn_rate: float,
        monthly_revenue: float,
        market_size: str,
        competition_level: str,
        description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run comprehensive analysis on a startup using all prediction models.

    Args:
        company_name: Name of the startup
        category: Business category
        funding_raised: Total funding raised in USD
        team_size: Number of team members
        founding_year: Year the company was founded
        monthly_burn_rate: Monthly cash burn rate in USD
        monthly_revenue: Current monthly revenue in USD
        market_size: Market size category (Small, Medium, Large)
        competition_level: Competition level (Low, Medium, High)
        description: Company description (optional)

    Returns:
        Dictionary with comprehensive analysis from all models
    """
    base_data = {
        "company_name": company_name,
        "category": category,
        "funding_raised": funding_raised,
        "team_size": team_size,
        "founding_year": founding_year,
        "description": description
    }

    # Run all predictions
    results = {
        "company_info": {
            "name": company_name,
            "category": category,
            "funding": funding_raised,
            "team_size": team_size
        },
        "predictions": {}
    }

    try:
        # Revenue prediction
        results["predictions"]["revenue"] = await predict_revenue_handler({
            **base_data,
            "monthly_active_users": None
        })

        # Success prediction
        results["predictions"]["success"] = await predict_success_handler({
            **base_data,
            "market_size": market_size,
            "competition_level": competition_level
        })

        # Breakeven prediction
        results["predictions"]["breakeven"] = await predict_breakeven_handler({
            **base_data,
            "monthly_burn_rate": monthly_burn_rate,
            "monthly_revenue": monthly_revenue
        })

        # Survival prediction
        months_operating = 2025 - founding_year
        runway_months = funding_raised / monthly_burn_rate if monthly_burn_rate > 0 else 0

        results["predictions"]["survival"] = await predict_survival_handler({
            **base_data,
            "months_operating": int(months_operating * 12),
            "runway_months": int(runway_months),
            "has_revenue": monthly_revenue > 0
        })

        # Overall assessment
        results["overall_assessment"] = {
            "status": "Analysis complete",
            "timestamp": "2025-12-08",
            "models_used": 4
        }

    except Exception as e:
        results["error"] = str(e)

    return results


# ============================================================================
# SERVER ENTRY POINT
# ============================================================================

def main():
    """Start the MCP server."""
    print("🚀 Starting Startup Analyzer MCP Server...", file=sys.stderr)
    print(f"   Server: startup-analyzer v1.0.0", file=sys.stderr)
    print(f"   Tools: 8 available", file=sys.stderr)
    mcp.run(transport='stdio')


if __name__ == "__main__":
    main()
