#!/usr/bin/env python3
"""
Competitor Analysis Tool - Deep competitive intelligence
"""
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.hybrid_search import HybridSearch


async def market_scout_handler(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform deep competitor analysis
    
    Args:
        data: Contains company_name, category, target_market, description
    
    Returns:
        Comprehensive competitive analysis
    """
    try:
        company_name = data.get("company_name", "Unknown")
        category = data.get("category", "saas")
        description = data.get("description", f"A {category} startup")
        
        # Initialize database connection
        hybrid_search = HybridSearch()
        
        # Find competitors using vector search
        competitors = hybrid_search.find_competitors(
            query_text=description,
            category=category,
            n_results=15
        )
        
        if not competitors or len(competitors) == 0:
            return {
                "success": True,
                "tool": "competitor_analysis",
                "company": company_name,
                "verdict": "blue_ocean",
                "competitor_count": 0,
                "message": "You're in an emerging market with few established players",
                "strategy": "land_grab",
                "top_competitors": [],
                "recommendations": [
                    "Focus on rapid user acquisition",
                    "Build network effects early",
                    "Establish category leadership"
                ]
            }
        
        # Analyze competitor landscape
        analysis = _analyze_landscape(competitors)
        
        # Generate positioning strategy
        positioning = _generate_positioning(analysis, data)
        
        return {
            "success": True,
            "tool": "competitor_analysis",
            "company": company_name,
            "verdict": analysis["verdict"],
            "competitor_count": len(competitors),
            "top_competitors": [
                {
                    "name": c.get("name", "Unknown"),
                    "revenue": c.get("revenue_monthly") or c.get("revenue_estimated"),
                    "funding": c.get("total_funding"),
                    "team_size": c.get("team_size"),
                    "similarity_score": c.get("similarity_score", 0),
                    "source": c.get("source")
                }
                for c in competitors[:5]
            ],
            "market_analysis": analysis,
            "positioning_strategy": positioning,
            "recommendations": _generate_recommendations(analysis, positioning)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tool": "competitor_analysis"
        }


def _analyze_landscape(competitors: List[Dict]) -> Dict:
    """Analyze competitive landscape metrics"""
    
    revenues = []
    fundings = []
    team_sizes = []
    
    for comp in competitors:
        # Extract revenue
        revenue = comp.get("revenue_monthly") or comp.get("revenue_estimated")
        if revenue:
            try:
                revenues.append(float(revenue))
            except:
                pass
        
        # Extract funding
        funding = comp.get("total_funding")
        if funding:
            try:
                fundings.append(float(funding))
            except:
                pass
        
        # Extract team size
        team_size = comp.get("team_size")
        if team_size:
            try:
                team_sizes.append(int(team_size))
            except:
                pass
    
    # Calculate market statistics
    import numpy as np
    
    result = {
        "total_competitors": len(competitors),
        "avg_revenue": round(np.mean(revenues), 2) if revenues else None,
        "median_revenue": round(np.median(revenues), 2) if revenues else None,
        "avg_funding": round(np.mean(fundings), 2) if fundings else None,
        "avg_team_size": round(np.mean(team_sizes), 1) if team_sizes else None,
    }
    
    # Determine market maturity
    if len(competitors) < 5:
        result["verdict"] = "emerging_market"
        result["maturity"] = "Early Stage"
    elif len(competitors) < 20:
        result["verdict"] = "growing_market"
        result["maturity"] = "Growth Stage"
    else:
        result["verdict"] = "crowded_market"
        result["maturity"] = "Mature"
    
    return result


def _generate_positioning(analysis: Dict, user_data: Dict) -> Dict:
    """Generate positioning strategy based on competitive landscape"""
    
    verdict = analysis.get("verdict", "unknown")
    
    strategies = {
        "emerging_market": {
            "primary_strategy": "land_grab",
            "focus": "Rapid user acquisition to establish category leadership",
            "tactics": ["Network effects", "Content marketing", "Community building"]
        },
        "growing_market": {
            "primary_strategy": "differentiation",
            "focus": "Unique positioning with distinct value proposition",
            "tactics": ["Product-led growth", "Partnerships", "Niche focus"]
        },
        "crowded_market": {
            "primary_strategy": "niche_domination",
            "focus": "Own a specific vertical or use case",
            "tactics": ["Vertical SaaS", "Premium positioning", "API-first"]
        }
    }
    
    strategy = strategies.get(verdict, strategies["growing_market"])
    
    return {
        "primary_strategy": strategy["primary_strategy"],
        "focus_area": strategy["focus"],
        "recommended_tactics": strategy["tactics"],
        "timeframe": "6-12 months"
    }


def _generate_recommendations(analysis: Dict, positioning: Dict) -> List[str]:
    """Generate actionable recommendations"""
    
    recommendations = []
    
    verdict = analysis.get("verdict", "unknown")
    
    if verdict == "emerging_market":
        recommendations = [
            "Move fast to capture market share",
            "Build strong brand presence early",
            "Focus on customer retention over acquisition cost"
        ]
    elif verdict == "growing_market":
        recommendations = [
            "Identify underserved customer segment",
            "Build unique features competitors lack",
            "Establish strategic partnerships"
        ]
    else:  # crowded_market
        recommendations = [
            "Focus on niche vertical with specific pain points",
            "Compete on execution and customer experience",
            "Consider premium positioning with white-glove service"
        ]
    
    return recommendations
