#!/usr/bin/env python3
"""
Journey Simulator - Simulate startup trajectory scenarios
"""
from typing import Dict, Any, List
import numpy as np


def simulate_journey(data: Dict[str, Any]) -> Dict:
    """
    Simulate 3 possible futures for the startup
    
    Args:
        data: Current startup state
    
    Returns:
        3 scenario simulations (optimistic, realistic, pessimistic)
    """
    
    current_revenue = data.get("current_revenue", 0)
    burn_rate = data.get("burn_rate_monthly_est", 0)
    funding = data.get("funding_raised", 0)
    team_size = data.get("team_size", 2)
    age_months = data.get("age_months", 6)
    
    # Calculate current state
    runway_months = funding / max(burn_rate, 1) if burn_rate > 0 else 36
    
    # Simulate 12 months forward with 3 scenarios
    scenarios = {}
    
    # === SCENARIO 1: OPTIMISTIC (30% monthly growth) ===
    optimistic = _simulate_scenario(
        current_revenue=current_revenue,
        burn_rate=burn_rate,
        funding=funding,
        growth_rate=0.30,
        burn_reduction=0.05,  # 5% burn reduction per month
        months=12,
        label="Optimistic"
    )
    scenarios["optimistic"] = optimistic
    
    # === SCENARIO 2: REALISTIC (15% monthly growth) ===
    realistic = _simulate_scenario(
        current_revenue=current_revenue,
        burn_rate=burn_rate,
        funding=funding,
        growth_rate=0.15,
        burn_reduction=0.02,
        months=12,
        label="Realistic"
    )
    scenarios["realistic"] = realistic
    
    # === SCENARIO 3: PESSIMISTIC (5% monthly growth) ===
    pessimistic = _simulate_scenario(
        current_revenue=current_revenue,
        burn_rate=burn_rate,
        funding=funding,
        growth_rate=0.05,
        burn_reduction=0.00,  # No burn reduction
        months=12,
        label="Pessimistic"
    )
    scenarios["pessimistic"] = pessimistic
    
    # Determine recommended path
    breakeven_month_realistic = realistic.get("breakeven_month", None)
    
    if breakeven_month_realistic and breakeven_month_realistic <= 12:
        recommendation = "realistic"
        confidence = 0.75
    elif optimistic.get("breakeven_month", None) and optimistic["breakeven_month"] <= 12:
        recommendation = "optimistic"
        confidence = 0.55
    else:
        recommendation = "fundraising_required"
        confidence = 0.90
    
    return {
        "scenarios": scenarios,
        "recommendation": recommendation,
        "confidence": confidence,
        "current_state": {
            "revenue": current_revenue,
            "burn": burn_rate,
            "runway_months": round(runway_months, 1),
            "team_size": team_size,
            "age_months": age_months
        }
    }


def _simulate_scenario(
    current_revenue: float,
    burn_rate: float,
    funding: float,
    growth_rate: float,
    burn_reduction: float,
    months: int,
    label: str
) -> Dict:
    """Simulate a single scenario month-by-month"""
    
    monthly_data = []
    revenue = current_revenue
    burn = burn_rate
    cash = funding
    
    for month in range(1, months + 1):
        # Grow revenue
        revenue = revenue * (1 + growth_rate)
        
        # Reduce burn (efficiency gains)
        burn = burn * (1 - burn_reduction)
        
        # Calculate cash flow
        net_cashflow = revenue - burn
        cash += net_cashflow
        
        monthly_data.append({
            "month": month,
            "revenue": round(revenue, 2),
            "burn": round(burn, 2),
            "net_cashflow": round(net_cashflow, 2),
            "cash": round(cash, 2),
            "runway_months": round(cash / burn, 1) if burn > 0 else 999
        })
        
        # Check if broke
        if cash < 0:
            break
    
    # Find breakeven month
    breakeven_month = None
    for m in monthly_data:
        if m["net_cashflow"] >= 0:
            breakeven_month = m["month"]
            break
    
    # Calculate 12-month outcome
    final_state = monthly_data[-1] if monthly_data else {}
    
    return {
        "label": label,
        "growth_rate": f"{growth_rate*100}%",
        "monthly_trajectory": monthly_data,
        "breakeven_month": breakeven_month,
        "final_revenue": final_state.get("revenue", 0),
        "final_cash": final_state.get("cash", 0),
        "survives": cash > 0
    }
