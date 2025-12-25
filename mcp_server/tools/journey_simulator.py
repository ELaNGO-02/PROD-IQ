#!/usr/bin/env python3
"""
Journey Simulator - Simulate startup trajectory scenarios
Provides data for LLM to create narrative story
"""
from typing import Dict, Any, List
import json


def simulate_journey(data: Dict[str, Any]) -> Dict:
    """
    Simulate 3 possible futures for the startup
    
    Args:
        data: Current startup state
    
    Returns:
        3 scenario simulations (optimistic, realistic, pessimistic)
    """
    
    # Extract current state with safe defaults
    current_revenue = data.get("monthly_revenue_current") or data.get("current_revenue") or 0
    burn_rate = data.get("burn_rate_monthly_est") or data.get("burn_rate_monthly") or 0
    cash_remaining = data.get("total_funding") or data.get("funding_raised") or data.get("cash_remaining") or 0
    team_size = data.get("team_size") or 2
    age_months = data.get("age_months") or 6
    category = data.get("main_category") or data.get("category") or "saas"
    current_stage = data.get("current_stage") or "launch"
    target_stage = data.get("target_stage") or "traction"
    
    # ✅ FIX: Ensure all values are numbers, not None
    current_revenue = float(current_revenue) if current_revenue else 0.0
    burn_rate = float(burn_rate) if burn_rate else 0.0
    cash_remaining = float(cash_remaining) if cash_remaining else 0.0
    
    # ✅ Calculate current runway (with safe math)
    net_burn = burn_rate - current_revenue  # Actual monthly cash loss
    
    if net_burn > 0 and cash_remaining > 0:
        runway_months = cash_remaining / net_burn
    elif net_burn <= 0:
        runway_months = 999  # Already profitable or break-even
    else:
        runway_months = 0  # No cash
    
    # Simulate 24 months forward with 3 scenarios
    scenarios = {}
    
    # === SCENARIO 1: OPTIMISTIC (30% monthly growth) ===
    optimistic = _simulate_scenario(
        current_revenue=current_revenue,
        burn_rate=burn_rate,
        cash_remaining=cash_remaining,
        growth_rate=0.30,
        burn_reduction=0.05,  # 5% burn efficiency improvement per month
        months=24,
        label="Optimistic"
    )
    scenarios["optimistic"] = optimistic
    
    # === SCENARIO 2: REALISTIC (15% monthly growth) ===
    realistic = _simulate_scenario(
        current_revenue=current_revenue,
        burn_rate=burn_rate,
        cash_remaining=cash_remaining,
        growth_rate=0.15,
        burn_reduction=0.02,  # 2% burn efficiency improvement per month
        months=24,
        label="Realistic"
    )
    scenarios["realistic"] = realistic
    
    # === SCENARIO 3: PESSIMISTIC (5% monthly growth) ===
    pessimistic = _simulate_scenario(
        current_revenue=current_revenue,
        burn_rate=burn_rate,
        cash_remaining=cash_remaining,
        growth_rate=0.05,
        burn_reduction=0.00,  # No burn improvement
        months=24,
        label="Pessimistic"
    )
    scenarios["pessimistic"] = pessimistic
    
    # Calculate key insights for LLM
    analysis = _analyze_scenarios(optimistic, realistic, pessimistic, current_revenue, burn_rate, cash_remaining)
    
    return {
        "tool": "journey_simulator",
        "current_state": {
            "revenue_monthly": current_revenue,
            "burn_monthly": burn_rate,
            "net_burn_monthly": net_burn,
            "cash_remaining": cash_remaining,
            "runway_months": round(runway_months, 1) if runway_months < 999 else "Infinite (profitable)",
            "is_profitable": current_revenue >= burn_rate,
            "team_size": team_size,
            "age_months": age_months,
            "category": category,
            "current_stage": current_stage,
            "target_stage": target_stage
        },
        "scenarios": scenarios,
        "analysis": analysis,
        "instruction_for_llm": "Use this simulation data to create a narrative story explaining the three possible paths. Reference specific months and numbers. Call out risks explicitly. End with actionable recommendations."
    }


def _simulate_scenario(
    current_revenue: float,
    burn_rate: float,
    cash_remaining: float,
    growth_rate: float,
    burn_reduction: float,
    months: int,
    label: str
) -> Dict:
    """Simulate a single scenario month-by-month"""
    
    monthly_data = []
    
    # Start with current values
    revenue = float(current_revenue)
    burn = float(burn_rate)
    cash = float(cash_remaining)
    
    for month in range(0, months + 1):
        
        # ✅ Calculate runway BEFORE any changes
        if burn > revenue and cash > 0:
            runway = cash / (burn - revenue)
        elif cash <= 0:
            runway = 0
        else:
            runway = 999  # Profitable
        
        # ✅ Calculate net cashflow
        net_cashflow = revenue - burn
        
        # ✅ Record CURRENT month's data BEFORE modifying
        monthly_data.append({
            "month": month,
            "revenue": round(revenue, 2),
            "burn": round(burn, 2),
            "net_cashflow": round(net_cashflow, 2),
            "cash": round(max(cash, 0), 2),  # Show current cash
            "runway_months": round(runway, 1) if runway < 999 else 999,
            "is_profitable": revenue >= burn
        })
        
        # ✅ Stop if cash runs out
        if cash <= 0:
            break
        
        # ✅ NOW update cash for NEXT month
        cash += net_cashflow
        
        # ✅ Grow revenue and reduce burn for NEXT month
        if month < months:
            revenue = revenue * (1 + growth_rate)
            burn = burn * (1 - burn_reduction)
    
    # Find breakeven month (first month where revenue >= burn)
    breakeven_month = None
    for m in monthly_data:
        if m["is_profitable"] and m["month"] > 0:
            breakeven_month = m["month"]
            break
    
    # Get key milestone months
    final_state = monthly_data[-1] if monthly_data else {}
    
    def get_month_data(month_num):
        return next((m for m in monthly_data if m["month"] == month_num), {})
    
    month_3 = get_month_data(3)
    month_6 = get_month_data(6)
    month_12 = get_month_data(12)
    month_24 = get_month_data(24)
    
    # Find when cash runs out
    cash_out_month = None
    for m in monthly_data:
        if m["cash"] <= 0:
            cash_out_month = m["month"]
            break
    
    return {
        "label": label,
        "growth_rate_monthly": f"{growth_rate*100:.0f}%",
        "burn_reduction_rate": f"{burn_reduction*100:.1f}%",
        "monthly_trajectory": monthly_data,
        "breakeven_month": breakeven_month,
        "cash_depletion_month": cash_out_month,
        "key_milestones": {
            "month_0": monthly_data[0] if len(monthly_data) > 0 else {},
            "month_3": month_3,
            "month_6": month_6,
            "month_12": month_12,
            "month_24": month_24,
            "final": final_state
        },
        "survives_24_months": len(monthly_data) >= 24 and final_state.get("cash", 0) > 0,
        "becomes_profitable": breakeven_month is not None,
        "total_months_simulated": len(monthly_data) - 1
    }


def _analyze_scenarios(optimistic: Dict, realistic: Dict, pessimistic: Dict, 
                       current_revenue: float, burn_rate: float, cash_remaining: float) -> Dict:
    """
    Analyze scenarios to provide insights for LLM storytelling
    """
    
    # Determine recommended scenario
    realistic_survives = realistic["survives_24_months"]
    realistic_breakeven = realistic["breakeven_month"]
    
    if realistic_survives and realistic_breakeven and realistic_breakeven <= 18:
        recommendation = "realistic"
        confidence = 0.75
        reason = "Realistic path leads to profitability within 18 months"
    elif optimistic["becomes_profitable"]:
        recommendation = "optimistic"
        confidence = 0.55
        reason = "Only optimistic scenario reaches profitability - aggressive execution required"
    else:
        recommendation = "fundraising_required"
        confidence = 0.90
        reason = "No scenario reaches profitability without additional funding"
    
    # Identify critical risks
    risks = []
    
    if pessimistic["cash_depletion_month"] and pessimistic["cash_depletion_month"] <= 6:
        risks.append({
            "severity": "critical",
            "message": f"In pessimistic case, cash runs out at Month {pessimistic['cash_depletion_month']}",
            "action": "Secure bridge funding or reduce burn immediately"
        })
    
    if realistic["cash_depletion_month"] and realistic["cash_depletion_month"] <= 12:
        risks.append({
            "severity": "high",
            "message": f"Realistic case shows cash depletion at Month {realistic['cash_depletion_month']}",
            "action": f"Must raise funds by Month {realistic['cash_depletion_month'] - 3} or achieve profitability"
        })
    
    if not realistic["becomes_profitable"]:
        risks.append({
            "severity": "medium",
            "message": "Realistic scenario does not achieve profitability in 24 months",
            "action": "Plan for Series A fundraising or major burn reduction"
        })
    
    # ✅ Calculate funding requirements (if any scenario needs it)
    funding_needed = {}
    
    for scenario_name, scenario in [("optimistic", optimistic), ("realistic", realistic), ("pessimistic", pessimistic)]:
        if scenario["cash_depletion_month"]:
            # Find minimum cash point
            min_cash = min([m["cash"] for m in scenario["monthly_trajectory"]])
            if min_cash < 0:
                funding_needed[scenario_name] = round(abs(min_cash) * 1.3, 2)  # 30% buffer
    
    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "reason": reason,
        "risks": risks,
        "funding_requirements": funding_needed if funding_needed else None,
        "probability_distribution": {
            "optimistic": 0.20,
            "realistic": 0.60,
            "pessimistic": 0.20
        },
        "critical_decision_points": _identify_decision_points(optimistic, realistic, pessimistic)
    }


def _identify_decision_points(optimistic: Dict, realistic: Dict, pessimistic: Dict) -> List[Dict]:
    """Identify months where critical decisions must be made"""
    
    decision_points = []
    
    # Check for funding decisions
    realistic_cash_out = realistic["cash_depletion_month"]
    if realistic_cash_out and realistic_cash_out <= 12:
        decision_points.append({
            "month": max(1, realistic_cash_out - 3),
            "type": "funding",
            "description": f"Begin fundraising process (3 months before cash depletion)",
            "trigger": f"If cash runway < 6 months"
        })
    
    # Check for growth validation
    realistic_month_3 = realistic["key_milestones"]["month_3"]
    if realistic_month_3 and realistic_month_3.get("revenue"):
        expected_revenue = realistic_month_3.get("revenue", 0)
        decision_points.append({
            "month": 3,
            "type": "validation",
            "description": f"Validate growth trajectory",
            "trigger": f"If revenue < ${expected_revenue * 0.8:.0f}, reassess growth assumptions"
        })
    
    # Check for breakeven milestone
    realistic_breakeven = realistic["breakeven_month"]
    if realistic_breakeven and realistic_breakeven <= 18:
        decision_points.append({
            "month": max(1, realistic_breakeven - 2),
            "type": "profitability",
            "description": "Prepare for profitability milestone",
            "trigger": f"If not on track by Month {realistic_breakeven - 3}, consider burn reduction or price increase"
        })
    
    return sorted(decision_points, key=lambda x: x["month"])


# Tool metadata for MCP server
TOOL_METADATA = {
    "name": "journey_simulator",
    "description": "Simulate 3 scenario trajectories (optimistic/realistic/pessimistic) for startup over 24 months",
    "input_schema": {
        "type": "object",
        "properties": {
            "monthly_revenue_current": {"type": "number"},
            "burn_rate_monthly_est": {"type": "number"},
            "cash_remaining": {"type": "number"},
            "team_size": {"type": "number"},
            "age_months": {"type": "number"},
            "main_category": {"type": "string"},
            "current_stage": {"type": "string"},
            "target_stage": {"type": "string"}
        },
        "required": ["monthly_revenue_current", "burn_rate_monthly_est", "cash_remaining"]
    }
}
