"""
Simple State Manager - Dictionary-based state with implicit schema
No explicit validation rules - just key existence and type inference
"""
from typing import Dict, Any, Optional
from datetime import datetime
import json


class SimpleStateManager:
    def __init__(self):
        self.state = {
            # Core metrics
            "product_name": None,
            "product_description": None,
            "monthly_revenue_current": None,
            "monthly_revenue_target": None,
            "price_per_unit": None,
            "burn_rate_monthly": None,
            "funding_raised_total": None,
            "cash_remaining": None,
            "user_count_total": None,
            "customer_count_paying": None,
            "team_size": 2,
            "company_age_months": None,
            "runway_months": None,
            "breakeven_months": None,
            
            # ✅ ADD: Order-based metrics (for cloud kitchens, marketplaces)
            "orders_per_day": None,
            "monthly_orders_current": None,
            "target_orders_monthly": None,
            "average_order_value": None,
            "commission_rate_pct": None,
            
            # Product details
            "category": "saas",
            "launch_stage": "idea",
            "business_model": "B2C",
            "target_audience": None,
            "pricing_model": "subscription",
            
            # Growth metrics
            "growth_rate_monthly_pct": 20.0,
            "timeframe_months": 6,
            "target_milestone": None,
            
            # Scenario tracking
            "scenario_active": False,
            "scenario_description": None,
            "scenario_price": None,
            "scenario_revenue": None,
            "scenario_burn": None,
            "scenario_target_audience": None,
            "scenario_business_model": None,
            
            # Question context
            "question_type": "initial",
            "core_question_intent": None,
            
            # Metadata
            "last_updated_turn": 0,
            "session_start_time": datetime.now().isoformat()
        }
        
        self.history = []
        self.turn_count = 0
    
    def update(self, delta: Dict[str, Any], turn_number: int) -> Dict[str, Any]:
        """
        Merge delta into state with soft validation
        
        Args:
            delta: Changes from LLM extraction
            turn_number: Current turn number
            
        Returns:
            Updated state
        """
        changes = []
        ignored = []
        
        for key, value in delta.items():
            # ✅ SAFETY: Only allow keys that exist
            if key not in self.state:
                ignored.append(key)
                continue
            
            # ✅ SOFT TYPE CASTING
            # Try to match type of existing value
            current_value = self.state[key]
            if current_value is not None and value is not None:
                try:
                    if isinstance(current_value, int) and not isinstance(current_value, bool):
                        value = int(float(value))  # Handle "50.0" -> 50
                    elif isinstance(current_value, float):
                        value = float(value)
                    elif isinstance(current_value, bool):
                        value = bool(value)
                    elif isinstance(current_value, str):
                        value = str(value)
                except (ValueError, TypeError):
                    # Type cast failed - keep raw value
                    pass
            
            # ✅ UPDATE if different
            if self.state[key] != value:
                old_value = self.state[key]
                self.state[key] = value
                changes.append({
                    'field': key,
                    'old': old_value,
                    'new': value
                })
        
        # Update metadata
        self.state['last_updated_turn'] = turn_number
        self.turn_count = turn_number
        
        # Store in history
        if changes:
            self.history.append({
                'turn': turn_number,
                'timestamp': datetime.now().isoformat(),
                'changes': changes,
                'delta_received': delta
            })
            
            print(f"   ✅ Updated {len(changes)} fields:")
            for change in changes:
                print(f"      {change['field']}: {change['old']} → {change['new']}")
        else:
            print(f"   ℹ️  No state changes")
        
        if ignored:
            print(f"   ⚠️  Ignored unknown fields: {ignored}")
        
        return self.state.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get single value"""
        return self.state.get(key, default)
    
    def get_state(self) -> Dict[str, Any]:
        """Get full state copy"""
        import copy
        return copy.deepcopy(self.state)
    
    def reset(self):
        """Clear session"""
        print(f"🔄 Session reset after {self.turn_count} turns")
        
        # Reset all values to initial state
        for key in self.state:
            if key == "team_size":
                self.state[key] = 2
            elif key == "category":
                self.state[key] = "saas"
            elif key == "launch_stage":
                self.state[key] = "idea"
            elif key == "business_model":
                self.state[key] = "B2C"
            elif key == "pricing_model":
                self.state[key] = "subscription"
            elif key == "growth_rate_monthly_pct":
                self.state[key] = 20.0
            elif key == "timeframe_months":
                self.state[key] = 6
            elif key == "question_type":
                self.state[key] = "initial"
            elif key == "scenario_active":
                self.state[key] = False
            else:
                self.state[key] = None
        
        self.state['session_start_time'] = datetime.now().isoformat()
        self.state['last_updated_turn'] = 0
        
        self.history = []
        self.turn_count = 0
    
    def get_summary(self, last_n_turns: int = 3) -> str:
        """Summary of recent changes"""
        recent = self.history[-last_n_turns:] if len(self.history) >= last_n_turns else self.history
        
        summary = f"Recent state changes (Turn {self.turn_count}):\n"
        for entry in recent:
            if entry['changes']:
                fields = [c['field'] for c in entry['changes']]
                summary += f"  Turn {entry['turn']}: {', '.join(fields)}\n"
        
        return summary
    
    def export_debug(self) -> str:
        """Export for debugging"""
        # Only show non-None values
        active_state = {k: v for k, v in self.state.items() if v is not None}
        
        return json.dumps({
            'active_state': active_state,
            'turn_count': self.turn_count,
            'history_entries': len(self.history)
        }, indent=2)
    
    def clear_scenario(self):
        """Clear scenario fields after scenario calculation"""
        scenario_fields = [
            'scenario_active', 'scenario_description', 'scenario_price',
            'scenario_revenue', 'scenario_burn', 'scenario_target_audience',
            'scenario_business_model'
        ]
        
        for field in scenario_fields:
            if field == 'scenario_active':
                self.state[field] = False
            else:
                self.state[field] = None
        
        print("   🧹 Scenario cleared")

