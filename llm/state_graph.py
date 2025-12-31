"""
State Graph - Single source of truth for conversation state
Manages delta updates and validation
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import json


class StateGraph:
    """
    Maintains conversation state as a graph of connected data points
    Accepts delta updates from LLM, validates, and merges
    """
    
    # ✅ SCHEMA: Define all possible fields (no hardcoding in prompts!)
    SCHEMA = {
        # Core metrics
        'current_revenue': {'type': float, 'min': 0, 'max': 1e9, 'default': None},
        'target_revenue': {'type': float, 'min': 0, 'max': 1e9, 'default': None},
        'price_point': {'type': float, 'min': 0, 'max': 1e6, 'default': None},
        'burn_rate_monthly': {'type': float, 'min': 0, 'max': 1e9, 'default': None},
        'funding_raised': {'type': float, 'min': 0, 'max': 1e12, 'default': None},
        'user_count': {'type': int, 'min': 0, 'max': 1e12, 'default': None},
        'team_size': {'type': int, 'min': 1, 'max': 10000, 'default': 2},
        'age_months': {'type': int, 'min': 0, 'max': 1200, 'default': None},
        
        # Product info
        'product_name': {'type': str, 'default': None},
        'category': {'type': str, 'allowed': ['saas', 'mobile', 'ecommerce', 'hardware', 'other'], 'default': 'saas'},
        'stage': {'type': str, 'allowed': ['idea', 'building', 'launched', 'scaling'], 'default': 'idea'},
        
        # Business model
        'model': {'type': str, 'allowed': ['B2C', 'B2B', 'B2B2C', 'marketplace'], 'default': 'B2C'},
        'target_audience': {'type': str, 'default': None},
        'pricing_model': {'type': str, 'allowed': ['subscription', 'one-time', 'freemium', 'usage-based'], 'default': 'subscription'},
        
        # Scenario tracking (for what-if questions)
        'scenario_active': {'type': bool, 'default': False},
        'scenario_type': {'type': str, 'default': None},
        'scenario_price': {'type': float, 'min': 0, 'max': 1e6, 'default': None},
        'scenario_revenue': {'type': float, 'min': 0, 'max': 1e9, 'default': None},
        'scenario_burn': {'type': float, 'min': 0, 'max': 1e9, 'default': None},
        'scenario_target': {'type': str, 'default': None},
        
        # Timeframes
        'timeframe_months': {'type': int, 'min': 1, 'max': 120, 'default': 6},
        'growth_rate_monthly': {'type': float, 'min': -0.5, 'max': 5.0, 'default': 0.20},
        
        # Question intent
        'question_type': {'type': str, 'allowed': ['initial', 'follow_up', 'scenario', 'benchmark', 'clarification'], 'default': 'initial'},
        'core_question': {'type': str, 'default': None}
    }
    
    # ✅ FIELD SYNONYMS: Teach LLM what maps to what field
    FIELD_SYNONYMS = {
        'current_revenue': ['MRR', 'monthly revenue', 'current revenue', 'revenue', 'making', 'earning'],
        'target_revenue': ['goal', 'target', 'aiming for', 'want to reach', 'target MRR'],
        'price_point': ['price', 'pricing', 'charge', 'cost', 'subscription price', 'per month'],
        'burn_rate_monthly': ['burn', 'burn rate', 'spending', 'costs', 'expenses', 'monthly burn'],
        'funding_raised': ['funding', 'raised', 'capital', 'investment', 'money raised'],
        'user_count': ['users', 'customers', 'subscribers', 'user base', 'downloads'],
        'team_size': ['team', 'people', 'employees', 'founders', 'team members'],
        'age_months': ['age', 'old', 'months old', 'launched', 'started'],
        'model': ['business model', 'B2B', 'B2C', 'marketplace', 'targeting'],
        'stage': ['stage', 'phase', 'launched', 'building', 'idea'],
        'scenario_price': ['if price', 'charge instead', 'at $ instead', 'pivot to $ price'],
        'scenario_target': ['pivot to', 'target instead', 'focus on', 'switch to']
    }
    
    def __init__(self):
        self.state: Dict[str, Any] = {}
        self.history: List[Dict] = []
        self.session_start = datetime.now()
        self.turn_count = 0
        
    def apply_delta(self, delta: Dict[str, Any], turn_number: int) -> Dict[str, Any]:
        """
        Apply delta update to state with validation
        
        Args:
            delta: Changes extracted by LLM
            turn_number: Current turn number
            
        Returns:
            Updated state
        """
        validated_delta = {}
        errors = []
        
        for field, value in delta.items():
            # Skip if not in schema
            if field not in self.SCHEMA:
                errors.append(f"Unknown field: {field}")
                continue
            
            # Validate value
            validation_result = self._validate_field(field, value)
            if validation_result['valid']:
                validated_delta[field] = validation_result['value']
            else:
                errors.append(f"{field}: {validation_result['error']}")
        
        # Merge validated delta into state
        old_state = self.state.copy()
        self.state.update(validated_delta)
        
        # Store in history
        self.history.append({
            'turn': turn_number,
            'timestamp': datetime.now().isoformat(),
            'delta': validated_delta,
            'errors': errors,
            'state_snapshot': self.state.copy()
        })
        
        self.turn_count = turn_number
        
        if errors:
            print(f"   ⚠️  Validation errors: {errors}")
        
        print(f"   ✅ State updated: {len(validated_delta)} fields changed")
        if validated_delta:
            for field, value in validated_delta.items():
                old_val = old_state.get(field, 'None')
                print(f"      {field}: {old_val} → {value}")
        
        return self.state
    
    def _validate_field(self, field: str, value: Any) -> Dict:
        """Validate a single field value against schema"""
        schema = self.SCHEMA[field]
        
        # Type checking
        expected_type = schema['type']
        
        try:
            # Type conversion
            if expected_type == float:
                value = float(value)
            elif expected_type == int:
                value = int(value)
            elif expected_type == bool:
                value = bool(value)
            elif expected_type == str:
                value = str(value)
            
            # Range checking
            if 'min' in schema and value < schema['min']:
                return {'valid': False, 'error': f'Value {value} below minimum {schema["min"]}'}
            if 'max' in schema and value > schema['max']:
                return {'valid': False, 'error': f'Value {value} above maximum {schema["max"]}'}
            
            # Allowed values checking
            if 'allowed' in schema and value not in schema['allowed']:
                return {'valid': False, 'error': f'Value {value} not in allowed: {schema["allowed"]}'}
            
            return {'valid': True, 'value': value}
            
        except (ValueError, TypeError) as e:
            return {'valid': False, 'error': f'Type conversion failed: {e}'}
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return self.state.copy()
    
    def get_field(self, field: str, default: Any = None) -> Any:
        """Get single field value"""
        return self.state.get(field, default)
    
    def reset(self):
        """Clear state (session end)"""
        print(f"🔄 Session reset after {self.turn_count} turns")
        self.state = {}
        self.history = []
        self.session_start = datetime.now()
        self.turn_count = 0
    
    def get_summary(self, last_n_turns: int = 3) -> str:
        """Get summary of recent state changes"""
        recent = self.history[-last_n_turns:] if len(self.history) >= last_n_turns else self.history
        
        summary = f"State summary (Turn {self.turn_count}):\n"
        for entry in recent:
            if entry['delta']:
                summary += f"  Turn {entry['turn']}: Updated {list(entry['delta'].keys())}\n"
        
        return summary
    
    def export_state(self) -> str:
        """Export state as JSON for debugging"""
        return json.dumps({
            'state': self.state,
            'turn_count': self.turn_count,
            'session_duration_mins': (datetime.now() - self.session_start).seconds // 60
        }, indent=2)
