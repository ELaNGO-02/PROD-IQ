"""
Simple Delta Extractor - Uses descriptive field names instead of synonym dictionary
"""
from typing import Dict, Any
import json
import re


def build_extraction_prompt(state_manager, turn_number: int) -> str:
    """Prompt with explicit number extraction examples"""
    
    current_state = {k: v for k, v in state_manager.state.items() 
                     if v is not None and not k.startswith('session_') and not k.startswith('last_')}
    
    state_display = json.dumps(current_state, indent=2) if current_state else "{}"
    
    prompt = f"""Extract numbers as JSON with <delta> tags.

CRITICAL: Extract ALL numbers mentioned by user!

EXAMPLES:

User: "We're at $2K MRR with 40 users at $50/month. We have $50K left and burn is $8K/month. Target is $10K MRR in 6 months."
<delta>
{{
  "monthly_revenue_current": 2000,
  "user_count_total": 40,
  "price_per_unit": 50,
  "cash_remaining": 50000,
  "burn_rate_monthly": 8000,
  "monthly_revenue_target": 10000,
  "timeframe_months": 6
}}
</delta>

User: "If we cut burn to $5K/month, what happens?"
<delta>
{{
  "scenario_active": true,
  "scenario_burn": 5000
}}
</delta>

User: "Our growth is 15% month-over-month"
<delta>
{{
  "growth_rate_monthly_pct": 15.0
}}
</delta>

UNITS TO WATCH:
- "$2K" or "$2000" → 2000
- "40 users" → user_count_total
- "$50/month" → price_per_unit
- "$50K" or "$50000" → depends on context (funding or cash)
- "burn is $8K" → burn_rate_monthly: 8000
- "15%" → growth_rate_monthly_pct: 15.0
- "6 months" → timeframe_months: 6

CURRENT STATE:
{state_display}

EXTRACT FROM USER INPUT (output ONLY <delta> block):"""
    
    return prompt


def extract_numbers_with_regex(user_query: str) -> Dict[str, Any]:
    """Fallback: Extract numbers using regex"""
    
    extracted = {}
    
    # Pattern: $XK or $X,XXX
    money_patterns = [
        (r'\$(\d+)K\s*MRR', 'monthly_revenue_current', 1000),
        (r'(\d+)\s*paying\s*users', 'user_count_total', 1),
        (r'\$(\d+)/month', 'price_per_unit', 1),
        (r'\$(\d+)K\s*left', 'cash_remaining', 1000),
        (r'burn\s*is\s*\$(\d+)K', 'burn_rate_monthly', 1000),
        (r'target.*?\$(\d+)K', 'monthly_revenue_target', 1000),
        (r'in\s*(\d+)\s*months', 'timeframe_months', 1),
        (r'(\d+)%\s*month', 'growth_rate_monthly_pct', 1)
    ]
    
    for pattern, field, multiplier in money_patterns:
        match = re.search(pattern, user_query, re.IGNORECASE)
        if match:
            value = int(match.group(1)) * multiplier
            extracted[field] = value
    
    return extracted


def extract_delta_simple(user_query: str, state_manager, turn_number: int) -> Dict[str, Any]:
    """Extract delta - with robust parsing"""
    from llm.local_inference import generate
    
    system_prompt = build_extraction_prompt(state_manager, turn_number)
    
    try:
        response = generate(
            user_prompt=user_query,
            system_prompt=system_prompt,
            max_tokens=400,
            temperature=0.05
        )
        
        print(f"\n   📋 RAW LLM RESPONSE:")
        print(f"   {'-'*50}")
        print(f"   {response[:500]}")
        print(f"   {'-'*50}\n")
        
        # Try multiple tag formats
        delta_json = None
        
        # 1. Try <delta>
        match = re.search(r'<delta>(.*?)</delta>', response, re.DOTALL)
        if match:
            delta_json = match.group(1).strip()
            print(f"   ✅ Found <delta> tag")
        
        # 2. Try <extraction> (LLM might use this)
        if not delta_json:
            match = re.search(r'<extraction>(.*?)</extraction>', response, re.DOTALL)
            if match:
                delta_json = match.group(1).strip()
                print(f"   ⚠️  Found <extraction> tag (LLM used wrong tag)")
        
        # 3. Try raw JSON
        if not delta_json:
            match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if match:
                delta_json = match.group(0).strip()
                print(f"   ⚠️  Found raw JSON (no tags)")
        
        # Parse JSON
        if not delta_json or delta_json in ["{}", ""]:
            print(f"   ℹ️  No changes detected from LLM")
            delta = {}
        else:
            delta = json.loads(delta_json)
        
        # Field name mapping (in case LLM uses wrong names)
        # Field name mapping (in case LLM uses wrong names)
        field_map = {
            'current_mrr': 'monthly_revenue_current',
            'mrr': 'monthly_revenue_current',
            'revenue': 'monthly_revenue_current',
            'target_mrr': 'monthly_revenue_target',
            'future_mrr_target': 'monthly_revenue_target',
            'goal': 'monthly_revenue_target',
            'current_burn': 'burn_rate_monthly',
            'burn': 'burn_rate_monthly',
            'spending': 'burn_rate_monthly',
            'price': 'price_per_unit',
            'users': 'user_count_total',
            'current_users': 'user_count_total',
            'customers': 'user_count_total',
            'funding': 'funding_raised_total',
            'funding_raised': 'funding_raised_total',
            'raised': 'funding_raised_total',
            'cash': 'cash_remaining',
            'left': 'cash_remaining',
            'age': 'company_age_months',
            'months_old': 'company_age_months',
            'future_months': 'timeframe_months',
            'months': 'timeframe_months',
            'growth': 'growth_rate_monthly_pct',
            
            # ✅ ADD: New field mappings
            'orders': 'orders_per_day',
            'daily_orders': 'orders_per_day',
            'orders_day': 'orders_per_day',
            'monthly_orders': 'orders_per_month',
            'aov': 'average_order_value',
            'average_order': 'average_order_value',
            'order_value': 'average_order_value',
            'commission': 'commission_rate_pct',
            'commission_rate': 'commission_rate_pct',
            'total_funding': 'funding_raised_total',
            'current_cash': 'cash_remaining'
        }

        
        normalized = {}
        for key, value in delta.items():
            correct_key = field_map.get(key.lower(), key)
            normalized[correct_key] = value
            if key != correct_key:
                print(f"      🔄 Remapped: {key} → {correct_key}")
        
        # ✅ VALIDATION: Check if first turn extracted enough fields
        if turn_number == 1:
            query_lower = user_query.lower()
            
            validation_checks = [
                ('mrr' in query_lower or 'revenue' in query_lower, 'monthly_revenue_current'),
                ('burn' in query_lower or 'spending' in query_lower, 'burn_rate_monthly'),
                ('users' in query_lower or 'customers' in query_lower, 'user_count_total'),
                ('left' in query_lower or 'cash' in query_lower, 'cash_remaining'),
                ('target' in query_lower or 'goal' in query_lower, 'monthly_revenue_target')
            ]
            
            missing_fields = []
            for (condition, field) in validation_checks:
                if condition and field not in normalized:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"   ⚠️  WARNING: User mentioned these but LLM didn't extract: {missing_fields}")
                print(f"   🔧 Trying regex fallback...")
                
                # Use regex fallback
                regex_extracted = extract_numbers_with_regex(user_query)
                for key, value in regex_extracted.items():
                    if key not in normalized:
                        normalized[key] = value
                        print(f"      ✅ Regex found: {key} = {value}")
        
        # ✅ SCENARIO DETECTION: Fallback if LLM missed it
        query_lower = user_query.lower()
        if any(kw in query_lower for kw in ['if we', 'if i', 'what if', 'suppose', "let's say", 'hypothetically']):
            if not normalized.get('scenario_active'):
                print(f"   🎭 Scenario keyword detected - flagging scenario")
                normalized['scenario_active'] = True
            
            # Move actual fields to scenario fields
            for actual, scenario in [
                ('burn_rate_monthly', 'scenario_burn'),
                ('monthly_revenue_current', 'scenario_revenue'),
                ('price_per_unit', 'scenario_price'),
                ('business_model', 'scenario_business_model')
            ]:
                if actual in normalized and scenario not in normalized:
                    normalized[scenario] = normalized.pop(actual)
                    print(f"      → Moved {actual} to {scenario}")
        
        # ✅ HALLUCINATION DETECTION: Remove fields user didn't mention in scenarios
        if normalized.get('scenario_active'):
            query_lower = user_query.lower()
            
            # Check each scenario field
            hallucination_checks = [
                ('burn' not in query_lower and 'spending' not in query_lower, 'scenario_burn'),
                ('revenue' not in query_lower and 'mrr' not in query_lower, 'scenario_revenue'),
                ('price' not in query_lower and 'pricing' not in query_lower, 'scenario_price')
            ]
            
            for (condition, field) in hallucination_checks:
                if condition and field in normalized:
                    print(f"   🚨 HALLUCINATION: Removing {field} (user didn't mention it)")
                    del normalized[field]
        
        print(f"   ✅ Final delta: {list(normalized.keys())}")

        
        return normalized
        
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON error: {e}")
        print(f"   🔧 Falling back to regex extraction...")
        return extract_numbers_with_regex(user_query)
    
    except Exception as e:
        print(f"   ❌ Extraction error: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
def extract_numbers_with_regex(user_query: str) -> Dict[str, Any]:
    """Fallback: Extract numbers using regex"""
    
    extracted = {}
    
    # ✅ ADD: Indian currency patterns (₹ and lakhs)
    indian_patterns = [
        (r'₹(\d+)\s*lakh[s]?', 'funding_raised_total', 100000),  # "₹25 lakhs"
        (r'₹(\d+)L', 'funding_raised_total', 100000),  # "₹25L"
        (r'₹(\d+)\s*lakh[s]?\s*funding', 'funding_raised_total', 100000),
        (r'burn.*?₹(\d+)\s*lakh[s]?', 'burn_rate_monthly', 100000),  # "burn is ₹4 lakhs"
        (r'₹(\d+)\s*average', 'average_order_value', 1),  # "₹300 average"
        (r'(\d+)\s*orders?/day', 'orders_per_day', 1),  # "200 orders/day"
        (r'(\d+)\s*daily\s*orders?', 'orders_per_day', 1),  # "200 daily orders"
        (r'(\d+)%\s*commission', 'commission_rate_pct', 1),  # "20% commission"
    ]
    
    # Original patterns (keep these)
    money_patterns = [
        (r'\$(\d+)K\s*MRR', 'monthly_revenue_current', 1000),
        (r'(\d+)\s*paying\s*users', 'user_count_total', 1),
        (r'\$(\d+)/month', 'price_per_unit', 1),
        (r'\$(\d+)K\s*left', 'cash_remaining', 1000),
        (r'burn\s*is\s*\$(\d+)K', 'burn_rate_monthly', 1000),
        (r'target.*?\$(\d+)K', 'monthly_revenue_target', 1000),
        (r'in\s*(\d+)\s*months', 'timeframe_months', 1),
        (r'(\d+)%\s*month', 'growth_rate_monthly_pct', 1)
    ]
    
    # Combine all patterns
    all_patterns = indian_patterns + money_patterns
    
    for pattern, field, multiplier in all_patterns:
        match = re.search(pattern, user_query, re.IGNORECASE)
        if match:
            value = int(match.group(1)) * multiplier
            extracted[field] = value
    
    return extracted

