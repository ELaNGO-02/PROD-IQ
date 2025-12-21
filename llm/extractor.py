#!/usr/bin/env python3
"""
Data Extractor - Parse LLM structured outputs
Handles <extraction>, <thinking>, <tools> tags
"""
import json
import re
from typing import Dict, Any


class DataExtractor:
    """Parse extractions from structured LLM responses"""
    
    def parse_extraction(self, response: str) -> Dict[str, Any]:
        """
        Extract structured data from LLM response
        LLM returns: <extraction>{json}</extraction>
        """
        
        # METHOD 1: Extract from <extraction> block (primary method)
        extraction_match = re.search(r'<extraction>(.*?)</extraction>', response, re.DOTALL)
        if extraction_match:
            try:
                json_str = extraction_match.group(1).strip()
                extracted = json.loads(json_str)
                print(f"   ✅ Extracted from <extraction> tag")
                return self._normalize_extraction(extracted)
            except json.JSONDecodeError as e:
                print(f"   ⚠️  JSON parse error: {e}")
        
        # METHOD 2: Look for any JSON block (fallback)
        json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        for json_str in json_matches:
            try:
                extracted = json.loads(json_str)
                if self._looks_like_extraction(extracted):
                    print(f"   ✅ Extracted from JSON block")
                    return self._normalize_extraction(extracted)
            except json.JSONDecodeError:
                continue
        
        # METHOD 3: Use defaults (should rarely happen with new LLM)
        print("   ⚠️  No extraction found, using defaults")
        return self._get_defaults()
    
    def parse_thinking(self, response: str) -> str:
        """Extract thinking process from <thinking> tag"""
        match = re.search(r'<thinking>(.*?)</thinking>', response, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def parse_tools(self, response: str) -> list:
        """Extract tool calls from <tools> tag"""
        tools_match = re.search(r'<tools>(.*?)</tools>', response, re.DOTALL)
        if tools_match:
            tools_text = tools_match.group(1)
            # Extract individual tools
            tool_list = re.findall(r'<tool>(.*?)</tool>', tools_text)
            return tool_list
        return []
    
    def parse_math(self, response: str) -> str:
        """Extract math calculations from <math_calculation> tag"""
        match = re.search(r'<math_calculation>(.*?)</math_calculation>', response, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _looks_like_extraction(self, obj: Dict) -> bool:
        """Check if dict looks like valid extraction"""
        expected_keys = ['current_mrr', 'target_mrr', 'current_revenue', 'category', 'product_name']
        return any(key in obj for key in expected_keys)
    
    def _normalize_extraction(self, data: Dict) -> Dict:
        """Normalize extracted data to standard format"""
        
        # Map LLM's keys to our standard keys
        normalized = {}
        
        # Revenue mapping
        normalized['current_revenue'] = (
            data.get('current_mrr') or 
            data.get('current_revenue') or 
            data.get('mrr') or 0
        )
        
        normalized['target_revenue'] = (
            data.get('target_mrr') or 
            data.get('target_revenue') or 
            data.get('goal_mrr') or None
        )
        
        # User/customer mapping
        normalized['user_count'] = (
            data.get('current_users') or 
            data.get('user_count') or 
            data.get('customers') or 0
        )
        
        # Pricing
        normalized['price_point'] = (
            data.get('price') or 
            data.get('price_point') or 
            data.get('avg_price') or None
        )
        
        # Financial
        normalized['funding_raised'] = (
            data.get('total_funding') or 
            data.get('funding_raised') or 
            data.get('cash') or 0
        )
        
        normalized['burn_rate_monthly_est'] = (
        data.get('current_burn') or 
        data.get('burn_rate') or 
        data.get('monthly_burn') or 
        data.get('burn_rate_monthly_est') or
        None  # Let tool executor calculate if not provided
        )
        
        # Timeline
        normalized['timeframe_months'] = (
            data.get('growth_months') or 
            data.get('timeframe_months') or 
            data.get('months') or 6
        )
        
        # Metadata
        normalized['product_name'] = data.get('product_name') or data.get('name')
        normalized['category'] = data.get('category') or 'saas'
        normalized['stage'] = data.get('stage') or 'launched'
        normalized['team_size'] = data.get('team_size') or 2
        normalized['age_months'] = data.get('age_months') or 6
        
        # Core question detection
        normalized['core_question'] = self._detect_core_question(data)
        
        return normalized
    
    def _detect_core_question(self, data: Dict) -> str:
        """Detect core question from extraction or context"""
        if 'core_question' in data:
            return data['core_question']
        
        # Infer from presence of certain fields
        if data.get('target_mrr') or data.get('target_revenue'):
            return 'revenue_validation'
        elif data.get('current_burn') or data.get('runway'):
            return 'survival_analysis'
        else:
            return 'general_advice'
    
    def _get_defaults(self) -> Dict:
        """Return default extraction structure"""
        return {
            'product_name': None,
            'category': 'saas',
            'stage': 'launched',
            'current_revenue': 0,
            'target_revenue': None,
            'user_count': 0,
            'price_point': None,
            'team_size': 2,
            'funding_raised': 0,
            'burn_rate_monthly_est': None,
            'timeframe_months': 6,
            'age_months': 6,
            'core_question': 'general_advice'
        }
