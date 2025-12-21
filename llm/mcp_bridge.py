#!/usr/bin/env python3
"""
MCP Bridge - Connects Orchestrator to MCP Server
Allows LLM to call MCP tools for predictions
"""
import json
from typing import Dict, Any, List
from pathlib import Path
import sys

# Add mcp_server to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mcp_server"))

# Import MCP tools
from mcp_server.tools.predict_success import predict_success_handler
from mcp_server.tools.predict_traction import predict_traction_handler
from mcp_server.tools.predict_revenue import predict_revenue_handler
from mcp_server.tools.predict_breakeven import predict_breakeven_handler
from mcp_server.tools.predict_survival import predict_survival_handler
from mcp_server.tools.market_scout import market_scout_handler  # ⬅️ This is competitor analysis
from mcp_server.tools.journey_simulator import simulate_journey  # ⬅️ Direct import


class MCPBridge:
    """Bridge between Orchestrator and MCP Server tools"""
    
    def __init__(self):
        """Initialize MCP tool mappings"""
        self.tool_map = {
            'predict_success_label': self._call_predict_success,
            'predict_traction_time': self._call_predict_traction,
            'predict_revenue_estimate': self._call_predict_revenue,
            'predict_break_even_time': self._call_predict_breakeven,
            'predict_survival_months': self._call_predict_survival,
            'find_competitors': self._call_find_competitors,
            'journey_simulator': self._call_journey_simulator
        }
        print("   🔌 MCP Bridge initialized with 7 tools")
    
    def call_tool(self, tool_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call MCP tool by name
        
        Args:
            tool_name: Name of tool to call
            data: Extracted data from user query
        
        Returns:
            Tool output as dict
        """
        if tool_name in self.tool_map:
            try:
                return self.tool_map[tool_name](data)
            except Exception as e:
                return {'error': f"MCP tool error: {str(e)}"}
        else:
            return {'error': f"Unknown MCP tool: {tool_name}"}
    
    # ================================================================
    # PREDICTION TOOL WRAPPERS
    # ================================================================
    
    def _call_predict_success(self, data: Dict) -> Dict:
        """Call MCP predict_success"""
        import asyncio
        mcp_input = self._format_mcp_input(data)
        result = asyncio.run(predict_success_handler(mcp_input))
        return result
    
    def _call_predict_traction(self, data: Dict) -> Dict:
        """Call MCP predict_traction"""
        import asyncio
        mcp_input = self._format_mcp_input(data)
        result = asyncio.run(predict_traction_handler(mcp_input))
        return result
    
    def _call_predict_revenue(self, data: Dict) -> Dict:
        """Call MCP predict_revenue"""
        import asyncio
        mcp_input = self._format_mcp_input(data)
        result = asyncio.run(predict_revenue_handler(mcp_input))
        return result
    
    def _call_predict_breakeven(self, data: Dict) -> Dict:
        """Call MCP predict_breakeven"""
        import asyncio
        mcp_input = self._format_mcp_input(data)
        result = asyncio.run(predict_breakeven_handler(mcp_input))
        return result
    
    def _call_predict_survival(self, data: Dict) -> Dict:
        """Call MCP predict_survival"""
        import asyncio
        mcp_input = self._format_mcp_input(data)
        result = asyncio.run(predict_survival_handler(mcp_input))
        return result
    
    def _call_find_competitors(self, data: Dict) -> Dict:
        """Call MCP competitor analysis (market_scout)"""
        import asyncio
        mcp_input = {
            'company_name': data.get('product_name', 'Unknown'),
            'category': data.get('category', 'saas'),
            'target_market': data.get('geography', 'global'),
            'description': data.get('product_description', f"A {data.get('category', 'saas')} product")
        }
        result = asyncio.run(market_scout_handler(mcp_input))
        return result
    
    def _call_journey_simulator(self, data: Dict) -> Dict:
        """Call MCP journey_simulator"""
        # Direct call (not async)
        return simulate_journey(data)
    
    # ================================================================
    # INPUT FORMATTING
    # ================================================================
    
    def _format_mcp_input(self, data: Dict) -> Dict:
        """Convert extracted data to MCP input format"""
        return {
            'company_name': data.get('product_name', 'Unknown Product'),  # ✅ FIX
            'product_name': data.get('product_name', 'Unknown Product'),
            'product_description': data.get('product_description', ''),
            'main_category': data.get('category', 'saas'),
            'sub_category': data.get('sub_category', 'productivity'),
            'target_audience': data.get('target_audience', 'small business'),
            'founding_year': 2025 - (data.get('age_months', 3) // 12),
            'team_size': data.get('team_size', 2),
            'has_technical_founder': True,
            'funding_stage': data.get('funding_stage', 'bootstrapped'),
            'total_funding': data.get('funding_raised', 0),
            'funding_raised': data.get('funding_raised', 0),
            'total_funding': data.get('funding_raised', 0),
            'monthly_revenue': data.get('current_revenue', 0),
            'revenue_model': 'subscription',
            'pricing_model': 'per_user',
            'geography': data.get('geography', 'global'),
            'has_revenue': data.get('current_revenue', 0) > 0,
            'monthly_burn_rate': data.get('burn_rate_monthly_est', data.get('team_size', 2) * 5000),
            'has_product': True,
            'launch_stage': data.get('stage', 'launched'),
            'user_count': data.get('user_count', 0),
            'paying_customers': data.get('user_count', 0) if data.get('current_revenue', 0) > 0 else 0
        }


