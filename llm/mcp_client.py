#!/usr/bin/env python3
"""
MCP Client - Connects orchestrator to MCP server tools
"""
import subprocess
import json
from typing import Dict, Any, Optional


class MCPClient:
    """Client for MCP server communication"""
    
    def __init__(self, server_path: str = "mcp_server/server.py"):
        self.server_path = server_path
        self.process = None
        self._start_server()
    
    def _start_server(self):
        """Start MCP server as subprocess"""
        try:
            self.process = subprocess.Popen(
                ["python", self.server_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("✅ MCP Server connected")
        except Exception as e:
            print(f"⚠️ MCP Server connection failed: {e}")
            self.process = None
    
    def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Optional[Dict]:
        """
        Call MCP tool and get response
        
        Args:
            tool_name: Name of MCP tool (e.g., 'predict_revenue')
            parameters: Tool parameters
        
        Returns:
            Tool response as dict
        """
        if not self.process:
            return None
        
        try:
            # Format MCP request
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": parameters
                }
            }
            
            # Send request
            request_json = json.dumps(request) + "\n"
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
            
            # Read response
            response_line = self.process.stdout.readline()
            response = json.loads(response_line)
            
            if "result" in response:
                return response["result"]
            elif "error" in response:
                print(f"⚠️ MCP error: {response['error']}")
                return None
            
        except Exception as e:
            print(f"⚠️ MCP call failed: {e}")
            return None
    
    def __del__(self):
        """Clean up server process"""
        if self.process:
            self.process.terminate()
            self.process.wait()


# Convenience wrapper
class MCPToolExecutor:
    """Wraps MCP client with tool-specific methods"""
    
    def __init__(self):
        self.client = MCPClient()
    
    def predict_revenue(self, data: Dict) -> Dict:
        """Call MCP predict_revenue"""
        return self.client.call_tool("predict_revenue", {
            "company_name": data.get("product_name", "Unknown"),
            "category": data.get("category", "saas"),
            "funding_raised": data.get("funding_raised", 0),
            "team_size": data.get("team_size", 2),
            "founding_year": 2025 - (data.get("age_months", 6) // 12),
            "monthly_active_users": data.get("user_count", 0),
            "description": data.get("description", "")
        }) or {"revenue": 0, "confidence": 0.5}
    
    def predict_success(self, data: Dict) -> Dict:
        """Call MCP predict_success"""
        return self.client.call_tool("predict_success", {
            "company_name": data.get("product_name", "Unknown"),
            "category": data.get("category", "saas"),
            "funding_raised": data.get("funding_raised", 0),
            "team_size": data.get("team_size", 2),
            "founding_year": 2025 - (data.get("age_months", 6) // 12),
            "market_size": "Medium",
            "competition_level": "Medium",
            "description": data.get("description", "")
        }) or {"probability": 0.5, "category": "Moderate Potential"}
    
    def predict_breakeven(self, data: Dict) -> Dict:
        """Call MCP predict_breakeven"""
        return self.client.call_tool("predict_breakeven", {
            "company_name": data.get("product_name", "Unknown"),
            "category": data.get("category", "saas"),
            "funding_raised": data.get("funding_raised", 0),
            "monthly_burn_rate": data.get("burn_rate_monthly_est", 0),
            "monthly_revenue": data.get("current_revenue", 0),
            "team_size": data.get("team_size", 2),
            "description": data.get("description", "")
        }) or {"months": 12, "confidence": 0.5}
    
    def predict_survival(self, data: Dict) -> Dict:
        """Call MCP predict_survival"""
        runway = data.get("funding_raised", 0) / max(data.get("burn_rate_monthly_est", 1), 1)
        return self.client.call_tool("predict_survival", {
            "company_name": data.get("product_name", "Unknown"),
            "category": data.get("category", "saas"),
            "funding_raised": data.get("funding_raised", 0),
            "months_operating": data.get("age_months", 6),
            "runway_months": int(runway),
            "team_size": data.get("team_size", 2),
            "has_revenue": data.get("current_revenue", 0) > 0,
            "description": data.get("description", "")
        }) or {"prob_6m": 0.8, "prob_12m": 0.7}
    
    def predict_traction(self, data: Dict) -> Dict:
        """Call MCP predict_traction"""
        return self.client.call_tool("predict_traction", {
            "company_name": data.get("product_name", "Unknown"),
            "category": data.get("category", "saas"),
            "founding_year": 2025 - (data.get("age_months", 6) // 12),
            "funding_raised": data.get("funding_raised", 0),
            "team_size": data.get("team_size", 2),
            "user_growth_rate": None,
            "revenue_growth_rate": None,
            "description": data.get("description", "")
        }) or {"months": 18, "confidence": 0.6}
