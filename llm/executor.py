#!/usr/bin/env python3
"""
Tool Executor - Wraps ML models and database calls with MCP integration
Provides clean interface for orchestrator
"""
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Import MCP bridge
try:
    from llm.mcp_bridge import MCPBridge
    MCP_AVAILABLE = True
except Exception as e:
    print(f"   ⚠️ MCP bridge unavailable: {e}")
    MCP_AVAILABLE = False


class ToolExecutor:
    """Executes tools and returns formatted outputs"""
    
    def __init__(self, ml_inference, hybrid_search):
        """
        Args:
            ml_inference: InferenceEngine instance
            hybrid_search: HybridSearch instance
        """
        self.ml = ml_inference
        self.db = hybrid_search
        
        # Initialize MCP bridge (optional)
        self.mcp = None
        if MCP_AVAILABLE:
            try:
                self.mcp = MCPBridge()
                print("   ✅ MCP bridge connected")
            except Exception as e:
                print(f"   ⚠️ MCP bridge failed, using local tools: {e}")
        
        print("   ✅ ToolExecutor initialized")
    
    # ================================================================
    # PREDICTION TOOLS (Try MCP first, fallback to local)
    # ================================================================
    
    def predict_success(self, data: Dict) -> Dict[str, Any]:
        """Call success prediction model"""
        # Try MCP first
        if self.mcp:
            try:
                mcp_result = self.mcp.call_tool('predict_success_label', data)
                if mcp_result and 'error' not in mcp_result:
                    return self._format_success_output(mcp_result, source='mcp')
            except Exception as e:
                print(f"⚠️ MCP predict_success failed: {e}")
        
        # Fallback to local ML
        try:
            input_data = {
                'funding_raised': data.get('funding_raised', 0),
                'founding_year': 2025 - data.get('age_months', 3) // 12,
                'has_revenue': data.get('current_revenue', 0) > 0,
                'team_size': data.get('team_size', 2),
                'monthly_revenue': data.get('current_revenue', 0)
            }
            
            result = self.ml.predict_success(input_data)
            
            if 'error' in result:
                return result
            
            return self._format_success_output(result, source='local')
            
        except Exception as e:
            print(f"[ERROR] predict_success: {e}", file=sys.stderr)
            return {'error': str(e)}
    
    def predict_traction(self, data: Dict) -> Dict[str, Any]:
        """Call traction time prediction model"""
        # Try MCP first
        if self.mcp:
            try:
                mcp_result = self.mcp.call_tool('predict_traction_time', data)
                if mcp_result and 'error' not in mcp_result:
                    return self._format_traction_output(mcp_result, data, source='mcp')
            except Exception as e:
                print(f"⚠️ MCP predict_traction failed: {e}")
        
        # Fallback to local ML
        try:
            age_months = int(data.get('age_months', 3)) if data.get('age_months') else 3
            user_count = int(data.get('user_count', 0)) if data.get('user_count') else 0
            current_revenue = float(data.get('current_revenue', 0)) if data.get('current_revenue') else 0
            
            input_data = {
                'founding_year': 2025 - age_months // 12,
                'user_growth_rate': 10 if user_count > 100 else 5,
                'has_revenue': current_revenue > 0
            }
            
            result = self.ml.predict_traction(input_data)
            
            if 'error' in result:
                return result
            
            return self._format_traction_output(result, data, source='local')
            
        except Exception as e:
            print(f"[ERROR] predict_traction: {e}", file=sys.stderr)
            return {'error': str(e)}
    
    def predict_revenue(self, data: Dict, timeframe: int = 12) -> Dict[str, Any]:
        """Call revenue prediction model"""
        # Try MCP first
        if self.mcp:
            try:
                mcp_result = self.mcp.call_tool('predict_revenue_estimate', data)
                if mcp_result and 'error' not in mcp_result:
                    return self._format_revenue_output(mcp_result, data, timeframe, source='mcp')
            except Exception as e:
                print(f"⚠️ MCP predict_revenue failed: {e}")
        
        # Fallback to local ML
        try:
            input_data = {
                'funding_raised': data.get('funding_raised', 0),
                'monthly_revenue': data.get('current_revenue', 0),
                'team_size': data.get('team_size', 2),
                'age_months': data.get('age_months', 3)
            }
            
            result = self.ml.predict_revenue(input_data)
            
            if 'error' in result:
                return result
            
            return self._format_revenue_output(result, data, timeframe, source='local')
            
        except Exception as e:
            print(f"[ERROR] predict_revenue: {e}", file=sys.stderr)
            return {'error': str(e)}
    
    def predict_breakeven(self, data: Dict) -> Dict[str, Any]:
        """Call break-even prediction model"""
        # Try MCP first
        if self.mcp:
            try:
                mcp_result = self.mcp.call_tool('predict_break_even_time', data)
                if mcp_result and 'error' not in mcp_result and mcp_result.get('success'):
                    print("   ✅ Using MCP breakeven prediction")  # ⬅️ DEBUG
                    return self._format_breakeven_output(mcp_result, data, source='mcp')
                else:
                    print(f"   ⚠️ MCP breakeven failed: {mcp_result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"⚠️ MCP predict_breakeven failed: {e}")
        
        # Fallback to local ML
        try:
            print("   🔄 Using local breakeven prediction")  # ⬅️ DEBUG
            
            # ⬅️ FIX: Handle None values properly
            team_size = int(data.get('team_size', 2) or 2)
            burn_rate = float(data.get('burn_rate_monthly_est') or (team_size * 5000))
            funding_raised = float(data.get('funding_raised', 0) or 0)
            current_revenue = float(data.get('current_revenue', 0) or 0)
            
            input_data = {
                'funding_raised': funding_raised,
                'monthly_burn_rate': burn_rate,
                'monthly_revenue': current_revenue
            }
            
            result = self.ml.predict_breakeven(input_data)
            
            if 'error' in result:
                return result
            
            return self._format_breakeven_output(result, data, source='local')
            
        except Exception as e:
            print(f"[ERROR] predict_breakeven: {e}", file=sys.stderr)
            return {'error': str(e)}

    
    def predict_survival(self, data: Dict) -> Dict[str, Any]:
        """Call survival prediction model"""
        # Try MCP first
        if self.mcp:
            try:
                mcp_result = self.mcp.call_tool('predict_survival_months', data)
                if mcp_result and 'error' not in mcp_result:
                    return self._format_survival_output(mcp_result, data, source='mcp')
            except Exception as e:
                print(f"⚠️ MCP predict_survival failed: {e}")
        
        # Fallback to local ML
        try:
            team_size = int(data.get('team_size', 2)) if data.get('team_size') else 2
            funding = float(data.get('funding_raised', 0)) if data.get('funding_raised') else 0
            revenue = float(data.get('current_revenue', 0)) if data.get('current_revenue') else 0
            age_months = int(data.get('age_months', 3)) if data.get('age_months') else 3
            
            burn_rate = team_size * 5000
            runway_months = (funding + revenue * 6) / max(burn_rate, 1)
            
            input_data = {
                'has_revenue': revenue > 0,
                'runway_months': runway_months,
                'founding_year': 2025 - age_months // 12
            }
            
            result = self.ml.predict_survival(input_data)
            
            if 'error' in result:
                return result
            
            return self._format_survival_output(result, data, source='local')
            
        except Exception as e:
            print(f"[ERROR] predict_survival: {e}", file=sys.stderr)
            return {'error': str(e)}
    
    # ================================================================
    # ANALYSIS TOOLS (MCP integrated)
    # ================================================================
    
    def find_competitors(self, data: Dict, n_results: int = 5) -> Dict[str, Any]:
        """Find similar products via MCP or hybrid search"""
        # Try MCP first
        if self.mcp:
            try:
                mcp_result = self.mcp.call_tool('find_competitors', data)
                if mcp_result and 'error' not in mcp_result:
                    return mcp_result
            except Exception as e:
                print(f"⚠️ MCP find_competitors failed: {e}")
        
        # Fallback to local database
        try:
            query_text = self._build_search_query(data)
            
            results = self.db.find_similar_products(
                query_text=query_text,
                n_results=n_results,
                category=data.get('category')
            )
            
            return {
                'competitors': results[:n_results],
                'count': len(results),
                'query_used': query_text
            }
        except Exception as e:
            print(f"[ERROR] find_competitors: {e}", file=sys.stderr)
            return {'error': str(e), 'competitors': [], 'count': 0}
    
    def journey_simulator(self, data: Dict) -> Dict[str, Any]:
        """Run journey simulation via MCP"""
        if self.mcp:
            try:
                mcp_result = self.mcp.call_tool('journey_simulator', data)
                if mcp_result and 'error' not in mcp_result:
                    return mcp_result
            except Exception as e:
                print(f"⚠️ MCP journey_simulator failed: {e}")
        
        # Fallback: Import local simulator
        try:
            from mcp_server.tools.journey_simulator import simulate_journey
            return simulate_journey(data)
        except Exception as e:
            return {'error': f'Journey simulator unavailable: {str(e)}'}
    
    # ================================================================
    # DATABASE TOOLS (Local only)
    # ================================================================
    
    def validate_revenue(self, data: Dict) -> Dict[str, Any]:
        """Validate revenue assumption"""
        try:
            query_text = self._build_search_query(data)
            target_revenue = data.get('target_revenue', data.get('current_revenue', 0))
            
            validation = self.db.cross_validate_revenue(
                query_text=query_text,
                assumed_revenue=target_revenue,
                category=data.get('category'),
                n_results=20
            )
            
            return validation
        except Exception as e:
            print(f"[ERROR] validate_revenue: {e}", file=sys.stderr)
            return {'error': str(e)}
    
    def query_benchmark(self, data: Dict) -> Dict[str, Any]:
        """Get benchmark statistics from SQL"""
        try:
            category = data.get('category', 'saas')
            
            benchmarks = {
                'saas': {'median': 15000, 'p25': 5000, 'p75': 50000},
                'mobile': {'median': 8000, 'p25': 2000, 'p75': 30000},
                'ecommerce': {'median': 25000, 'p25': 10000, 'p75': 100000}
            }
            
            stats = benchmarks.get(category, benchmarks['saas'])
            
            return {
                'category': category,
                'median_revenue': stats['median'],
                'p25_revenue': stats['p25'],
                'p75_revenue': stats['p75'],
                'sample_size': 100
            }
        except Exception as e:
            print(f"[ERROR] query_benchmark: {e}", file=sys.stderr)
            return {'error': str(e)}
    
    def calculate_math(self, data: Dict) -> Dict[str, Any]:
        """Perform revenue calculations"""
        try:
            current = float(data.get('current_revenue', 0)) if data.get('current_revenue') else 0
            target = float(data.get('target_revenue', 0)) if data.get('target_revenue') else 0
            months = int(data.get('timeframe_months', 12)) if data.get('timeframe_months') else 12
            price = float(data.get('price_point', 50)) if data.get('price_point') else 50
            
            if current > 0 and target > 0 and months > 0:
                growth_required = ((target / current) ** (1/months) - 1) * 100
            else:
                growth_required = None
            
            users_needed = target / price if price > 0 else None
            
            return {
                'current_revenue': current,
                'target_revenue': target,
                'timeframe_months': months,
                'monthly_growth_required_pct': growth_required,
                'users_needed': users_needed,
                'price_point': price
            }
        except Exception as e:
            print(f"[ERROR] calculate_math: {e}", file=sys.stderr)
            return {'error': str(e)}
    
    # ================================================================
    # FORMATTING HELPERS
    # ================================================================
    
    def _format_success_output(self, result: Dict, source: str) -> Dict:
        """Format success prediction output"""
        prob = result.get('probability', result.get('success_probability', 0.5))
        
        return {
            'success_probability': float(prob),
            'verdict': self._classify_success_prob(prob),
            'confidence': result.get('confidence', 0.7),
            'category': result.get('category', 'Unknown'),
            'source': source
        }
    
    def _format_traction_output(self, result: Dict, data: Dict, source: str) -> Dict:
        """Format traction prediction output with intelligent fallback"""
        months = result.get('months', result.get('predicted_months', 0))
        
        # ✅ If model predicts 0, use intelligent estimate
        if months <= 0:
            print("   ⚠️  Model predicted 0 traction time, using estimate...")
            current_revenue = float(data.get('current_revenue', 0) or 0)
            user_count = int(data.get('user_count', 0) or 0)
            
            if current_revenue > 1000 or user_count > 50:
                months = 6  # Already has some traction, 6 months to next milestone
            else:
                months = 12  # Early stage, 12 months typical
            
            source = 'estimated'
        
        return {
            'predicted_months': float(months),
            'verdict': self._classify_traction_time(months),
            'milestone': self._traction_milestone(data.get('category', 'saas')),
            'confidence': result.get('confidence', 0.65 if source == 'estimated' else 0.75),
            'source': source
        }
    
    def _format_revenue_output(self, result: Dict, data: Dict, timeframe: int, source: str) -> Dict:
        """Format revenue prediction output"""
        predicted_revenue = result.get('revenue', result.get('predicted_revenue', 0))
        current_revenue = data.get('current_revenue', 1)
        
        return {
            'predicted_revenue': float(predicted_revenue),
            'timeframe_months': timeframe,
            'current_revenue': current_revenue,
            'growth_multiple': predicted_revenue / max(current_revenue, 1),
            'confidence': result.get('confidence', 0.75),
            'revenue_range': {
                'low': result.get('revenue_low', predicted_revenue * 0.7),
                'high': result.get('revenue_high', predicted_revenue * 1.5)
            },
            'source': source
        }
    
    def _format_survival_output(self, result: Dict, data: Dict, source: str) -> Dict:
        """Format survival prediction output with calculated fallback"""
        # Get model prediction
        survival_months = result.get('survival_months', 
                                    result.get('months', 
                                            result.get('runway_months', 0)))
        
        # ✅ If model predicts 0 or negative, CALCULATE from data
        if survival_months <= 0:
            print("   ⚠️  Model predicted invalid survival, calculating from runway...")
            survival_months = self._calculate_runway_months(data)
            source = 'calculated'
        
        # Get detailed risk metrics
        survival_metrics = self._calculate_survival_probability(data, survival_months)
        
        return {
            'survival_months': float(survival_months),
            'verdict': self._classify_survival(survival_months),
            'critical_risk': survival_metrics['critical_risk'],
            'risk_level': survival_metrics['risk_level'],
            'prob_12m': survival_metrics['prob_12m'],
            'runway_months': survival_metrics['runway_months'],
            'warnings': result.get('warnings', []),
            'source': source
        }


    def _format_breakeven_output(self, result: Dict, data: Dict, source: str) -> Dict:
        """Format breakeven prediction output with calculated fallback"""
        months = result.get('months', result.get('breakeven_months', 0))
        
        # ✅ If model predicts 0 or 1 (suspiciously low), CALCULATE
        if months <= 1:
            print("   ⚠️  Model predicted unrealistic breakeven, calculating from metrics...")
            months = self._calculate_breakeven_months(data)
            source = 'calculated'
        
        # Calculate runway to check feasibility
        runway_months = self._calculate_runway_months(data)
        
        return {
            'breakeven_months': float(months),
            'verdict': self._classify_breakeven_time(months),
            'runway_sufficient': months < runway_months,
            'runway_months': runway_months,
            'months_short': max(0, months - runway_months),
            'confidence': result.get('confidence', 0.7),
            'source': source
        }
    
    # ================================================================
    # HELPER METHODS (unchanged)
    # ================================================================

    def _calculate_runway_months(self, data: Dict) -> float:
        """Calculate actual runway from data"""
        funding = float(data.get('funding_raised', 0) or 0)
        revenue = float(data.get('current_revenue', 0) or 0)
        burn_rate = float(data.get('burn_rate_monthly_est') or 0)
        
        # If no burn rate provided, estimate from team size
        if burn_rate == 0:
            team_size = int(data.get('team_size', 2) or 2)
            burn_rate = team_size * 5000  # $5K per person average
        
        # Calculate net burn (burn - revenue)
        net_burn = max(burn_rate - revenue, 1)  # At least $1 to avoid division by zero
        
        # Calculate runway
        runway_months = funding / net_burn
        
        return max(0, min(runway_months, 120))  # Clamp between 0-120 months


    def _calculate_breakeven_months(self, data: Dict) -> float:
        """Calculate breakeven from actual metrics"""
        current_revenue = float(data.get('current_revenue', 0) or 0)
        burn_rate = float(data.get('burn_rate_monthly_est') or 0)
        
        # If no burn rate, estimate from team
        if burn_rate == 0:
            team_size = int(data.get('team_size', 2) or 2)
            burn_rate = team_size * 5000
        
        # If already profitable
        if current_revenue >= burn_rate:
            return 1  # Already at breakeven
        
        # Calculate revenue gap
        revenue_gap = burn_rate - current_revenue
        
        # Estimate growth rate from timeframe (default: 10% MoM growth)
        monthly_growth_rate = 0.10
        
        # Calculate months to reach breakeven
        # Formula: revenue * (1 + growth)^months = burn_rate
        # months = log(burn_rate / revenue) / log(1 + growth)
        if current_revenue > 0:
            import math
            months = math.log(burn_rate / current_revenue) / math.log(1 + monthly_growth_rate)
            return max(1, min(months, 120))
        else:
            # No revenue yet, use longer estimate
            return 18  # Typical for pre-revenue startups


    def _calculate_survival_probability(self, data: Dict, survival_months: float) -> Dict:
        """Calculate survival risk metrics"""
        runway = self._calculate_runway_months(data)
        
        if survival_months <= 0:
            # Model failed, use runway calculation
            survival_months = runway
        
        # Risk levels
        if survival_months >= 24:
            risk_level = "Low"
            prob_12m = 0.85
        elif survival_months >= 12:
            risk_level = "Medium"
            prob_12m = 0.70
        elif survival_months >= 6:
            risk_level = "High"
            prob_12m = 0.50
        else:
            risk_level = "Critical"
            prob_12m = 0.25
        
        return {
            'survival_months': survival_months,
            'runway_months': runway,
            'risk_level': risk_level,
            'prob_12m': prob_12m,
            'critical_risk': survival_months < 12
        }

    
    def _build_search_query(self, data: Dict) -> str:
        """Build search query from extracted data"""
        parts = []
        
        if data.get('product_name'):
            parts.append(data['product_name'])
        
        if data.get('product_description'):
            parts.append(data['product_description'])
        
        if data.get('category'):
            parts.append(f"{data['category']} product")
        
        return " ".join(parts) if parts else "startup product"
    
    def _classify_success_prob(self, prob: float) -> str:
        if prob >= 0.75: return "STRONG SUCCESS SIGNALS"
        if prob >= 0.50: return "MODERATE SUCCESS POTENTIAL"
        if prob >= 0.30: return "HIGH RISK"
        return "VERY HIGH RISK"
    
    def _classify_traction_time(self, months: float) -> str:
        if months <= 6: return "FAST TRACK"
        if months <= 12: return "TYPICAL TIMELINE"
        if months <= 24: return "LONG ROAD"
        return "VERY LONG TIMELINE"
    
    def _classify_breakeven_time(self, months: float) -> str:
        if months <= 6: return "NEAR-TERM PROFITABILITY"
        if months <= 12: return "STANDARD PATH"
        if months <= 24: return "LONG ROAD"
        return "UNSUSTAINABLE"
    
    def _classify_survival(self, months: float) -> str:
        if months >= 36: return "LONG-TERM VIABLE"
        if months >= 24: return "MODERATE RUNWAY"
        if months >= 12: return "SHORT RUNWAY"
        return "CRITICAL RISK"
    
    def _traction_milestone(self, category: str) -> str:
        milestones = {
            'saas': '$10K MRR or 100+ paying customers',
            'mobile': '10K+ MAU or app store featured',
            'ecommerce': '$50K+ monthly GMV',
            'food': '1K+ orders/month'
        }
        return milestones.get(category, 'Meaningful user traction')
