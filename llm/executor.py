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
    
    def execute_with_flag(self, tool_name: str, data: Dict) -> Dict:
        """
        Execute a single tool with explicit flag
        Used when frontend sends explicit tool request
        """
        
        print(f"   🎯 Explicit tool call: {tool_name}")
        
        # Map tool names to methods
        tool_map = {
            'find_competitors': self.find_competitors,
            'validate_revenue': self.validate_revenue,
            'predict_success': self.predict_success,
            'predict_revenue': self.predict_revenue,
            'predict_survival': self.predict_survival,
            'calculate_math': self.calculate_math,
            'journey_simulator': self.journey_simulator,  # ✅ Story telling
            'benchmark_query': self.query_benchmark
        }
        
        if tool_name not in tool_map:
            return {'error': f'Unknown tool: {tool_name}'}
        
        try:
            result = tool_map[tool_name](data)
            print(f"      ✅ {tool_name} complete")
            return result
        except Exception as e:
            print(f"      ❌ {tool_name} failed: {e}")
            return {'error': str(e)}

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
                    print("   ✅ Using MCP breakeven prediction")
                    return self._format_breakeven_output(mcp_result, data, source='mcp')
                else:
                    print(f"   ⚠️ MCP breakeven failed: {mcp_result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"⚠️ MCP predict_breakeven failed: {e}")
        
        # Fallback to local ML
        try:
            print("   🔄 Using local breakeven prediction")
            
            # ✅ FIX: Handle None values properly with proper defaults
            team_size = int(data.get('team_size') or 2)
            burn_rate_input = data.get('burn_rate_monthly_est')
            
            # Only estimate if burn_rate is truly None/missing
            if burn_rate_input is None or burn_rate_input == 0:
                burn_rate = float(team_size * 5000)
            else:
                burn_rate = float(burn_rate_input)
            
            funding_raised = float(data.get('funding_raised') or 0)
            current_revenue = float(data.get('current_revenue') or 0)
            
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
            import traceback
            traceback.print_exc()
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
    
    def calculate_math(self, state_manager) -> Dict[str, Any]:
        """Enhanced math calculations with growth projections and scenarios"""
        try:
            # ✅ FIX 1: CHECK FOR SCENARIO state_manager FIRST
            # if state_manager.get('scenario') or state_manager.get('scenario_revenue') or state_manager.get('scenario_burn') or state_manager.get('scenario_ad_spend'):
            #     print(f"   📊 Scenario detected: {state_manager.get('scenario', 'hypothetical')}")
            #     return self._calculate_scenario_comparison(state_manager)
            
            # Extract base metrics
            current_revenue = float(state_manager.get('current_revenue', 0) or 0)
            target_revenue = float(state_manager.get('target_revenue', 0) or 0)
            months = int(state_manager.get('timeframe_months', 6) or 6)
            price = float(state_manager.get('price_point', 50) or 50)
            if state_manager.get('scenario_active', False):
                return self._calculate_scenario_comparison(state_manager)

            burn_rate = float(state_manager.get('burn_rate_monthly_est', 0) or 0)
            funding = float(state_manager.get('funding_raised', 0) or 0)
            user_count = int(state_manager.get('user_count', 0) or 0)
            growth_rate_monthly = float(state_manager.get('growth_rate_monthly', 0.20) or 0.20)
            age_months = int(state_manager.get('age_months', 0) or 0)  # ✅ ADDED
            
            # Calculate runway
            runway_months = self._calculate_runway_months(state_manager)
            
            # Calculate remaining funding
            already_spent = burn_rate * age_months
            remaining_funding = max(funding - already_spent, 0)
            
            # ✅ SCENARIO 1: Revenue Growth Projections
            projections = {}
            if current_revenue > 0:
                for multiplier, label in [(1, 'current_growth'), (2, '2x_growth'), (3, '3x_growth')]:
                    adjusted_rate = growth_rate_monthly * multiplier
                    projected_revenue = {}
                    
                    for horizon in [3, 6, 12]:
                        if horizon <= months * 2:
                            future_revenue = current_revenue * ((1 + adjusted_rate) ** horizon)
                            projected_revenue[f'{horizon}_months'] = round(future_revenue, 2)
                    
                    projections[label] = {
                        'monthly_growth_rate': f"{adjusted_rate*100:.1f}%",
                        'annual_growth_rate': f"{((1+adjusted_rate)**12 - 1)*100:.1f}%",
                        'projections': projected_revenue
                    }
            
            # ✅ SCENARIO 2: Burn Rate Scenarios (FIXED)
            burn_scenarios = {}
            if burn_rate > 0:
                for reduction_pct, label in [(0, 'current'), (0.20, 'cut_20pct'), (0.30, 'cut_30pct')]:
                    new_burn = burn_rate * (1 - reduction_pct)
                    net_burn = new_burn - current_revenue
                    
                    if net_burn <= 0:
                        new_runway = 999
                        runway_status = "profitable"
                    else:
                        new_runway = remaining_funding / net_burn if remaining_funding > 0 else 0
                        runway_status = "burning_cash"
                    
                    burn_scenarios[label] = {
                        'burn_rate': round(new_burn, 2),
                        'net_burn': round(net_burn, 2),
                        'runway_months': round(new_runway, 1) if new_runway < 999 else 'infinite',
                        'runway_extension': 'N/A' if new_runway >= 999 else round(new_runway - runway_months, 2),
                        'status': runway_status
                    }
            
            # ✅ SCENARIO 3-6: Keep your existing code unchanged
            equity_valuations = {}
            if current_revenue > 0:
                annual_revenue = current_revenue * 12
                
                for multiple, label in [(5, 'conservative'), (10, 'market'), (15, 'optimistic')]:
                    valuation = annual_revenue * multiple
                    
                    equity_valuations[label] = {
                        'company_valuation': round(valuation, 2),
                        'valuation_multiple': f"{multiple}x ARR",
                        'equity_30_pct': round(valuation * 0.30, 2),
                        'equity_25_pct': round(valuation * 0.25, 2),
                        'equity_20_pct': round(valuation * 0.20, 2),
                        'equity_15_pct': round(valuation * 0.15, 2)
                    }
            
            growth_analysis = {}
            if current_revenue > 0 and target_revenue > 0 and months > 0:
                growth_required = ((target_revenue / current_revenue) ** (1/months) - 1) * 100
                projected_with_current_growth = current_revenue * ((1 + growth_rate_monthly) ** months)
                gap = target_revenue - projected_with_current_growth
                gap_pct = (gap / target_revenue) * 100 if target_revenue > 0 else 0
                
                growth_analysis = {
                    'current_growth_rate_monthly': f"{growth_rate_monthly*100:.1f}%",
                    'required_growth_rate_monthly': f"{growth_required:.1f}%",
                    'growth_acceleration_needed': f"{(growth_required - growth_rate_monthly*100):.1f}%",
                    'achievable_with_current_growth': gap < 0,
                    'projected_revenue_at_current_growth': round(projected_with_current_growth, 2),
                    'revenue_gap': round(gap, 2),
                    'gap_percentage': f"{abs(gap_pct):.1f}%"
                }
            
            user_economics = {}
            if price > 0:
                users_needed_target = target_revenue / price if target_revenue > 0 else 0
                users_needed_growth = users_needed_target - user_count if user_count > 0 else users_needed_target
                users_per_month = users_needed_growth / months if months > 0 else 0
                
                user_economics = {
                    'price_point': price,
                    'current_users': user_count,
                    'users_for_target': round(users_needed_target, 0),
                    'new_users_needed': round(users_needed_growth, 0),
                    'users_per_month_needed': round(users_per_month, 0),
                    'weekly_signups_needed': round(users_per_month / 4, 0)
                }
            
            survival_timeline = {}
            if runway_months > 0 or runway_months == 999:
                breakeven_months = self._calculate_breakeven_months(state_manager)
                
                survival_timeline = {
                    'runway_months': round(runway_months, 1) if runway_months < 999 else 'infinite',
                    'breakeven_months': round(breakeven_months, 1),
                    'runway_sufficient': breakeven_months < runway_months,
                    'months_short': round(max(0, breakeven_months - runway_months), 1) if runway_months < 999 else 0,
                    'critical_deadline': runway_months < 12 if runway_months < 999 else False,
                    'runway_end_date': self._calculate_date_from_now(runway_months) if runway_months < 999 else 'N/A'
                }
            
            return {
                'success': True,
                'base_metrics': {
                    'current_revenue': current_revenue,
                    'target_revenue': target_revenue,
                    'timeframe_months': months,
                    'current_burn_rate': burn_rate,
                    'runway_months': round(runway_months, 1) if runway_months < 999 else 'infinite',
                    'growth_rate_monthly': f"{growth_rate_monthly*100:.1f}%",
                    'remaining_funding': remaining_funding,
                    'already_spent': already_spent
                },
                'revenue_projections': projections,
                'burn_scenarios': burn_scenarios,
                'equity_valuations': equity_valuations,
                'growth_analysis': growth_analysis,
                'user_economics': user_economics,
                'survival_timeline': survival_timeline
            }
            
        except Exception as e:
            print(f"[ERROR] calculate_math: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'base_metrics': {
                    'current_revenue': state_manager.get('current_revenue', 0),
                    'target_revenue': state_manager.get('target_revenue', 0)
                }
            }

        
    def _calculate_scenario_comparison(self, data: Dict) -> Dict[str, Any]:
        """Calculate scenario vs baseline comparison for what-if questions"""
        try:
            # Extract baseline metrics
            baseline_revenue = float(data.get('current_revenue', 0) or 0)
            baseline_burn = float(data.get('burn_rate_monthly_est', 0) or 0)
            funding = float(data.get('funding_raised', 0) or 0)
            age_months = int(data.get('age_months', 0) or 0)
            
            # Extract scenario metrics (use baseline if not provided)
            scenario_revenue = float(data.get('scenario_revenue', baseline_revenue) or baseline_revenue)
            scenario_burn = float(data.get('scenario_burn', baseline_burn) or baseline_burn)
            scenario_ad_spend = float(data.get('scenario_ad_spend', 0) or 0)
            
            # If scenario_ad_spend provided, adjust scenario_burn
            if scenario_ad_spend > 0:
                current_ad_spend = float(data.get('ad_spend_monthly', 0) or 0)
                ad_spend_increase = scenario_ad_spend - current_ad_spend
                scenario_burn = baseline_burn + ad_spend_increase
            
            # Calculate remaining funding
            already_spent = baseline_burn * age_months
            remaining_funding = max(funding - already_spent, 0)
            
            # Calculate net burn (negative = profitable)
            baseline_net_burn = baseline_burn - baseline_revenue
            scenario_net_burn = scenario_burn - scenario_revenue
            
            # Calculate runway for both scenarios
            if baseline_net_burn <= 0:
                baseline_runway = 999  # Profitable
                baseline_status = "profitable"
            else:
                baseline_runway = remaining_funding / baseline_net_burn
                baseline_status = "burning_cash"
            
            if scenario_net_burn <= 0:
                scenario_runway = 999  # Profitable
                scenario_status = "profitable"
            else:
                scenario_runway = remaining_funding / scenario_net_burn
                scenario_status = "burning_cash"
            
            # Determine if scenario is better
            runway_change = scenario_runway - baseline_runway
            
            if scenario_net_burn < baseline_net_burn:
                verdict = "speeds_up_profitability"
                impact = "positive"
            elif scenario_net_burn > baseline_net_burn:
                verdict = "slows_down_profitability"
                impact = "negative"
            else:
                verdict = "no_change"
                impact = "neutral"
            
            # Calculate breakeven for both
            baseline_breakeven = self._calculate_breakeven_from_metrics(
                baseline_revenue, baseline_burn, data.get('growth_rate_monthly', 0.10)
            )
            scenario_breakeven = self._calculate_breakeven_from_metrics(
                scenario_revenue, scenario_burn, data.get('growth_rate_monthly', 0.10)
            )
            
            return {
                'success': True,
                'scenario_type': 'comparison',
                'baseline': {
                    'revenue': baseline_revenue,
                    'burn': baseline_burn,
                    'net_burn': baseline_net_burn,
                    'runway_months': round(baseline_runway, 1) if baseline_runway < 999 else 'infinite',
                    'breakeven_months': round(baseline_breakeven, 1),
                    'status': baseline_status
                },
                'scenario': {
                    'revenue': scenario_revenue,
                    'burn': scenario_burn,
                    'net_burn': scenario_net_burn,
                    'runway_months': round(scenario_runway, 1) if scenario_runway < 999 else 'infinite',
                    'breakeven_months': round(scenario_breakeven, 1),
                    'status': scenario_status
                },
                'comparison': {
                    'revenue_change': scenario_revenue - baseline_revenue,
                    'burn_change': scenario_burn - baseline_burn,
                    'net_burn_change': scenario_net_burn - baseline_net_burn,
                    'runway_change': round(runway_change, 1) if runway_change < 999 else 'improved',
                    'breakeven_change': round(scenario_breakeven - baseline_breakeven, 1),
                    'verdict': verdict,
                    'impact': impact
                },
                'remaining_funding': remaining_funding,
                'already_spent': already_spent
            }
            
        except Exception as e:
            print(f"[ERROR] _calculate_scenario_comparison: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _calculate_breakeven_from_metrics(self, revenue: float, burn: float, growth_rate: float) -> float:
        """Calculate breakeven months from specific metrics"""
        if revenue >= burn:
            return 1  # Already at breakeven
        
        if revenue <= 0:
            return 18  # Pre-revenue
        
        if growth_rate <= 0:
            return 999  # Can't reach breakeven without growth
        
        try:
            import math
            months = math.log(burn / revenue) / math.log(1 + growth_rate)
            return max(1, min(months, 120))
        except:
            return 12  # Default fallback

    
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
        
        # ✅ Calculate cash-out date
        cashout_date = self._calculate_date_from_now(survival_months)
        
        return {
            'survival_months': float(survival_months),
            'verdict': self._classify_survival(survival_months),
            'critical_risk': survival_metrics['critical_risk'],
            'risk_level': survival_metrics['risk_level'],
            'prob_12m': survival_metrics['prob_12m'],
            'runway_months': survival_metrics['runway_months'],
            'cash_out_date': cashout_date,
            'days_remaining': int(survival_months * 30),
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
        months_short = max(0, months - runway_months)
        
        # ✅ Calculate breakeven date
        breakeven_date = self._calculate_date_from_now(months)
        
        return {
            'breakeven_months': float(months),
            'verdict': self._classify_breakeven_time(months),
            'runway_sufficient': months < runway_months,
            'runway_months': runway_months,
            'months_short': months_short,
            'breakeven_date': breakeven_date,
            'funding_gap': round(months_short * float(data.get('burn_rate_monthly_est', 0) or 5000), 2) if months_short > 0 else 0,
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
        burn_rate = float(data.get('burn_rate_monthly_est', 0) or 0)
        growth_rate_monthly = float(data.get('growth_rate_monthly', 0.10) or 0.10)
        
        # If no burn rate, estimate from team
        if burn_rate == 0:
            team_size = int(data.get('team_size', 2) or 2)
            burn_rate = team_size * 5000
        
        # If already profitable
        if current_revenue >= burn_rate:
            return 1  # Already at breakeven
        
        # If no revenue yet
        if current_revenue <= 0:
            return 18  # Typical for pre-revenue startups
        
        # ✅ Calculate months to reach burn rate with growth
        # Formula: revenue * (1 + growth)^months = burn_rate
        # months = log(burn_rate / revenue) / log(1 + growth)
        try:
            import math
            if growth_rate_monthly > 0:
                months = math.log(burn_rate / current_revenue) / math.log(1 + growth_rate_monthly)
                return max(1, min(months, 120))  # Clamp between 1-120 months
            else:
                # No growth, calculate linear
                revenue_gap = burn_rate - current_revenue
                return revenue_gap / max(current_revenue * 0.05, 100)  # Assume 5% monthly increase minimum
        except:
            return 12  # Default fallback



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
    
    def _calculate_date_from_now(self, months: float) -> str:
        """Calculate future date from now"""
        try:
            from datetime import datetime, timedelta
            future_date = datetime.now() + timedelta(days=months * 30)
            return future_date.strftime("%B %Y")
        except:
            return f"~{int(months)} months from now"

    def get_previous_calculation(self, tool_name: str) -> Optional[Dict]:
        """
        Get previous calculation result for follow-up questions
        This would be called by orchestrator to reference past results
        """
        # This is a placeholder - actual implementation would
        # query the conversation state manager through orchestrator
        return None

