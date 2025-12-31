#!/usr/bin/env python3
"""
PROD-IQ Orchestrator - Connects LLM + ML Models + Database
"""
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from llm.mcp_client import MCPToolExecutor

from llm.conversation_state import ConversationStateManager, TurnType

# Import your backends
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_core.inference import InferenceEngine
from llm.extractor import DataExtractor
from llm.executor import ToolExecutor

class ProdIQOrchestrator:
    """Main orchestrator connecting LLM + ML + Database"""
    
    def __init__(self, prompts_dir: str = "prompts"):
        """Initialize all components"""
        
        print("🚀 Initializing PROD-IQ Orchestrator...")
        
        # 1. Test LLM API connection
        print("   Connecting to LLM API...")
        from llm.local_inference import get_llm
        self.llm = get_llm()  # This now just tests connection
        print("   ✅ LLM API ready")
            
        # 2. Load prompts (FIX: Use absolute path from script location)
        print("   Loading prompts...")
        script_dir = Path(__file__).resolve().parent.parent  # Go up to project root
        self.prompts_dir = script_dir / prompts_dir
        print(f"   📁 Prompts directory: {self.prompts_dir}")
        self.prompts = self._load_all_prompts()
        print(f"   ✅ Loaded {len(self.prompts)} prompts")
        
        # 3. Initialize backends
        print("   Loading backends...")
        self.ml_inference = InferenceEngine()
        
        # Use mock for now (skip slow PyTorch loading)
        print("   Loading databases...")
        from database.hybrid_search import HybridSearch
        self.hybrid_search = HybridSearch()
        
        # 4. Initialize DataExtractor
        self.data_extractor = DataExtractor()
        
        # 5. Initialize ToolExecutor
        self.tool_executor = ToolExecutor(
            ml_inference=self.ml_inference,
            hybrid_search=self.hybrid_search
        )
        
        # 6. Conversation history (OLD - kept for backward compatibility)
        self.conversation_history = []

        # ✅ ADD: Conversation state manager (NEW)
        self.state_manager = ConversationStateManager()

        try:
            self.mcp_tools = MCPToolExecutor()
            print("   ✅ MCP tools initialized")
        except Exception as e:
            print(f"   ⚠️ MCP unavailable, using local tools: {e}")
            self.mcp_tools = None

        from llm.simple_state import SimpleStateManager
        self.state = SimpleStateManager()
        
        print("✅ PROD-IQ Orchestrator ready with Simple State!\n")
    
    def _load_all_prompts(self) -> Dict[str, str]:
        """Load all prompt files"""
        prompts = {}
        
        prompt_files = [
            'global_prompt.txt',
            'guardrails.txt',
            'task_extraction.txt',
            'task_orchestration.txt',
            'task_predict_success.txt',
            'task_predict_traction_time.txt',
            'task_predict_revenue.txt',
            'task_predict_breakeven.txt',
            'task_predict_survival.txt',
            'task_journey_simulator.txt',
            'task_revenue_validation.txt',
            'task_competitor_analysis.txt',
            'task_psychology_coaching.txt',
            'task_math_calculation.txt',
            'follow_up_prompt.txt',
            'clarification_prompt.txt',
            'benchmark_query_prompt.txt'
        ]
        
        for filename in prompt_files:
            filepath = self.prompts_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    key = filename.replace('.txt', '')
                    prompts[key] = f.read().strip()
                print(f"      ✓ {filename}")
            else:
                print(f"      ✗ {filename} NOT FOUND at {filepath}")
        
        return prompts

    def _load_prompt(self, prompt_filename: str) -> str:
        """
        Load a single prompt file
        
        Args:
            prompt_filename: Name of prompt file (e.g., "journey_simulator_prompt.txt")
        
        Returns:
            Prompt content as string
        """
        # Try to get from already loaded prompts first
        prompt_key = prompt_filename.replace('.txt', '').replace('_prompt', '')
        
        # Check various formats
        possible_keys = [
            prompt_filename.replace('.txt', ''),  # "journey_simulator_prompt"
            prompt_key,  # "journey_simulator"
            f"task_{prompt_key}",  # "task_journey_simulator"
        ]
        
        for key in possible_keys:
            if key in self.prompts:
                print(f"   ✅ Using cached prompt: {key}")
                return self.prompts[key]
        
        # If not found in cache, try loading from file
        prompt_path = self.prompts_dir / prompt_filename
        
        if prompt_path.exists():
            print(f"   ✅ Loading prompt from file: {prompt_filename}")
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        # Fallback to global prompt
        print(f"   ⚠️ Prompt not found: {prompt_filename}, using global_prompt")
        return self.prompts.get('global_prompt', 'You are a helpful AI assistant.')

    
    def process_query(self, user_query: str, chat_history: Optional[List[Dict]] = None,
                      explicit_tools: Optional[List[str]] = None, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Main entry point with conversation state management"""
        
        print(f"\n{'='*60}")
        print(f"🧑‍💻 USER: {user_query}")
        print(f"{'='*60}\n")

        if chat_history:
            self._restore_chat_history(chat_history)
            print(f"📚 Restored {len(chat_history)} previous turns\n")
        
        # ✅ STEP 0: Determine turn type
        turn_type = self.state_manager.get_turn_type(user_query)
        print(f"🔍 Turn Type: {turn_type}")
        print(f"📊 Turn Number: {len(self.state_manager.turns) + 1}\n")
        
        # STEP 1: Extract data (with accumulated context)
        print("📊 STEP 1: Extracting data...")
        extracted_data = self._extract_data(user_query, context, turn_type)
        print(f"   ✅ Extracted: {json.dumps(extracted_data, indent=2)}\n")
        
        print("🔀 STEP 2: Routing to tools...")
        if explicit_tools:
            tools_to_call = explicit_tools
            print(f"   🎯 Using explicit tools from frontend: {tools_to_call}")
        else:
            tools_to_call = self._route_query(user_query, extracted_data, turn_type)
            print(f"   ✅ Auto-routed tools: {tools_to_call}")
        print()
        
        # STEP 3: Execute tools
        print("⚙️ STEP 3: Executing tools...")
        tool_outputs = self._execute_tools(tools_to_call, self.state)
        print(f"   ✅ Tool outputs received\n")
        
        # STEP 4: Generate response (style depends on turn_type)
        print("🤖 STEP 4: Generating response...")
        final_response = self._generate_response(
            user_query, extracted_data, tools_to_call, 
            tool_outputs, turn_type
        )
        print(f"   ✅ Response generated\n")
        
        # ✅ STEP 5: Store in conversation state
        self.state_manager.add_turn(
            user_query=user_query,
            turn_type=turn_type,
            extracted_data=extracted_data,
            tools_called=tools_to_call,
            tool_outputs=tool_outputs,
            response=final_response
        )
        
        # Also store in old format (backward compatibility)
        self.conversation_history.append({
            'user': user_query,
            'assistant': final_response,
            'extracted_data': extracted_data,
            'tools_called': tools_to_call
        })
        
        return {
            'response': final_response,
            'tools_called': tools_to_call,
            'extracted_data': extracted_data,
            'turn_type': turn_type,
            'turn_number': len(self.state_manager.turns),
            'conversation_history': self.state_manager.export_history(),  # ✅ For frontend to save
            'raw_tool_outputs': tool_outputs,
            'state_snapshot': self.state.get_state()  # ✅ Current state
        }
    
    def process_special_tool_query(
    self, 
    user_query: str,
    tool_name: str,  # "journey_simulator" or "competitor_analysis"
    context: Optional[Dict] = None,
    chat_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Handle user-triggered special tools (Journey Simulator & Competitor Analysis)
        This is a separate flow from normal query processing.
        
        Args:
            user_query: User's input text
            tool_name: "journey_simulator" or "competitor_analysis"
            context: Optional context override
            chat_history: Previous conversation history
        
        Returns:
            Response dict with tool output and narrative
        """
        
        print(f"\n{'='*60}")
        print(f"🔧 SPECIAL TOOL TRIGGERED: {tool_name}")
        print(f"🧑‍💻 USER: {user_query}")
        print(f"{'='*60}\n")
        
        # Restore chat history if provided
        if chat_history:
            self._restore_chat_history(chat_history)
        
        # Extract data from query
        # ✅ CORRECT: Use the same extraction logic as normal queries
        from llm.simple_extractor import extract_delta_simple

        delta = extract_delta_simple(user_query, self.state, self.state.turn_count + 1)
        extracted_data = self.state.update(delta, self.state.turn_count + 1)
        extracted_data = self._calculate_derived_metrics(extracted_data)

        print(f"📊 EXTRACTED DATA:\n{json.dumps(extracted_data, indent=2)}\n")
        
        # Execute the special tool
        print(f"⚙️ Executing: {tool_name}\n")
        tool_output = self._execute_special_tool(tool_name, extracted_data)
        
        # Generate response using special prompt
        response = self._generate_special_tool_response(
            user_query=user_query,
            tool_name=tool_name,
            tool_output=tool_output,
            extracted_data=extracted_data
        )
        
        # Update state using the correct method
        self.state_manager.add_turn(
            user_query=user_query,
            turn_type='special_tool',  # or TurnType.SPECIAL
            extracted_data=extracted_data,
            tools_called=[tool_name],
            tool_outputs={tool_name: tool_output},
            response=response
        )

        self.conversation_history.append({
            'turn': self.state_manager.turns,
            'user_query': user_query,
            'special_tool': tool_name,
            'response': response
        })
        
        return {
            'response': response,
            'tool_used': tool_name,
            'tool_output': tool_output,
            'extracted_data': extracted_data,
            'turn_number': len(self.state_manager.turns), 
            'conversation_history': self.conversation_history
        }


    def _execute_special_tool(self, tool_name: str, data: Dict) -> Dict:
        """
        # Execute special user-triggered tools via MCP
        # """
        
        if tool_name == "journey_simulator":
            # ✅ CORRECT: Call journey_simulator
            from mcp_server.tools.journey_simulator import simulate_journey
            return simulate_journey(data)
        
        elif tool_name == "competitor_analysis":
            from mcp_server.tools.market_scout import market_scout_handler
            
            # ✅ REMOVED asyncio - call directly
            result = market_scout_handler(data)
            return result
        
        else:
            raise ValueError(f"Unknown special tool: {tool_name}")


    def _generate_special_tool_response(
    self, 
    user_query: str,
    tool_name: str,
    tool_output: Dict,
    extracted_data: Dict
) -> str:
        """
        Generate LLM response for special tools WITHOUT showing raw JSON/metadata
        """
        
        # ✅ Get prompt key
        if tool_name == "journey_simulator":
            prompt_key = "task_journey_simulator"
        elif tool_name == "competitor_analysis":
            prompt_key = "task_competitor_analysis"
        else:
            prompt_key = "global_prompt"
        
        # Get system prompt
        system_prompt = self.prompts.get(prompt_key, self.prompts.get('global_prompt', ''))
        
        # ✅ CLEAN SUMMARY - No JSON, no metadata
        def clean_tool_summary(output: Dict, tool_type: str) -> str:
            """Convert tool output to clean narrative data"""
            
            if tool_type == "journey_simulator":
                current = output.get("current_state", {})
                scenarios = output.get("scenarios", {})
                realistic = scenarios.get("realistic", {})
                analysis = output.get("analysis", {})
                
                summary = "SIMULATION DATA (use these numbers in your response):\n\n"
                summary += "Current Financial State:\n"
                summary += f"- Monthly Revenue: ${current.get('revenue_monthly', 0):,.0f}\n"
                summary += f"- Monthly Burn: ${current.get('burn_monthly', 0):,.0f}\n"
                summary += f"- Net Burn: ${current.get('net_burn_monthly', 0):,.0f}\n"
                summary += f"- Cash Remaining: ${current.get('cash_remaining', 0):,.0f}\n"
                summary += f"- Current Runway: {current.get('runway_months', 'N/A')} months\n"
                
                summary += "\nRealistic Scenario (15% monthly growth):\n"
                
                # Key milestones
                trajectory = realistic.get("monthly_trajectory", [])
                for month_data in trajectory:
                    m = month_data.get("month", 0)
                    if m in [0, 3, 6, 9, 12]:
                        rev = month_data.get("revenue", 0)
                        burn = month_data.get("burn", 0)
                        cash = month_data.get("cash", 0)
                        runway = month_data.get("runway_months", 0)
                        profitable = "✅ PROFITABLE" if month_data.get("is_profitable") else ""
                        summary += f"  Month {m}: Revenue ${rev:,.0f}, Burn ${burn:,.0f}, Cash ${cash:,.0f}, Runway {runway:.1f}mo {profitable}\n"
                
                summary += f"\nBreakeven Month: {realistic.get('breakeven_month', 'Never')}\n"
                summary += f"Cash Depletion: Month {realistic.get('cash_depletion_month', 'Never')}\n"
                
                # Risks
                risks = analysis.get("risks", [])
                if risks:
                    summary += "\nCritical Risks:\n"
                    for risk in risks[:3]:
                        summary += f"  - {risk.get('severity', '').upper()}: {risk.get('message', '')}\n"
                
                return summary
                
            elif tool_type == "competitor_analysis":
                company = output.get("company", "Unknown")
                verdict = output.get("verdict", "unknown")
                count = output.get("competitor_count", 0)
                top_comps = output.get("top_competitors", [])
                market = output.get("market_analysis", {})
                
                summary = "COMPETITIVE ANALYSIS DATA (use these facts in your response):\n\n"
                summary += f"Your Company: {company}\n"
                summary += f"Market Verdict: {verdict.replace('_', ' ').title()}\n"
                summary += f"Direct Competitors Found: {count}\n\n"
                
                summary += "Top 5 Competitors:\n"
                for i, comp in enumerate(top_comps[:5], 1):
                    name = comp.get("name", "Unknown")
                    rev = comp.get("revenue", 0)
                    funding = comp.get("funding", 0)
                    summary += f"  {i}. {name}"
                    if rev:
                        summary += f" - Revenue: ${rev:,.0f}/mo"
                    if funding:
                        summary += f", Funding: ${funding:,.0f}"
                    summary += "\n"
                
                summary += f"\nMarket Maturity: {market.get('maturity', 'Unknown')}\n"
                summary += f"Average Competitor Revenue: ${market.get('avg_revenue', 0):,.0f}/mo\n"
                
                return summary
            
            else:
                return "No data available"
        
        # ✅ Build CLEAN user prompt
        clean_data = clean_tool_summary(tool_output, tool_name)
        
        user_prompt = f"""USER QUESTION:

    YOUR TASK:
    Write a comprehensive narrative response analyzing the startup's situation.

    STRUCTURE YOUR RESPONSE:
    1. Opening Diagnosis (2-3 sentences summarizing current state)
    2. Detailed Analysis (reference specific numbers from data above)
    3. Critical Risks (use ⚠️ for warnings)
    4. Actionable Recommendations (3-5 bullet points)

    STRICT REQUIREMENTS:
    ✅ Use actual numbers from the data above
    ✅ Keep everything in USD ($) - DO NOT convert currencies
    ✅ Write in clear, direct English
    ❌ DO NOT include any JSON or tool metadata
    ❌ DO NOT echo back the data section
    ❌ DO NOT show currency conversions
    ❌ DO NOT include tool names or parameters

    Maximum 600 words. Begin your response now:
    """
        
        # ✅ Generate response
        from llm.local_inference import generate
        
        response = generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=1500,  # Increased from 800
            temperature=0.7
        )
        
        # ✅ Clean any leaked metadata
        response = self._remove_metadata_leaks(response)
        
        return response


    def _remove_metadata_leaks(self, text: str) -> str:
        """Remove any JSON or metadata that leaked into response"""
        import re
        
        # Remove JSON blocks
        text = re.sub(r'\{[^}]*"tool":[^}]*\}', '', text)
        text = re.sub(r'\{[^}]*"parameters":[^}]*\}', '', text)
        text = re.sub(r'\{[^}]*"revenue_estimated":[^}]*\}', '', text)
        
        # Remove tool call indicators
        text = re.sub(r'Query:.*?Found \d+.*?\n', '', text, flags=re.DOTALL)
        
        # Remove currency conversion lines
        text = re.sub(r'Currency:.*?₹\d+.*?\n', '', text)
        text = re.sub(r'Currency Standardization.*?\n', '', text)
        
        # Remove "SIMULATION DATA" section if it appears in output
        text = re.sub(r'SIMULATION DATA.*?YOUR TASK:', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()



    def _summarize_journey_output(self, tool_output: Dict) -> str:
        """
        Compact summary of journey simulator output (with None-safe formatting)
        """
        
        current_state = tool_output.get("current_state", {})
        scenarios = tool_output.get("scenarios", {})
        analysis = tool_output.get("analysis", {})
        
        # ✅ Safe number conversion
        def safe_num(value, default=0):
            try:
                return float(value) if value is not None else float(default)
            except (TypeError, ValueError):
                return float(default)
        
        # Extract and convert BEFORE f-string
        revenue = safe_num(current_state.get('revenue_monthly'))
        cash = safe_num(current_state.get('cash_remaining'))
        runway = current_state.get('runway_months', 'N/A')
        
        summary = f"""
    SIMULATION RESULTS:

    Current: ${revenue:,.0f}/mo revenue, ${cash:,.0f} cash, {runway} mo runway

    Scenarios (24 months):
    """
        
        for name in ["optimistic", "realistic", "pessimistic"]:
            s = scenarios.get(name, {})
            m24 = s.get("key_milestones", {}).get("month_24", {})
            
            breakeven = s.get('breakeven_month', 'Never')
            cash_24 = safe_num(m24.get('cash'))
            rev_24 = safe_num(m24.get('revenue'))
            
            summary += f"  {name.upper()}: Breakeven={breakeven}, Cash@24mo=${cash_24:,.0f}, Rev@24mo=${rev_24:,.0f}\n"
        
        # Add top risk only
        risks = analysis.get("risks", [])
        if risks and len(risks) > 0:
            summary += f"\nTOP RISK: {risks[0].get('message', 'N/A')}\n"
        
        summary += f"Recommendation: {analysis.get('recommendation', 'N/A')}"
        
        return summary

    def _restore_chat_history(self, chat_history: List[Dict]):
        """Restore conversation from frontend"""
        self.state_manager.reset()  
        
        for turn in chat_history:
            self.state_manager.add_turn(
                user_query=turn.get('user', ''),
                turn_type=turn.get('turn_type', 'follow_up'),
                extracted_data=turn.get('extracted_data', {}),
                tools_called=turn.get('tools_called', []),
                tool_outputs=turn.get('tool_outputs', {}),
                response=turn.get('assistant', '')
            )
            
            # Also restore state
            if turn.get('extracted_data'):
                self.state.update(turn['extracted_data'], turn.get('turn', 1))

    def _extract_data(self, query: str, context: Optional[Dict], turn_type: str) -> Dict:
        """Extract delta and update state"""
        from llm.simple_extractor import extract_delta_simple
        
        # Extract changes
        delta = extract_delta_simple(query, self.state, self.state.turn_count + 1)
        
        # Apply to state
        updated_state = self.state.update(delta, self.state.turn_count + 1)
        
        # ✅ ADD: Calculate derived metrics
        updated_state = self._calculate_derived_metrics(updated_state)
        
        # Context override (if provided externally)
        if context:
            context_delta = {k: v for k, v in context.items() if k in self.state.state}
            if context_delta:
                updated_state = self.state.update(context_delta, self.state.turn_count + 1)
        
        return updated_state


    def _calculate_derived_metrics(self, state: Dict) -> Dict:
        """Calculate revenue, runway, etc. from base metrics"""
        
        # ✅ 1. Handle commission rate
        raw_commission = state.get('commission_rate_pct')
        commission_rate = (float(raw_commission) / 100.0) if raw_commission else 0.0
        
        # ✅ 2. Calculate revenue from multiple sources
        
        # Option A: Order-based revenue (cloud kitchens, marketplaces)
        orders_per_day = state.get('orders_per_day')
        avg_order_value = state.get('average_order_value')
        
        if orders_per_day and avg_order_value:
            monthly_gmv = orders_per_day * 30 * avg_order_value
            
            if commission_rate > 0:
                monthly_revenue = monthly_gmv * commission_rate
                state['monthly_revenue_current'] = monthly_revenue
                print(f"   🧮 Revenue: {orders_per_day} orders/day × ₹{avg_order_value} × 30 × {commission_rate*100}% = ₹{monthly_revenue:,.0f}/month")
            else:
                # No commission = full GMV is revenue
                state['monthly_revenue_current'] = monthly_gmv
                print(f"   🧮 Revenue: {orders_per_day} orders/day × ₹{avg_order_value} × 30 = ₹{monthly_gmv:,.0f}/month")
        
        # Option B: Subscription revenue (if not order-based)
        elif not state.get('monthly_revenue_current'):
            paying_customers = state.get('customer_count_paying') or state.get('user_count_total')
            price_per_unit = state.get('price_per_unit')
            
            if paying_customers and price_per_unit:
                monthly_revenue = paying_customers * price_per_unit
                state['monthly_revenue_current'] = monthly_revenue
                print(f"   🧮 Revenue: {paying_customers} customers × ₹{price_per_unit} = ₹{monthly_revenue:,.0f}/month")
        
        # ✅ 3. Calculate runway
        cash = state.get('cash_remaining') or state.get('funding_raised_total') or 0
        burn = state.get('burn_rate_monthly') or 0
        revenue = state.get('monthly_revenue_current') or 0
        
        if burn > 0:
            net_burn = burn - revenue
            
            if net_burn > 0 and cash > 0:
                runway = cash / net_burn
                state['runway_months'] = round(runway, 1)
                print(f"   🧮 Runway: ₹{cash:,.0f} ÷ (₹{burn:,.0f} - ₹{revenue:,.0f}) = {runway:.1f} months")
            elif net_burn <= 0:
                state['runway_months'] = 999
                print(f"   🧮 Profitable: Revenue ≥ Burn")
        
        # ✅ 4. Calculate breakeven (optional)
        if revenue > 0 and burn > 0:
            months_to_breakeven = burn / (revenue * 0.2)  # Assuming 20% MoM growth
            state['breakeven_months'] = round(months_to_breakeven, 1)
        
        return state


    def _fallback_extraction(self, query: str, context: Optional[Dict]) -> Dict:
        """Fallback extraction when prompts fail"""
        
        # Start with previous context if available
        if self.conversation_history:
            last_turn = self.conversation_history[-1]
            base_data = last_turn.get('extracted_data', {}).copy()
        else:
            base_data = {
                'product_name': 'AI Writing Tool',
                'category': 'saas',
                'stage': 'launched',
                'current_revenue': 2000,
                'target_revenue': 10000,
                'team_size': 2,
                'funding_raised': 50000,
                'user_count': 40,
                'price_point': 50,
                'age_months': 6,
                'burn_rate_monthly_est': 8000
            }
        
        # Update core_question based on query
        query_lower = query.lower()
        if 'competitor' in query_lower:
            base_data['core_question'] = 'competitor_analysis'
        elif 'survive' in query_lower or 'runway' in query_lower:
            base_data['core_question'] = 'survival_analysis'
        elif 'realistic' in query_lower or 'validate' in query_lower:
            base_data['core_question'] = 'revenue_validation'
        else:
            base_data['core_question'] = 'general_advice'
        
        # Merge with provided context
        if context:
            base_data.update(context)
        
        return base_data


    def _route_query(self, query: str, extracted_data: Dict, turn_type: TurnType) -> List[str]:
        """Route tools based on query and turn type"""
        

        # ✅ FIX: CHECK SCENARIO FIRST (before turn type)
        if extracted_data.get('scenario_active', False):
            print("   🎭 Scenario active - re-running ALL predictions")
            return [
                'predict_success_label',
                'predict_revenue_estimate',
                'predict_traction_time',
                'predict_break_even_time',
                'predict_survival_months'
            ]
        
        # ✅ For clarifications, don't call new tools
        if turn_type == 'clarification':
            print("   ℹ️  Clarification detected - using previous tool outputs")
            return []
        
        # ✅ NOW check follow-up logic
        if turn_type == 'follow_up':
            query_lower = query.lower()
            
            calculation_keywords = [
                'calculate', 'math', 'if we', 'what if', 'how much',
                'cut', 'reduce', 'increase', 'grow', 'extend',
                'how long', 'runway', 'survive', 'last', 'have left',
                'boost', 'speed up', 'worse', 'better', 'impact'
            ]
            
            if any(word in query_lower for word in calculation_keywords):
                print("   ℹ️  Follow-up with calculation needed")
                return ['calculate_math']
            
            print("   ℹ️  Simple follow-up - no new tools needed")
            return []
        
        # ✅ For benchmark queries
        if turn_type == 'benchmark_query':
            print("   ℹ️  Benchmark query detected")
            return ['find_competitors', 'validate_revenue']
        
        # ✅ For data updates, re-run relevant tools
        if turn_type == 'data_update':
            print("   ℹ️  Data update detected - re-running core tools")
            return ['predict_success_label', 'predict_survival_months', 'calculate_math']
        
        # ✅ For initial queries, use LLM routing
        return self._full_routing(query, extracted_data)


    def _full_routing(self, query: str, extracted_data: Dict) -> List[str]:
        """Full LLM-based routing for initial queries"""
        
        system_prompt = """You are PROD-IQ routing agent. Output ONLY the <tools> block, nothing else.

    AVAILABLE TOOLS:
    - predict_success_label
    - predict_revenue_estimate
    - predict_traction_time
    - predict_break_even_time
    - predict_survival_months
    - find_competitors
    - validate_revenue
    - calculate_math

    OUTPUT FORMAT (required):
    <tools>
    <tool>tool_name_1</tool>
    <tool>tool_name_2</tool>
    </tools>

    EXAMPLES:

    Query: "Is $10K MRR realistic?"
    <tools>
    <tool>validate_revenue</tool>
    <tool>predict_revenue_estimate</tool>
    <tool>calculate_math</tool>
    </tools>

    Query: "Will we survive?"
    <tools>
    <tool>predict_survival_months</tool>
    <tool>predict_break_even_time</tool>
    <tool>calculate_math</tool>
    </tools>

    Output ONLY <tools> block for this query:"""

        user_message = f"{query}\n\nExtracted data: {json.dumps(extracted_data)}"
        
        from llm.local_inference import generate
        response = generate(
            user_prompt=user_message,
            system_prompt=system_prompt,
            max_tokens=200,
            temperature=0.1
        )
        
        # Parse tools
        tools = self.data_extractor.parse_tools(response)
        
        if not tools:
            print(f"   ⚠️  LLM routing failed, using fallback...")
            # Fallback based on keywords
            query_lower = query.lower()
            if 'revenue' in query_lower or 'realistic' in query_lower:
                tools = ['validate_revenue', 'predict_revenue_estimate', 'calculate_math']
            elif 'competitor' in query_lower:
                tools = ['find_competitors']
            elif 'survive' in query_lower or 'runway' in query_lower:
                tools = ['predict_survival_months', 'predict_break_even_time', 'calculate_math']
            else:
                tools = ['predict_success_label', 'predict_revenue_estimate']
        
        return tools


    def _execute_tools(self, tools: List[str], state_manager) -> Dict[str, Any]:
        """Execute tools with state manager"""
        
        outputs = {}
        
        # 1. Translate State to ML Inputs
        data = self._state_to_ml_input(state_manager)
        
        # ✅ FIX: STORE BASELINE FIRST
        baseline_data = None
        if data.get('scenario_active'):
            print("   🎭 Scenario detected - storing baseline")
            
            # Store original values
            baseline_data = {
                'revenue': data.get('current_revenue', 0),
                'burn': data.get('burn_rate_monthly_est', 0),
                'price': data.get('price_point', 0),
                'business_model': data.get('business_model', 'B2C')
            }
            
            print(f"   📊 Baseline: Rev=${baseline_data['revenue']}, Burn=${baseline_data['burn']}, Model={baseline_data['business_model']}")
            
            # NOW swap to scenario values
            if data.get('scenario_price') and data['scenario_price'] > 0:
                data['price_point'] = data['scenario_price']
                print(f"   🔄 Price: ${baseline_data['price']} → ${data['price_point']}")
            
            if data.get('scenario_revenue') and data['scenario_revenue'] > 0:
                data['current_revenue'] = data['scenario_revenue']
                print(f"   🔄 Revenue: ${baseline_data['revenue']} → ${data['current_revenue']}")
            
            if data.get('scenario_burn') and data['scenario_burn'] > 0:
                data['burn_rate_monthly_est'] = data['scenario_burn']
                print(f"   🔄 Burn: ${baseline_data['burn']} → ${data['burn_rate_monthly_est']}")
            
            if data.get('scenario_business_model'):
                data['business_model'] = data['scenario_business_model']
                print(f"   🔄 Model: {baseline_data['business_model']} → {data['business_model']}")
        
        # 2. Execute tools
        for tool_name in tools:
            print(f"   🔧 Calling {tool_name}...")
            
            try:
                result = None
                
                if tool_name == 'calculate_math':
                    result = self.tool_executor.calculate_math(data)
                
                elif tool_name == 'predict_success_label':
                    result = self.tool_executor.predict_success(data)
                    # ✅ Tag with baseline for comparison
                    if baseline_data:
                        result['_baseline'] = baseline_data
                        result['_is_scenario'] = True
                
                elif tool_name == 'predict_revenue_estimate':
                    result = self.tool_executor.predict_revenue(data)
                    if baseline_data:
                        result['_baseline'] = baseline_data
                        result['_is_scenario'] = True
                
                if result:
                    outputs[tool_name] = result
                    print(f"      ✅ {tool_name} complete")
                
            except Exception as e:
                print(f"      ❌ {tool_name} failed: {e}")
                outputs[tool_name] = {'error': str(e)}
        
        # ✅ ADD: Include previous turn's predictions as baseline
        if baseline_data and len(self.state_manager.turns) > 0:
            prev_turn = self.state_manager.turns[-1]
            outputs['_baseline_predictions'] = {
                'success': prev_turn.tool_outputs.get('predict_success_label'),
                'revenue': prev_turn.tool_outputs.get('predict_revenue_estimate'),
                'survival': prev_turn.tool_outputs.get('predict_survival_months'),
                'breakeven': prev_turn.tool_outputs.get('predict_break_even_time')
            }
        
        return outputs


    def _format_previous_tool_outputs(self) -> str:
        """Format tool outputs from previous turns for reference"""
        if self.state_manager.is_first_turn():
            return "None (first turn)"
        
        recent_outputs = []
        for turn in self.state_manager.turns[-2:]:
            if turn.tool_outputs:
                for tool, output in turn.tool_outputs.items():
                    # Format output snippet
                    output_str = json.dumps(output, indent=2)
                    if len(output_str) > 200:
                        output_str = output_str[:200] + "..."
                    recent_outputs.append(f"Turn {turn.turn_number} - {tool}:\n{output_str}")
        
        return "\n\n".join(recent_outputs) if recent_outputs else "None"


    def _is_json_response(self, response: str) -> bool:
        """Detect if response is primarily JSON"""
        response_stripped = response.strip()
        
        # Check if starts with JSON
        if response_stripped.startswith('{') or response_stripped.startswith('['):
            try:
                json.loads(response_stripped)
                return True  # Valid JSON = bad response
            except:
                pass
        
        # Check JSON density
        json_indicators = response.count('{') + response.count('}') + response.count('"tool"') + response.count('"parameters"')
        total_length = len(response)
        
        if total_length > 0 and json_indicators > 10:  # More than 10 JSON chars = likely JSON
            return True
        
        return False


    def _generate_response(self, user_query: str, extracted_data: Dict, 
                      tools_called: List[str], tool_outputs: Dict,
                      turn_type: TurnType) -> str:
        """Generate response based on turn type"""
        
        # ✅ Select appropriate prompt based on turn type
        if turn_type == 'initial':
            base_prompt = self.prompts.get('global_prompt', '')
            use_template = True
            max_words = 150
        elif turn_type == 'follow_up':
            base_prompt = self.prompts.get('follow_up_prompt', '')
            use_template = False
            max_words = 120
        elif turn_type == 'clarification':
            base_prompt = self.prompts.get('clarification_prompt', '')
            use_template = False
            max_words = 100
        elif turn_type == 'benchmark_query':
            base_prompt = self.prompts.get('benchmark_query_prompt', '')
            use_template = False
            max_words = 130
        else:
            base_prompt = self.prompts.get('global_prompt', '')
            use_template = True
            max_words = 150
        
        # ✅ Add conversation context for non-initial turns
        conversation_context = ""
        if turn_type != 'initial' and not self.state_manager.is_first_turn():
            conversation_context = self.state_manager.get_context_summary(last_n_turns=2)
            conversation_context += "\n⚠️ This is a FOLLOW-UP question. Reference previous context!\n"
        
        # ✅ ADD: Scenario comparison context
        scenario_context = ""
        if extracted_data.get('scenario_active'):
            baseline = tool_outputs.get('_baseline_predictions', {})
            
            if baseline:
                scenario_context = f"""
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    🎭 SCENARIO COMPARISON MODE - YOU MUST COMPARE!
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    📊 BASELINE (Current Situation):
    - Success: {self._safe_get(baseline, 'success', 'success_probability', 'N/A')}
    - Revenue: ${self._safe_get(baseline, 'revenue', 'predicted_revenue', 'N/A')}
    - Survival: {self._safe_get(baseline, 'survival', 'survival_months', 'N/A')} months
    - Breakeven: {self._safe_get(baseline, 'breakeven', 'breakeven_months', 'N/A')} months

    🔮 NEW SCENARIO (What-If):
    - Success: {self._safe_get(tool_outputs, 'predict_success_label', 'success_probability', 'N/A')}
    - Revenue: ${self._safe_get(tool_outputs, 'predict_revenue_estimate', 'predicted_revenue', 'N/A')}
    - Survival: {self._safe_get(tool_outputs, 'predict_survival_months', 'survival_months', 'N/A')} months
    - Breakeven: {self._safe_get(tool_outputs, 'predict_break_even_time', 'breakeven_months', 'N/A')} months

    ⚠️ CRITICAL INSTRUCTIONS:
    1. Calculate percentage changes for each metric
    2. Explicitly state which scenario is better
    3. Explain WHY the changes occur
    4. Give clear recommendation: "Pivot" or "Stay with current"
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
        
        # ✅ Build task context
        task_context = f"""
    {conversation_context}
    {scenario_context}

    CURRENT TASK CONTEXT:
    --------------------
    User Question: {user_query}
    Turn Type: {turn_type}

    Accumulated Metrics (all conversation):
    {json.dumps(extracted_data, indent=2)}

    Tool Results (current turn):
    {self._format_tool_outputs(tool_outputs)}

    Previous Tool Results (for reference):
    {self._format_previous_tool_outputs()}

    RESPONSE REQUIREMENTS:
    1. {'USE TEMPLATE FORMAT (Diagnosis/Options/Timeline/Next Steps)' if use_template else 'BE CONVERSATIONAL - answer naturally, NO TEMPLATE'}
    2. Keep response under {max_words} words
    3. Use EXACT numbers from tool results
    4. Write in natural language - NO JSON OUTPUT
    5. {'Reference previous conversation context' if turn_type != 'initial' else 'Provide comprehensive initial analysis'}

    Generate your response now:"""

        system_prompt = base_prompt + task_context
        
        # Call LLM with retry logic
        max_retries = 2
        for attempt in range(max_retries):
            response = self._call_llm(
                system_prompt=system_prompt,
                user_prompt=user_query,
                max_tokens=600 if use_template else 400,
                temperature=0.7
            )
            
            if self._is_json_response(response):
                print(f"   ⚠️  Attempt {attempt+1}: LLM returned JSON, retrying...")
                system_prompt += """\n\n🚨 CRITICAL: You returned JSON. This is FORBIDDEN. Use natural English only."""
                continue
            
            break
        
        # Clean response
        response_match = re.search(r'<response>(.*?)</response>', response, re.DOTALL)
        if response_match:
            final_response = response_match.group(1).strip()
        else:
            final_response = response
        
        final_response = re.sub(r'<[^>]+>', '', final_response)
        final_response = re.sub(r'\n{3,}', '\n\n', final_response)
        
        # Final fallback if still JSON
        if self._is_json_response(final_response):
            print("   🔧 Using intelligent fallback...")
            final_response = self._convert_json_to_text(final_response, user_query, tool_outputs, extracted_data)
        
        return final_response.strip()
    
    # ✅ ADD: Validate response numbers against inputs
    def _validate_response_numbers(self, response: str, extracted_data: Dict, tool_outputs: Dict) -> str:
        """Check if response numbers match tool outputs"""
        
        # Extract numbers from response
        import re
        response_numbers = re.findall(r'₹(\d+(?:,\d+)*)', response)
        
        # Check against actual revenue
        actual_revenue = extracted_data.get('monthly_revenue_current')
        if actual_revenue and actual_revenue > 100000:  # > 1 lakh
            # Check if response mentions revenue
            if 'revenue' in response.lower() or 'earning' in response.lower():
                # Verify at least one number is close to actual
                actual_lakhs = actual_revenue / 100000
                found_match = False
                
                for num_str in response_numbers:
                    num = int(num_str.replace(',', ''))
                    if abs(num - actual_lakhs) < actual_lakhs * 0.2:  # Within 20%
                        found_match = True
                        break
                
                if not found_match:
                    print(f"   ⚠️  Response revenue mismatch: mentions {response_numbers}, actual is ₹{actual_revenue:,.0f}")
                    # Don't auto-fix, but log for debugging
        
        return response


    def _safe_get(self, dict_obj, *keys, default='N/A'):
        """Safely extract nested dict values"""
        try:
            result = dict_obj
            for key in keys:
                result = result.get(key, {})
            return result if result != {} else default
        except:
            return default


    def _clean_response(self, response: str, strict_limit: bool = False) -> str:
        """Clean response and optionally enforce word limit"""
        
        # Remove XML tags
        response = re.sub(r'<[^>]+>', '', response)
        
        # Remove excessive whitespace
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        # Only enforce word limit if strict (usually False for production)
        if strict_limit:
            words = response.split()
            if len(words) > 160:
                # Cut at sentence boundary
                sentences = response.split('. ')
                result = ''
                for sent in sentences:
                    if len((result + sent).split()) <= 150:
                        result += sent + '. '
                    else:
                        break
                response = result.strip()
        
        return response.strip()

    def _convert_json_to_text(self, json_response: str, user_query: str, tool_outputs: Dict, extracted_data: Dict) -> str:  # ✅ ADDED PARAMETER
        """Emergency fallback: Convert tool outputs to natural language"""
        
        # ✅ Extract from tool outputs
        success_output = tool_outputs.get('predict_success_label', {})
        survival_output = tool_outputs.get('predict_survival_months', {})
        revenue_output = tool_outputs.get('predict_revenue_estimate', {})
        breakeven_output = tool_outputs.get('predict_break_even_time', {})
        calc_output = tool_outputs.get('calculate_math', {})
        
        # ✅ Get values with safe defaults
        success_prob = success_output.get('success_probability', 0.5)
        survival_months = survival_output.get('survival_months', 6)
        revenue_predicted = revenue_output.get('predicted_revenue', 0)
        breakeven_months = breakeven_output.get('breakeven_months', 12)
        runway_months = survival_output.get('runway_months', survival_months)
        
        # Get extracted data context (NOW DEFINED!)
        current_revenue = extracted_data.get('current_revenue', 0)
        target_revenue = extracted_data.get('target_revenue', 0)
        burn_rate = extracted_data.get('burn_rate_monthly_est', 0)
        
        # ✅ Format numbers cleanly
        success_pct = int(success_prob * 100)
        survival_str = f"{survival_months:.1f}"
        breakeven_str = f"{breakeven_months:.1f}"
        
        # Build response based on what data we have
        diagnosis = f"Your startup shows {success_pct}% success probability. "
        
        if survival_months < 12:
            diagnosis += f"With {survival_str} months of runway, immediate action is critical."
        else:
            diagnosis += f"You have {survival_str} months of runway to work with."
        
        # Options based on situation
        options = []
        
        if survival_months < 12:
            options.append("A. **Cut Burn Immediately**: Reduce costs by 30-40% to extend runway to 12+ months")
        
        if current_revenue > 0:
            growth_needed = ((target_revenue / current_revenue) - 1) * 100 if target_revenue > current_revenue else 0
            if growth_needed > 0:
                options.append(f"B. **Accelerate Revenue**: Need {growth_needed:.0f}% growth to hit ${target_revenue:,.0f} target")
        else:
            options.append("B. **Get First Revenue**: Focus exclusively on getting first paying customers")
        
        if breakeven_months < runway_months:
            options.append(f"C. **Path to Profitability**: {breakeven_str} months to breakeven if you maintain trajectory")
        else:
            shortfall = breakeven_months - runway_months
            options.append(f"C. **Funding Needed**: You'll run out of money {shortfall:.1f} months before breakeven")
        
        # Timeline
        if breakeven_months < runway_months:
            timeline = f"You can reach breakeven in {breakeven_str} months without additional funding."
        else:
            timeline = f"Critical: You need to extend runway or accelerate growth within {survival_str} months."
        
        # Next steps
        next_steps = [
            "1. Calculate exact weekly burn rate and cut non-essential costs now",
            "2. Double down on your best customer acquisition channel",
            "3. Set aggressive weekly growth targets and track daily"
        ]
        
        # ✅ Assemble response in correct format (matching global prompt)
        response = f"""**Diagnosis:**
    {diagnosis}

    **Options:**
    {chr(10).join(options)}

    **Timeline:**
    {timeline}

    **Next Steps:**
    {chr(10).join(next_steps)}
    """
        
        return response


    def predict_revenue_estimate(self, data: Dict) -> Dict:
        """Try MCP first, fallback to local"""
        if self.orchestrator.mcp_tools:
            try:
                result = self.orchestrator.mcp_tools.predict_revenue(data)
                if result:
                    return result
            except:
                pass
        
        # Fallback to local model
        return self.inference_engine.predict_revenue(data)


    def _call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Call LLM using new API format"""
        from llm.local_inference import generate
        
        # Truncate if too long
        # if len(user_prompt) > 4000:
        #     print(f"   ⚠️  User prompt too long ({len(user_prompt)} chars), truncating...")
        #     user_prompt = user_prompt[:4000] + "\n\n[Context truncated. Answer based on above.]"
        
        response = generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.strip()

    def _validate_response_completeness(self, response: str) -> bool:
        pass

    def _format_chat_prompt(self, messages: List[Dict]) -> str:
       pass

    def _format_conversation_history(self, max_turns: int = 2) -> str:
        """Format history"""
        if not self.conversation_history:
            return "No previous conversation"
        
        recent = self.conversation_history[-max_turns:]
        formatted = []
        for turn in recent:
            formatted.append(f"User: {turn['user']}\nAssistant: {turn['assistant']}")
        return "\n".join(formatted)
    
    def _parse_tools_from_response(self, response: str) -> List[str]:
        """Extract tool names"""
        tools = []
        
        tool_block_match = re.search(r'<tool_calls>(.*?)</tool_calls>', response, re.DOTALL)
        if tool_block_match:
            tool_matches = re.findall(r'<tool>(.*?)</tool>', tool_block_match.group(1))
            tools.extend(tool_matches)
        
        tool_patterns = [
            'predict_success_label', 'predict_traction_time', 'predict_revenue_estimate',
            'predict_break_even_time', 'predict_survival_months', 'find_competitors',
            'validate_revenue', 'query_benchmark', 'calculate_math'
        ]
        
        for pattern in tool_patterns:
            if pattern in response and pattern not in tools:
                tools.append(pattern)
        
        return tools
    
    def _select_task_prompt(self, tools: List[str]) -> str:
        """Select task prompt"""
        if 'predict_success_label' in tools:
            return self.prompts.get('task_predict_success', '')
        if 'predict_revenue_estimate' in tools:
            return self.prompts.get('task_predict_revenue', '')
        if 'validate_revenue' in tools:
            return self.prompts.get('task_revenue_validation', '')
        return ""
    
    def _format_tool_outputs(self, outputs: Dict) -> str:
        """Format outputs"""
        formatted = []
        for tool, output in outputs.items():
            formatted.append(f"{tool}: {json.dumps(output, indent=2)}")
        return "\n\n".join(formatted)
    
    def reset_conversation(self):
        """Clear history"""
        self.conversation_history = []
        self.state_manager.reset()

    def load_conversation_history(self, history: List[Dict]):
        """Load conversation history from frontend"""
        # TODO: Implement when frontend integration is ready
        for turn_data in history:
            if isinstance(turn_data, dict):
                self.state_manager.add_turn(
                    user_query=turn_data.get('user_query', ''),
                    turn_type=turn_data.get('turn_type', 'follow_up'),
                    extracted_data=turn_data.get('extracted_data', {}),
                    tools_called=turn_data.get('tools_called', []),
                    tool_outputs=turn_data.get('tool_outputs', {}),
                    response=turn_data.get('response', '')
                )
    def _detect_data_changes(self, new_data: Dict) -> List[str]:
        """
        Compare new extracted data against the accumulated history.
        Returns a list of fields that have changed or are new.
        """
        changed_fields = []
        
        # Get the previous state (what we knew before this turn)
        # Note: You need to access the state manager's data BEFORE the update
        # For simplicity, we assume new_data contains the MERGED result, 
        # so we check if the user input actually provided these keys this turn.
        
        # Better approach: Check what the extractor output specifically for THIS turn
        # In your _process_query, you get 'extracted_data'. 
        # We need to look at what was explicitly extracted from the user's LATEST message.
        
        # Let's assume you pass the *raw extraction* (before merge) to this function
        # Or check if values differ from state_manager.accumulated_data
        
        current_state = self.state_manager.accumulated_data
        
        for key, new_value in new_data.items():
            # Skip metadata fields
            if key in ['question_type', 'core_question']:
                continue
                
            old_value = current_state.get(key)
            
            # If value is different and not None/Zero (unless changing TO zero)
            if new_value != old_value:
                # Type safe comparison
                if str(new_value) != str(old_value): 
                    changed_fields.append(key)
                    
        return changed_fields
    
    def _state_to_ml_input(self, state_manager) -> Dict:
        """Convert state to format expected by ML models (Handle None values!)"""
        state = state_manager.get_state()
        
        # Helper to ensure we never send None to math functions
        def val(key, default):
            v = state.get(key)
            return v if v is not None else default

        return {
            'current_revenue': val('monthly_revenue_current', 0),
            'funding_raised': val('funding_raised_total', 0),
            'team_size': val('team_size', 2),
            # Use 'or 0' to handle cases where extraction returns None explicitly
            'age_months': val('company_age_months', 0), 
            'product_name': val('product_name', 'Product'),
            'category': val('category', 'saas'),
            'stage': val('launch_stage', 'idea'),
            'burn_rate_monthly_est': val('burn_rate_monthly', 0),
            'price_point': val('price_per_unit', 0),
            'user_count': val('user_count_total', 0),
            'target_revenue': val('monthly_revenue_target', 0),
            'timeframe_months': val('timeframe_months', 6),
            'growth_rate_monthly': val('growth_rate_monthly_pct', 20) / 100,
            
            # Scenario fields
            'scenario': state.get('scenario_description'),
            'scenario_price': state.get('scenario_price'),
            'scenario_revenue': state.get('scenario_revenue'),
            'scenario_burn': state.get('scenario_burn'),
            'scenario_active': state.get('scenario_active', False)
        }

    # ✅ ADD reset method:
    def reset_session(self):
        """Reset conversation"""
        self.state.reset()
        print("✅ Session reset!\n")

def create_orchestrator():
    return ProdIQOrchestrator()
