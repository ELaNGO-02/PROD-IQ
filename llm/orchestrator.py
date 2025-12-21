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
        
        # 6. Conversation history
        self.conversation_history = []

        try:
            self.mcp_tools = MCPToolExecutor()
            print("   ✅ MCP tools initialized")
        except Exception as e:
            print(f"   ⚠️ MCP unavailable, using local tools: {e}")
            self.mcp_tools = None

        
        print("✅ PROD-IQ Orchestrator ready!\n")
    
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
            'task_math_calculation.txt'
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
    
    def process_query(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Main entry point"""
        
        print(f"\n{'='*60}")
        print(f"🧑‍💻 USER: {user_query}")
        print(f"{'='*60}\n")
        
        # STEP 1: Extract data
        print("📊 STEP 1: Extracting data...")
        extracted_data = self._extract_data(user_query, context)
        print(f"   ✅ Extracted: {json.dumps(extracted_data, indent=2)}\n")
        
        # STEP 2: Route to tools
        print("🔀 STEP 2: Routing to tools...")
        tools_to_call = self._route_query(user_query, extracted_data)
        print(f"   ✅ Tools to call: {tools_to_call}\n")
        
        # STEP 3: Execute tools
        print("⚙️ STEP 3: Executing tools...")
        tool_outputs = self._execute_tools(tools_to_call, extracted_data)
        print(f"   ✅ Tool outputs received\n")
        
        # STEP 4: Generate response
        print("🤖 STEP 4: Generating response...")
        final_response = self._generate_response(user_query, extracted_data, tools_to_call, tool_outputs)
        print(f"   ✅ Response generated\n")
        
        # Store history
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
            'raw_tool_outputs': tool_outputs
        }
    
    def _extract_data(self, query: str, context: Optional[Dict]) -> Dict:
        """Extract structured data using LLM's <extraction> tag"""
        
        # Get previous context for follow-up queries
        previous_context = {}
        if self.conversation_history:
            last_turn = self.conversation_history[-1]
            previous_context = last_turn.get('extracted_data', {})
        
        # Build extraction prompt
        system_prompt = """You are PROD-IQ extraction agent. Extract startup metrics from user query.

    OUTPUT FORMAT:
    <thinking>Brief analysis of what user is asking</thinking>
    <extraction>
    {
    "current_mrr": number,
    "target_mrr": number or null,
    "current_users": number,
    "price": number or null,
    "total_funding": number,
    "current_burn": number or null,
    "growth_months": number,
    "product_name": "string",
    "category": "saas|mobile|ecommerce|other"
    }
    </extraction>

    Use actual numbers. If not mentioned, use null or sensible defaults."""

        # Include context for follow-up
        user_message = query
        if previous_context:
            user_message = f"Previous context: {json.dumps(previous_context)}\n\nNew query: {query}"
        
        # Call LLM
        from llm.local_inference import generate
        response = generate(
            user_prompt=user_message,
            system_prompt=system_prompt,
            max_tokens=400,
            temperature=0.3
        )
        
        # Parse extraction
        extracted = self.data_extractor.parse_extraction(response)
        
        # Merge with previous context (for follow-ups)
        if previous_context:
            merged = previous_context.copy()
            for key, value in extracted.items():
                if value not in [None, 0, '', 'saas', 'general_advice']:
                    merged[key] = value
            extracted = merged
        
        # Merge with provided context
        if context:
            extracted.update(context)
        
        return extracted

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

    
    # In orchestrator.py - Update _route_query method

    def _route_query(self, query: str, extracted_data: Dict) -> List[str]:
        """Route using LLM's <tools> tag"""
        
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
    <tool>tool_name_3</tool>
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

    Query: "Who are our competitors?"
    <tools>
    <tool>find_competitors</tool>
    <tool>query_benchmark</tool>
    </tools>

    Output ONLY <tools> block for this query:"""

        user_message = f"{query}\n\nExtracted data: {json.dumps(extracted_data)}"
        
        from llm.local_inference import generate
        response = generate(
            user_prompt=user_message,
            system_prompt=system_prompt,
            max_tokens=200,
            temperature=0.1  # ✅ VERY LOW for structured output
        )
        
        # Parse tools
        tools = self.data_extractor.parse_tools(response)
        
        if not tools:
            print(f"   ⚠️  LLM response: {response[:200]}...")  # DEBUG
            # Fallback based on core_question
            core_q = extracted_data.get('core_question', '')
            if 'revenue' in core_q or 'validate' in core_q:
                tools = ['validate_revenue', 'predict_revenue_estimate', 'calculate_math']
            elif 'competitor' in core_q:
                tools = ['find_competitors', 'query_benchmark']
            elif 'survival' in core_q:
                tools = ['predict_survival_months', 'predict_break_even_time']
            else:
                tools = ['predict_success_label', 'predict_revenue_estimate']
            print(f"   ✅ Using fallback tools based on core_question: {tools}")
        
        return tools


    
    def _execute_tools(self, tools: List[str], extracted_data: Dict) -> Dict:
        """Execute all tools"""
        outputs = {}
        
        for tool_name in tools:
            try:
                print(f"   🔧 Calling {tool_name}...")
                
                if tool_name == 'predict_success_label':
                    outputs[tool_name] = self.tool_executor.predict_success(extracted_data)
                elif tool_name == 'predict_traction_time':
                    outputs[tool_name] = self.tool_executor.predict_traction(extracted_data)
                elif tool_name == 'predict_revenue_estimate':
                    outputs[tool_name] = self.tool_executor.predict_revenue(extracted_data)
                elif tool_name == 'predict_break_even_time':
                    outputs[tool_name] = self.tool_executor.predict_breakeven(extracted_data)
                elif tool_name == 'predict_survival_months':
                    outputs[tool_name] = self.tool_executor.predict_survival(extracted_data)
                elif tool_name == 'find_competitors':
                    outputs[tool_name] = self.tool_executor.find_competitors(extracted_data)
                elif tool_name == 'validate_revenue':
                    outputs[tool_name] = self.tool_executor.validate_revenue(extracted_data)
                elif tool_name == 'query_benchmark':
                    outputs[tool_name] = self.tool_executor.query_benchmark(extracted_data)
                elif tool_name == 'calculate_math':
                    outputs[tool_name] = self.tool_executor.calculate_math(extracted_data)
                
                print(f"      ✅ {tool_name} complete")
                
            except Exception as e:
                print(f"      ⚠️ {tool_name} error: {e}")
                outputs[tool_name] = {'error': str(e)}
        
        return outputs
    
    def _generate_response(self, user_query: str, extracted_data: Dict, tools_called: List[str], tool_outputs: Dict) -> str:
        """Generate final response using structured LLM with JSON detection"""
        
        # Build system prompt with STRONG anti-JSON instructions
        system_prompt = """You are PROD-IQ, a startup advisor. Generate actionable advice based on data and tool outputs.

    CRITICAL OUTPUT RULES:
    - NEVER output raw JSON or structured data
    - NEVER output tool results directly
    - ALWAYS write in natural, conversational language
    - Use Markdown formatting (bold, bullets, headers)
    - Speak as if advising a founder in person

    OUTPUT FORMAT:
    <thinking>Brief analysis</thinking>
    <response>
    **Diagnosis:**
    [2 sentences with specific numbers from tools]

    **Options:**
    A. [Specific option with numbers]
    B. [Alternative with numbers]
    C. [Third option if applicable]

    **Timeline:**
    [1-2 sentences with specific months/dates]

    **Next Steps:**
    - [Actionable step 1]
    - [Actionable step 2]
    - [Actionable step 3]
    </response>

    Use SPECIFIC NUMBERS from tool outputs. Be direct and actionable."""

        # Build user prompt
        tool_context = self._format_tool_outputs(tool_outputs)
        
        user_prompt = f"""USER QUESTION:
    {user_query}

    EXTRACTED DATA:
    {json.dumps(extracted_data, indent=2)}

    TOOL RESULTS:
    {tool_context}

    Generate a response using the exact format above. Use specific numbers from tool results."""

        # Call LLM with retry logic
        max_retries = 2
        for attempt in range(max_retries):
            response = self._call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=600,
                temperature=0.7
            )
            
            # ✅ CHECK: Is response JSON?
            if self._is_json_response(response):
                print(f"   ⚠️  Attempt {attempt+1}: LLM returned JSON, retrying with stronger instruction...")
                
                # Add VERY strong anti-JSON instruction
                system_prompt = system_prompt.replace(
                    "CRITICAL OUTPUT RULES:",
                    "CRITICAL OUTPUT RULES (STRICT ENFORCEMENT):\n- JSON OUTPUT IS STRICTLY FORBIDDEN\n- DO NOT COPY TOOL OUTPUTS"
                )
                continue
            
            # Valid response, break
            break
        
        # Extract response from <response> tag
        response_match = re.search(r'<response>(.*?)</response>', response, re.DOTALL)
        if response_match:
            final_response = response_match.group(1).strip()
        else:
            # Fallback: use everything after <thinking>
            thinking_end = response.find('</thinking>')
            if thinking_end != -1:
                final_response = response[thinking_end + len('</thinking>'):].strip()
            else:
                final_response = response
        
        # Clean up
        final_response = re.sub(r'<[^>]+>', '', final_response)
        final_response = re.sub(r'\n{3,}', '\n\n', final_response)
        
        # ✅ FINAL CHECK: If still JSON, convert to text
        if self._is_json_response(final_response):
            print("   🔧 Converting JSON response to natural language...")
            final_response = self._convert_json_to_text(final_response, user_query, tool_outputs)
        
        return final_response.strip()


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


    def _convert_json_to_text(self, json_response: str, user_query: str, tool_outputs: Dict) -> str:
        """Emergency fallback: Convert JSON to natural language"""
        
        # Extract key metrics from tool outputs
        success_prob = tool_outputs.get('predict_success_label', {}).get('success_probability', 0)
        survival_months = tool_outputs.get('predict_survival_months', {}).get('survival_months', 0)
        revenue_predicted = tool_outputs.get('predict_revenue_estimate', {}).get('predicted_revenue', 0)
        breakeven_months = tool_outputs.get('predict_break_even_time', {}).get('breakeven_months', 0)
        
        # Generate simple response
        response = f"""**Diagnosis:**
    Your startup shows a {success_prob*100:.0f}% success probability based on current metrics. With {survival_months} months of predicted survival, you need to focus on accelerating growth immediately.

    **Options:**
    A. **Cost-Cutting Path**: Reduce burn rate by 30% to extend runway while maintaining growth investments
    B. **Revenue Focus**: Prioritize customer acquisition to reach ${revenue_predicted:,.0f} predicted revenue potential
    C. **Hybrid Approach**: Balance cost reduction with strategic growth spending

    **Timeline:**
    Expected to reach breakeven in {breakeven_months} months if current trajectory continues. Critical checkpoints at 3, 6, and 9 months.

    **Next Steps:**
    - Calculate exact runway with current burn rate
    - Identify top 3 customer acquisition channels
    - Set monthly growth targets (users and revenue)
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


    def _clean_response(self, response: str, strict_limit: bool = True) -> str:
        """Clean response (optionally enforce word limit)"""
        
        # Remove XML tags
        response = re.sub(r'<[^>]+>', '', response)
        
        # Remove excessive whitespace
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        # Only enforce word limit if strict
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
    
    def _clean_response(self, response: str) -> str:
        """Clean response"""
        response = re.sub(r'<[^>]+>', '', response)
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        words = response.split()
        if len(words) > 150:
            response = ' '.join(words[:150]) + '...'
        
        return response.strip()
    
    def reset_conversation(self):
        """Clear history"""
        self.conversation_history = []

def create_orchestrator():
    return ProdIQOrchestrator()
