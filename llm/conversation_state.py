#!/usr/bin/env python3
"""
Conversation State Manager
Tracks conversation history, context, and turn types
"""
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime
import json


TurnType = Literal['initial', 'follow_up', 'clarification', 'data_update', 'benchmark_query']


@dataclass
class ConversationTurn:
    """Single turn in conversation"""
    turn_number: int
    timestamp: str
    user_query: str
    turn_type: TurnType
    extracted_data: Dict
    tools_called: List[str]
    tool_outputs: Dict
    response: str


class ConversationStateManager:
    """
    Manages conversation state across multiple turns
    Accumulates context and provides it to orchestrator
    """
    
    def __init__(self):
        self.turns: List[ConversationTurn] = []
        self.accumulated_data: Dict = {}  # All extracted data merged
        self.tool_outputs_history: Dict = {}  # All tool outputs by turn
        self.conversation_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_turn(self, user_query: str, turn_type: TurnType, 
                 extracted_data: Dict, tools_called: List[str],
                 tool_outputs: Dict, response: str):
        """Add a new turn to history"""
        
        turn = ConversationTurn(
            turn_number=len(self.turns) + 1,
            timestamp=datetime.now().isoformat(),
            user_query=user_query,
            turn_type=turn_type,
            extracted_data=extracted_data,
            tools_called=tools_called,
            tool_outputs=tool_outputs,
            response=response
        )
        
        self.turns.append(turn)
        
        # Update accumulated data (smart merge)
        self._merge_accumulated_data(extracted_data)
        
        # Store tool outputs
        self.tool_outputs_history[f"turn_{turn.turn_number}"] = tool_outputs
    
    def _merge_accumulated_data(self, new_data: Dict):
        """Smart merge: only update if value is meaningful"""
        for key, value in new_data.items():
            # Skip null/empty/default values
            if value in [None, 0, '', 'saas', 'general_advice']:
                continue
            
            # Update if:
            # 1. Key doesn't exist
            # 2. New value is more specific
            # 3. It's a genuine update (scenario, question_type, etc.)
            if key not in self.accumulated_data or key in ['scenario', 'question_type', 'topic']:
                self.accumulated_data[key] = value
    
    def get_turn_type(self, user_query: str) -> TurnType:
        """Classify turn type based on query and history"""
        
        # First message
        if len(self.turns) == 0:
            return 'initial'
        
        query_lower = user_query.lower()
        
        # Clarification keywords
        if any(word in query_lower for word in [
            'what do you mean', 'can you explain', 'why did you say',
            'what does that mean', 'how did you calculate', 'where did'
        ]):
            return 'clarification'
        
        # Data update keywords
        if any(word in query_lower for word in [
            'actually', 'correction', 'sorry', 'i meant', 'let me correct',
            'update:', 'change that to', 'it\'s actually'
        ]):
            return 'data_update'
        
        # Benchmark/comparison query
        if any(word in query_lower for word in [
            'typical', 'average', 'what do others', 'compared to',
            'industry standard', 'benchmark', 'similar companies'
        ]):
            return 'benchmark_query'
        
        # Default: follow-up
        return 'follow_up'
    
    def get_context_summary(self, last_n_turns: int = 2) -> str:
        """Get summary of recent conversation for prompt"""
        
        if not self.turns:
            return ""
        
        recent_turns = self.turns[-last_n_turns:]
        
        summary = "PREVIOUS CONVERSATION CONTEXT:\n"
        summary += "="*60 + "\n"
        
        for turn in recent_turns:
            summary += f"\nTurn {turn.turn_number} ({turn.turn_type}):\n"
            summary += f"User: {turn.user_query}\n"
            summary += f"Tools used: {', '.join(turn.tools_called)}\n"
            summary += f"Key metrics: {self._format_key_metrics(turn.extracted_data)}\n"
            summary += f"Your response (first 150 chars): {turn.response[:150]}...\n"
        
        summary += "\n" + "="*60 + "\n"
        summary += f"ACCUMULATED DATA (all turns):\n{json.dumps(self.accumulated_data, indent=2)}\n"
        summary += "="*60 + "\n\n"
        
        return summary
    
    def _format_key_metrics(self, data: Dict) -> str:
        """Format key metrics for display"""
        key_fields = ['current_revenue', 'target_revenue', 'runway_months', 
                     'burn_rate_monthly_est', 'growth_rate_monthly']
        
        metrics = {k: v for k, v in data.items() if k in key_fields and v}
        return json.dumps(metrics) if metrics else "None"
    
    def get_tool_output(self, tool_name: str, turn_number: Optional[int] = None) -> Optional[Dict]:
        """Get tool output from specific turn or most recent"""
        
        if turn_number:
            return self.tool_outputs_history.get(f"turn_{turn_number}", {}).get(tool_name)
        
        # Search backwards for most recent
        for turn in reversed(self.turns):
            if tool_name in turn.tool_outputs:
                return turn.tool_outputs[tool_name]
        
        return None
    
    def is_first_turn(self) -> bool:
        """Check if this is the first turn"""
        return len(self.turns) == 0
    
    def reset(self):
        """Reset conversation state"""
        self.turns = []
        self.accumulated_data = {}
        self.tool_outputs_history = {}
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def export_history(self) -> Dict:
        """Export conversation history for frontend"""
        return {
            'conversation_id': self.conversation_id,
            'total_turns': len(self.turns),
            'turns': [
                {
                    'turn_number': turn.turn_number,
                    'timestamp': turn.timestamp,
                    'user_query': turn.user_query,
                    'turn_type': turn.turn_type,
                    'tools_called': turn.tools_called,
                    'response': turn.response
                }
                for turn in self.turns
            ],
            'accumulated_data': self.accumulated_data
        }
