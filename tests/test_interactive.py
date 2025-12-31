#!/usr/bin/env python3
"""
Interactive Chat with PROD-IQ Orchestrator
Real-time conversation in terminal
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from llm.orchestrator import ProdIQOrchestrator


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*70)
    print("🚀 PROD-IQ INTERACTIVE CHAT")
    print("="*70)
    print("💡 Tips:")
    print("   - Ask about your startup's survival, revenue goals, competitors")
    print("   - Follow up with additional questions based on previous answers")
    print("   - Type 'exit' or 'quit' to end conversation")
    print("   - Type 'reset' to start a new conversation")
    print("   - Type 'history' to see conversation history")
    print("="*70 + "\n")


def print_response(result: dict, turn_number: int):
    """Format and print orchestrator response"""
    print(f"\n{'─'*70}")
    print(f"🤖 PROD-IQ (Turn {turn_number}):")
    print(f"{'─'*70}")
    print(result['response'])
    print(f"\n{'─'*70}")
    print(f"📊 Tools Used: {', '.join(result['tools_called'])}")
    print(f"{'─'*70}\n")


def show_history(orchestrator):
    """Display conversation history"""
    if not orchestrator.conversation_history:
        print("\n📜 No conversation history yet.\n")
        return
    
    print("\n" + "="*70)
    print("📜 CONVERSATION HISTORY")
    print("="*70)
    
    for i, turn in enumerate(orchestrator.conversation_history, 1):
        print(f"\n--- Turn {i} ---")
        print(f"👤 User: {turn['user'][:100]}{'...' if len(turn['user']) > 100 else ''}")
        print(f"🤖 PROD-IQ: {turn['assistant'][:150]}{'...' if len(turn['assistant']) > 150 else ''}")
        print(f"🔧 Tools: {', '.join(turn['tools_called'])}")
    
    print("="*70 + "\n")


def main():
    """Main interactive loop"""
    
    # Initialize orchestrator
    print("\n🔄 Initializing PROD-IQ...\n")
    orchestrator = ProdIQOrchestrator()
    
    print_banner()
    
    turn_number = 0
    
    while True:
        try:
            # Get user input
            user_input = input("👤 You: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Thanks for using PROD-IQ! Good luck with your startup!\n")
                break
            
            if user_input.lower() == 'reset':
                orchestrator.reset_conversation()
                turn_number = 0
                print("\n🔄 Conversation reset. Starting fresh!\n")
                continue
            
            if user_input.lower() == 'history':
                show_history(orchestrator)
                continue
            
            if not user_input:
                print("⚠️  Please enter a question.\n")
                continue
            
            # Process query
            turn_number += 1
            print(f"\n⏳ Processing (Turn {turn_number})...")
            
            result = orchestrator.process_query(user_input)
            
            # Display response
            print_response(result, turn_number)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Type 'exit' to quit properly.\n")
            continue
        
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
