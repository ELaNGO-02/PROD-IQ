# api/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.backend_executor import QueryExecutor
from database.unified_backend import UnifiedBackend
from llm.orchestrator import ProdIQOrchestrator

unified = UnifiedBackend()
orchestrator = ProdIQOrchestrator()

app = Flask(__name__)
CORS(app)

executor = QueryExecutor()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'ProductIQ API'}), 200


@app.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint for frontend
    
    Expected JSON:
    {
        "user_input": "Show me my growth journey",
        "chat_history": [...],
        "session_id": "abc-123",
        "active_tool": null  // "journey_simulator" | "competitor_analysis" | null
    }
    """
    try:
        data = request.json
        
        if not data or 'user_input' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing user_input field'
            }), 400
        
        # ✅ Check for special tool activation
        active_tool = data.get('active_tool', None)
        
        if active_tool in ['journey_simulator', 'competitor_analysis']:
            # Use separate method for special tools
            result = orchestrator.process_special_tool_query(
                user_query=data['user_input'],
                tool_name=active_tool,
                chat_history=data.get('chat_history')
            )
            
            return jsonify({
                'success': True,
                'response': result['response'],
                'tools_called': [result['tool_used']],
                'tool_output': result.get('tool_output'),
                'turn_number': result['turn_number']
            }), 200
        
        
        # ✅ Normal query processing
        result = orchestrator.process_query(
            user_query=data['user_input'],
            context=data.get('context'),
            chat_history=data.get('chat_history')
        )
        
        return jsonify({
            'success': True,
            'response': result['response'],
            'tools_called': result['tools_called'],
            'turn_number': result['turn_number'],
            'conversation_history': result['conversation_history'],
            'tool_outputs': result.get('raw_tool_outputs', {})
        }), 200
    
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/query', methods=['POST'])
def handle_query():
    """Handle query from LLM"""
    try:
        llm_output = request.json
        
        if not llm_output or 'intent' not in llm_output:
            return jsonify({
                'success': False,
                'error': 'Invalid input. Expected JSON with "intent" field'
            }), 400
        
        result = executor.execute_from_llm_output(llm_output)
        
        return jsonify({
            'success': True,
            'data': result['results'],
            'metadata': {
                'template': result['template_used'],
                'params': result['params'],
                'count': result['count']
            }
        }), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/unified_query', methods=['POST'])
def unified_query():
    """Unified endpoint - handles both SQL and Vector queries"""
    try:
        llm_output = request.json
        
        if not llm_output or 'intent' not in llm_output:
            return jsonify({
                'success': False,
                'error': 'Invalid input. Expected JSON with "intent" field'
            }), 400
        
        result = unified.execute(llm_output)
        
        return jsonify(result), 200 if result['success'] else 500
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint with sample query"""
    
    test_query = {
        "intent": "benchmark",
        "params": {
            "category": "saas",
            "metric": "revenue"
        }
    }
    
    try:
        result = executor.execute_from_llm_output(test_query)
        return jsonify({
            'success': True,
            'test_query': test_query,
            'result': result
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/available-intents', methods=['GET'])
def available_intents():
    """List all available intents"""
    from database.intent_mapper import INTENT_TO_TEMPLATE
    from database.query_templates import QUERY_TEMPLATES
    
    intents = {}
    for intent, template_name in INTENT_TO_TEMPLATE.items():
        template = QUERY_TEMPLATES.get(template_name, {})
        intents[intent] = {
            'template': template_name,
            'description': template.get('description', ''),
            'required_params': template.get('params', []),
            'optional_params': template.get('optional_params', [])
        }
    
    return jsonify({
        'success': True,
        'intents': intents
    }), 200

@app.route('/generate_charts', methods=['POST'])
def generate_charts():
    """
    Extract Python code from LLM response and execute to generate charts
    Returns base64-encoded images
    """
    try:
        data = request.json
        response_text = data.get('response', '')
        
        # Extract all Python code blocks
        import re
        code_blocks = re.findall(r'``````', response_text, re.DOTALL)
        
        if not code_blocks:
            return jsonify({
                'success': False,
                'error': 'No code blocks found in response'
            })
        
        # Execute each code block and capture images
        chart_images = []
        
        for i, code in enumerate(code_blocks):
            try:
                # Create temporary directory for charts
                import tempfile
                import os
                import base64
                from io import BytesIO
                
                temp_dir = tempfile.mkdtemp()
                
                # Modify code to save in temp directory
                modified_code = code.replace(
                    "plt.savefig('", 
                    f"plt.savefig('{temp_dir}/"
                )
                modified_code = modified_code.replace("plt.show()", "")
                
                # Execute code
                exec_globals = {}
                exec(modified_code, exec_globals)
                
                # Find generated PNG files
                png_files = [f for f in os.listdir(temp_dir) if f.endswith('.png')]
                
                for png_file in png_files:
                    png_path = os.path.join(temp_dir, png_file)
                    
                    # Read image and convert to base64
                    with open(png_path, 'rb') as img_file:
                        img_data = img_file.read()
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                        
                        chart_images.append({
                            'name': png_file,
                            'data': f'data:image/png;base64,{img_base64}'
                        })
                    
                    # Clean up
                    os.remove(png_path)
                
                os.rmdir(temp_dir)
                
            except Exception as e:
                print(f"Error executing code block {i}: {e}")
                continue
        
        return jsonify({
            'success': True,
            'charts': chart_images,
            'count': len(chart_images)
        })
        
    except Exception as e:
        print(f"Chart generation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting PROD-IQ API Server...")
    print("📍 Health: http://localhost:5000/health")
    print("💬 Chat: http://localhost:5000/chat")
    app.run(debug=True, host='0.0.0.0', port=5000)
