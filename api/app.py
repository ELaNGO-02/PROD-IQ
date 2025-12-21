# api/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.backend_executor import QueryExecutor
from database.unified_backend import UnifiedBackend
unified = UnifiedBackend()

app = Flask(__name__)
CORS(app)

# Initialize query executor
executor = QueryExecutor()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'ProductIQ API'}), 200

@app.route('/query', methods=['POST'])
def handle_query():
    """
    Handle query from LLM
    
    Expected JSON:
    {
        "intent": "benchmark",
        "params": {
            "category": "saas",
            "metric": "revenue"
        }
    }
    """
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
    """
    Unified endpoint - handles both SQL and Vector queries
    
    SQL Intent Example:
    {
        "intent": "benchmark",
        "params": {"category": "saas"}
    }
    
    Vector Intent Example:
    {
        "intent": "cross_validate",
        "params": {
            "query_text": "B2B SaaS for HR teams",
            "assumed_revenue": 450000,
            "category": "saas"
        }
    }
    """
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
    
    # Sample LLM output
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
