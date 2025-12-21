#!/usr/bin/env python3
"""
LLM Inference via Remote API (Colab + ngrok)
Updated for structured output format
"""
import requests
import time

API_URL = "https://subdolichocephalous-notour-keeley.ngrok-free.dev/generate"

_connection_tested = False


def test_connection():
    """Test API connection"""
    global _connection_tested
    if _connection_tested:
        return

    print("[LLM] Testing connection to Colab API...")
    try:
        r = requests.post(
            API_URL,
            json={
                "system": "Health check",
                "user": "Ping"
            },
            timeout=60
        )
        r.raise_for_status()
        print("[LLM] ✅ API connection successful")
        _connection_tested = True
    except Exception as e:
        print(f"[LLM] ❌ Connection failed: {e}")
        raise


def get_llm():
    """Compatibility function - tests connection and returns True"""
    test_connection()
    return True


def generate(
    user_prompt: str,
    system_prompt: str = "You are PROD-IQ, a startup coach.",
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate response from LLM API
    
    Returns structured output with <thinking>, <extraction>, <tools>, etc.
    """
    test_connection()

    payload = {
        "system": system_prompt,
        "user": user_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    try:
        r = requests.post(API_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["response"]
    except requests.exceptions.Timeout:
        print("[LLM] ⚠️  Request timeout (120s)")
        return "Error: Request timeout"
    except Exception as e:
        print(f"[LLM] ❌ Generation error: {e}")
        return f"Error: {e}"
