#!/usr/bin/env python3
"""
Quick test of LLM API endpoint
"""
import sys
sys.path.insert(0, '..')

from llm.local_inference import generate

print("Testing LLM API...")
print("="*60)

prompt = """We're building an AI writing tool for content creators. Currently at $2K MRR 
        with 40 paying users at $50/month. We want to hit $10K MRR in 6 months. 
        We have $50K left and burn is $8K/month. Is this realistic? Will we survive?
        """

response = generate(prompt, max_tokens=500, temperature=0.7)

print("PROMPT:")
print(prompt)
print("\nRESPONSE:")
print(response)
print("="*60)

if len(response) > 5:
    print("✅ API working!")
else:
    print("❌ API returned empty/short response")
