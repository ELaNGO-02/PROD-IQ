# tests/test_text_based_search.py
#!/usr/bin/env python3
"""Test text-based vector search"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.vector_search_text_based import VectorSearchTextBased

def test_basic_search():
    print("\n" + "="*60)
    print("TEST: Text-Based Semantic Search")
    print("="*60)
    
    vs = VectorSearchTextBased()
    
    query = "SaaS tool for project management"
    
    results = vs.find_similar_products(
        query_text=query,
        n_results=3,
        source="indiehackers_cleaned"
    )
    
    print(f"\n✅ Query: {query}")
    
    for source, data in results.items():
        if data['count'] > 0:
            print(f"\n📦 {source}: Found {data['count']} products")
            
            for i, product in enumerate(data['results'], 1):
                print(f"\n{i}. Similarity: {product['similarity_score']}")
                print(f"   Document preview:")
                print(f"   {product['document'][:250]}...")

if __name__ == '__main__':
    test_basic_search()
