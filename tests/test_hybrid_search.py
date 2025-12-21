# tests/test_hybrid_search.py
#!/usr/bin/env python3
"""Test Hybrid Search - FIXED VERSION"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.hybrid_search import HybridSearch

def safe_format(value, default='N/A'):
    """Safe format for potentially None values"""
    if value is None:
        return default
    try:
        if isinstance(value, (int, float)):
            return f"{float(value):,.0f}"
        return str(value)
    except:
        return default

def test_hybrid_search():
    print("\n" + "="*60)
    print("TEST: Hybrid Search (ChromaDB + MySQL)")
    print("="*60)
    
    hs = HybridSearch()
    
    query = "SaaS tool for team collaboration and productivity"
    
    print(f"\n✅ Query: {query}")
    
    results = hs.find_similar_products(
        query_text=query,
        n_results=5,
        source="indiehackers_cleaned"
    )
    
    print(f"\n📦 Found {len(results)} products with full data:")
    
    for i, product in enumerate(results, 1):
        print(f"\n{i}. {product.get('name', 'Unknown')}")
        print(f"   Similarity: {product.get('similarity_score', 0)}")
        
        # Try both revenue fields
        revenue = product.get('revenue_monthly') or product.get('revenue_estimated')
        print(f"   Revenue: {safe_format(revenue)}/mo")
        
        print(f"   Category: {product.get('main_category') or product.get('category_type', 'N/A')}")
        print(f"   Users: {safe_format(product.get('active_users'))}")
        print(f"   Team Size: {safe_format(product.get('team_size'))}")
        print(f"   Source: {product.get('source', 'N/A')}")
        
        desc = product.get('description', '')
        if desc:
            print(f"   Description: {desc[:100]}...")

def test_cross_validation():
    print("\n" + "="*60)
    print("TEST: Revenue Cross-Validation")
    print("="*60)
    
    hs = HybridSearch()
    
    query = "SaaS productivity tool"
    assumed_revenue = 25000
    
    validation = hs.cross_validate_revenue(
        query_text=query,
        assumed_revenue=assumed_revenue,
        n_results=20
    )
    
    print(f"\n✅ Query: {query}")
    print(f"   Assumed Revenue: {assumed_revenue:,}/mo")
    
    if validation['verdict'] != 'insufficient_data':
        print(f"\n📊 Market Stats ({validation['sample_size']} products with revenue):")
        stats = validation['market_stats']
        print(f"   Median: {stats['median']:,.0f}")
        print(f"   Average: {stats['average']:,.0f}")
        print(f"   Range: {stats['min']:,.0f} - {stats['max']:,.0f}")
        
        print(f"\n🎯 Verdict: {validation['verdict'].upper().replace('_', ' ')}")
        print(f"   Deviation from median: {validation['deviation_pct']:+.1f}%")
        
        print(f"\n📦 Top Similar Products:")
        for i, p in enumerate(validation['similar_products'][:5], 1):
            print(f"   {i}. {p['name']}")
            print(f"      Revenue: {p['revenue']:,.0f}/mo")
            print(f"      Similarity: {p['similarity']}")
            print(f"      Category: {p.get('category', 'N/A')}")
    else:
        print(f"\n⚠️  {validation['note']}")
        print(f"   Similar products found: {validation.get('similar_products_found', 0)}")

if __name__ == '__main__':
    test_hybrid_search()
    test_cross_validation()
