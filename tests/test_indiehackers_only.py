# tests/test_indiehackers_only.py
#!/usr/bin/env python3
"""Test with IndiHackers only - has real revenue data"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.hybrid_search import HybridSearch

def test_indiehackers_validation():
    print("\n" + "="*60)
    print("TEST: IndiHackers Revenue Validation")
    print("="*60)
    
    hs = HybridSearch()
    
    query = "SaaS productivity tool"
    assumed_revenue = 25000
    
    # Search ONLY IndiHackers
    similar = hs.find_similar_products(
        query_text=query,
        n_results=20,
        source='indiehackers_cleaned'
    )
    
    print(f"\n✅ Found {len(similar)} similar products from IndiHackers")
    
    # Extract revenue
    revenues = []
    for product in similar:
        revenue = product.get('revenue_monthly')
        if revenue:
            try:
                revenue_val = float(revenue)
                if 10 < revenue_val < 10000000:
                    revenues.append(revenue_val)
                    print(f"   - {product.get('name', 'Unknown')[:40]}: ${revenue_val:,.0f}/mo")
            except:
                pass
    
    if revenues:
        revenues.sort()
        n = len(revenues)
        
        print(f"\n📊 Revenue Statistics ({n} products):")
        print(f"   Median: ${revenues[n//2]:,.0f}")
        print(f"   Average: ${sum(revenues)/n:,.0f}")
        print(f"   Range: ${revenues[0]:,.0f} - ${revenues[-1]:,.0f}")
        
        print(f"\n🎯 Your assumption: ${assumed_revenue:,.0f}")
        
        deviation = ((assumed_revenue - revenues[n//2]) / revenues[n//2]) * 100
        print(f"   Deviation from median: {deviation:+.1f}%")
        
        if assumed_revenue < revenues[n//4]:
            print(f"   Verdict: CONSERVATIVE")
        elif assumed_revenue <= revenues[3*n//4]:
            print(f"   Verdict: REALISTIC ✅")
        else:
            print(f"   Verdict: OPTIMISTIC")
    else:
        print("\n⚠️  No revenue data found")

if __name__ == '__main__':
    test_indiehackers_validation()
