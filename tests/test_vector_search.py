#!/usr/bin/env python3
"""
Test Vector Search - SAFE version with None handling
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.vector_search import VectorSearch

def safe_format_number(value, default='N/A'):
    """Safely format numbers, handle None"""
    if value is None:
        return default
    try:
        return f"{float(value):,}"
    except (ValueError, TypeError):
        return default

def safe_format_currency(value, default='N/A'):
    """Safely format currency, handle None"""
    if value is None:
        return default
    try:
        return f"${float(value):,.2f}"
    except (ValueError, TypeError):
        return default

def test_collection_loading():
    """Test that all collections are loaded"""
    print("\n" + "="*60)
    print("TEST 1: Collection Loading & Schema Detection")
    print("="*60)
    
    try:
        vs = VectorSearch()
        stats = vs.get_collection_stats()
        
        print(f"\n✅ PASSED: Loaded {len(stats)} collections")
        for name, info in stats.items():
            print(f"\n   📦 {name}:")
            print(f"      Items: {info['count']}")
            print(f"      Status: {info['status']}")
            if info['available_fields']:
                print(f"      Fields ({len(info['available_fields'])}): {', '.join(info['available_fields'][:8])}...")
        
        return len(stats) > 0
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_semantic_search_indiehackers():
    """Test semantic search on Indie Hackers"""
    print("\n" + "="*60)
    print("TEST 2: Semantic Search - Indie Hackers")
    print("="*60)
    
    try:
        vs = VectorSearch()
        
        query = "SaaS product for project management and team collaboration"
        
        results = vs.find_similar_products(
            query_text=query,
            n_results=5,
            source="indiehackers_cleaned"
        )
        
        total_found = sum(data['count'] for data in results.values())
        
        print(f"\n✅ PASSED: Found {total_found} similar products")
        print(f"   Query: {query}")
        
        for source, data in results.items():
            if data['count'] > 0:
                print(f"\n   📦 {source}: {data['count']} products")
                
                for i, product in enumerate(data['results'][:3], 1):
                    name = product.get('name') or 'Unknown'
                    revenue = safe_format_currency(product.get('revenue'), 'No data')
                    users = safe_format_number(product.get('users'), 'No data')
                    similarity = product.get('similarity_score', 0)
                    
                    print(f"\n      {i}. {name}")
                    print(f"         Revenue: {revenue}/mo")
                    print(f"         Users: {users}")
                    print(f"         Similarity: {similarity}")
                    
                    # Show raw metadata for debugging
                    if product.get('raw_metadata'):
                        print(f"         Raw fields available: {list(product['raw_metadata'].keys())[:5]}")
        
        return total_found > 0
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cross_validation():
    """Test cross-validation"""
    print("\n" + "="*60)
    print("TEST 3: Cross-Validation - Revenue Assumption")
    print("="*60)
    
    try:
        vs = VectorSearch()
        
        query = "SaaS tool for productivity and time tracking"
        assumed_revenue = 25000
        
        validation = vs.cross_validate_assumption(
            query_text=query,
            assumed_revenue=assumed_revenue,
            category="productivity",
            n_results=20
        )
        
        print(f"\n✅ PASSED: Cross-validation complete")
        print(f"   Query: {query}")
        print(f"   Assumed Revenue: {safe_format_currency(assumed_revenue)}/month")
        
        if validation['verdict'] != 'insufficient_data':
            print(f"\n   📊 Market Statistics (from {validation['sample_size']} products):")
            stats = validation['market_stats']
            print(f"      Median: {safe_format_currency(stats['median_revenue'])}")
            print(f"      Average: {safe_format_currency(stats['avg_revenue'])}")
            print(f"      Range: {safe_format_currency(stats['min_revenue'])} - {safe_format_currency(stats['max_revenue'])}")
            
            print(f"\n   🎯 Verdict: {validation['verdict'].upper().replace('_', ' ')}")
            print(f"   Data sources: {', '.join(validation.get('data_sources', []))}")
        else:
            print(f"      ⚠️  {validation.get('note', 'Insufficient data')}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_app_store_search():
    """Test App Store search"""
    print("\n" + "="*60)
    print("TEST 4: App Store Search - Mobile Apps")
    print("="*60)
    
    try:
        vs = VectorSearch()
        
        query = "fitness tracking app with workout plans"
        
        results = vs.find_similar_products(
            query_text=query,
            n_results=5,
            source="appstore_cleaned"
        )
        
        print(f"\n✅ PASSED: App Store search complete")
        print(f"   Query: {query}")
        
        for source, data in results.items():
            if data['count'] > 0:
                print(f"\n   📱 Found {data['count']} apps:")
                
                for i, product in enumerate(data['results'][:3], 1):
                    name = product.get('name') or 'Unknown'
                    rating = product.get('rating')
                    reviews = safe_format_number(product.get('reviews'))
                    price = safe_format_currency(product.get('price'), 'Free')
                    similarity = product.get('similarity_score', 0)
                    
                    print(f"\n      {i}. {name}")
                    print(f"         Rating: {rating or 'N/A'}/5")
                    print(f"         Reviews: {reviews}")
                    print(f"         Price: {price}")
                    print(f"         Similarity: {similarity}")
                    
                    # Debug: show raw fields
                    if product.get('raw_metadata'):
                        print(f"         Available fields: {list(product['raw_metadata'].keys())[:5]}")
        
        return sum(data['count'] for data in results.values()) > 0
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_raw_metadata_check():
    """TEST 5: Check what's actually in the metadata"""
    print("\n" + "="*60)
    print("TEST 5: Raw Metadata Inspection")
    print("="*60)
    
    try:
        vs = VectorSearch()
        
        # Get one item from each collection
        test_collections = ['indiehackers_cleaned', 'appstore_cleaned']
        
        for coll_name in test_collections:
            if coll_name not in vs.collections:
                continue
            
            collection = vs.collections[coll_name]
            sample = collection.get(limit=1, include=["metadatas"])
            
            if sample['metadatas']:
                print(f"\n   📦 {coll_name}:")
                metadata = sample['metadatas'][0]
                print(f"      Fields: {list(metadata.keys())}")
                
                # Show a few sample values
                for key, value in list(metadata.items())[:10]:
                    print(f"        {key}: {value}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("\n🧪 RUNNING CHROMADB VECTOR SEARCH TESTS")
    print("Safe version with None handling and metadata debugging")
    
    results = []
    
    # Run all tests
    results.append(("Collection Loading", test_collection_loading()))
    results.append(("Raw Metadata Check", test_raw_metadata_check()))
    results.append(("Indie Hackers Search", test_semantic_search_indiehackers()))
    results.append(("Cross-Validation", test_cross_validation()))
    results.append(("App Store Search", test_app_store_search()))
    
    # Summary
    print("\n\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
