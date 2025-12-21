# tests/validate_hybrid_search.py
#!/usr/bin/env python3
"""
Validate Hybrid Search Results
Check if semantic similarity is actually working
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.hybrid_search import HybridSearch
import mysql.connector

def check_revenue_distribution():
    """Check if all products have same revenue"""
    print("\n" + "="*60)
    print("VALIDATION 1: Revenue Distribution")
    print("="*60)
    
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='June#12345',
        database='prod-iq_db'
    )
    cursor = conn.cursor()
    
    # Check revenue distribution
    cursor.execute("""
        SELECT 
            data_source,
            COUNT(*) as total,
            COUNT(DISTINCT revenue_monthly) as unique_revenues,
            AVG(revenue_monthly) as avg_revenue,
            MIN(revenue_monthly) as min_revenue,
            MAX(revenue_monthly) as max_revenue
        FROM master_products_features
        WHERE revenue_monthly IS NOT NULL AND revenue_monthly > 0
        GROUP BY data_source
    """)
    
    results = cursor.fetchall()
    
    print("\n📊 Revenue Distribution by Source:")
    for row in results:
        source, total, unique, avg, min_val, max_val = row
        print(f"\n   {source}:")
        print(f"      Total products: {total}")
        print(f"      Unique revenue values: {unique}")
        print(f"      Avg: ${avg}, Min: ${min_val}, Max: ${max_val}")
        
        if unique == 1:
            print(f"      ⚠️  WARNING: All products have same revenue!")
    
    cursor.close()
    conn.close()

def validate_semantic_similarity():
    """Check if similar products are actually similar"""
    print("\n" + "="*60)
    print("VALIDATION 2: Semantic Similarity Quality")
    print("="*60)
    
    hs = HybridSearch()
    
    # Test cases
    test_cases = [
        {
            'query': "SaaS tool for project management",
            'expected_keywords': ['saas', 'project', 'management', 'team', 'collaboration', 'productivity'],
            'source': 'indiehackers_cleaned'
        },
        {
            'query': "mobile fitness tracking app",
            'expected_keywords': ['fitness', 'health', 'workout', 'exercise', 'track'],
            'source': 'appstore_cleaned'
        },
        {
            'query': "food delivery service",
            'expected_keywords': ['food', 'delivery', 'restaurant', 'order', 'meal'],
            'source': 'playstore_cleaned'
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {test['query']}")
        print(f"{'='*60}")
        
        results = hs.find_similar_products(
            query_text=test['query'],
            n_results=5,
            source=test['source']
        )
        
        print(f"\nFound {len(results)} products:")
        
        relevance_scores = []
        
        for j, product in enumerate(results, 1):
            name = product.get('name', 'Unknown').lower()
            desc = product.get('description', '').lower()
            category = (product.get('main_category') or product.get('category_type', '')).lower()
            
            # Check if any expected keywords appear
            matched_keywords = [
                kw for kw in test['expected_keywords'] 
                if kw in name or kw in desc or kw in category
            ]
            
            relevance = len(matched_keywords) / len(test['expected_keywords'])
            relevance_scores.append(relevance)
            
            print(f"\n   {j}. {product.get('name', 'Unknown')}")
            print(f"      Similarity: {product.get('similarity_score', 0)}")
            print(f"      Category: {category}")
            print(f"      Matched keywords: {matched_keywords}")
            print(f"      Relevance: {relevance*100:.0f}%")
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        print(f"\n   📊 Average Relevance: {avg_relevance*100:.0f}%")
        
        if avg_relevance < 0.3:
            print(f"   ⚠️  LOW RELEVANCE - Semantic search may not be working well")
        elif avg_relevance < 0.6:
            print(f"   ⚙️  MODERATE RELEVANCE - Some improvement needed")
        else:
            print(f"   ✅ GOOD RELEVANCE - Semantic search working well")

def check_similarity_scores():
    """Check if similarity scores are reasonable"""
    print("\n" + "="*60)
    print("VALIDATION 3: Similarity Score Distribution")
    print("="*60)
    
    hs = HybridSearch()
    
    query = "SaaS productivity tool"
    
    results = hs.find_similar_products(
        query_text=query,
        n_results=20,
        source='indiehackers_cleaned'
    )
    
    if results:
        scores = [p.get('similarity_score', 0) for p in results]
        
        print(f"\n📊 Similarity Scores for: '{query}'")
        print(f"   Total results: {len(scores)}")
        print(f"   Min: {min(scores):.3f}")
        print(f"   Max: {max(scores):.3f}")
        print(f"   Average: {sum(scores)/len(scores):.3f}")
        
        negative_count = sum(1 for s in scores if s < 0)
        if negative_count > 0:
            print(f"\n   ⚠️  WARNING: {negative_count} results have negative similarity!")
            print(f"   This suggests ChromaDB distance metric issues")
        
        low_similarity = sum(1 for s in scores if 0 <= s < 0.3)
        if low_similarity > len(scores) * 0.7:
            print(f"\n   ⚠️  WARNING: {low_similarity}/{len(scores)} results have low similarity (<0.3)")
            print(f"   The embeddings may not be high quality")

if __name__ == '__main__':
    print("\n🔍 VALIDATING HYBRID SEARCH SYSTEM")
    print("="*80)
    
    # Run all validations
    check_revenue_distribution()
    validate_semantic_similarity()
    check_similarity_scores()
    
    print("\n" + "="*80)
    print("✅ VALIDATION COMPLETE")
    print("\nSummary:")
    print("- If revenue is always 18,296 → Data import issue")
    print("- If semantic relevance < 30% → ChromaDB embeddings issue")
    print("- If many negative similarities → Distance metric issue")
