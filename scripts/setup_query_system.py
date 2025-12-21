
import mysql.connector
import json
import os

def setup_benchmark_table():
    """Create and populate benchmark table - FIXED VERSION"""
    
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='June#12345',
        database='prod-iq_db'
    )
    cursor = conn.cursor()
    
    # First, drop existing table if it has wrong schema
    try:
        cursor.execute("DROP TABLE IF EXISTS category_benchmarks")
        print("✅ Dropped existing table (if any)")
    except:
        pass
    
    # Create table with correct schema
    cursor.execute("""
    CREATE TABLE category_benchmarks (
        id INT AUTO_INCREMENT PRIMARY KEY,
        category VARCHAR(100) UNIQUE NOT NULL,
        success_rate DECIMAL(10,6),
        avg_price DECIMAL(15,2),
        median_price DECIMAL(15,2),
        price_std DECIMAL(15,2),
        avg_rating DECIMAL(5,3),
        median_reviews DECIMAL(15,2),
        product_count INT,
        avg_funding DECIMAL(20,2),
        avg_revenue DECIMAL(15,2),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_category (category)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)
    
    conn.commit()
    print("✅ Table 'category_benchmarks' created with correct schema")
    
    # Load and insert benchmark data
    benchmark_path = 'ml_core/artifacts/category_benchmarks.json'
    
    if not os.path.exists(benchmark_path):
        print(f"⚠️  Benchmark file not found at {benchmark_path}")
        print("   Continuing without benchmark data...")
        conn.close()
        return
    
    with open(benchmark_path, 'r') as f:
        benchmarks = json.load(f)
    
    inserted = 0
    failed = 0
    
    for category, data in benchmarks.items():
        try:
            # Use parameterized query to avoid SQL injection
            sql = """
                INSERT INTO category_benchmarks 
                (category, success_rate, avg_price, median_price, price_std, 
                 avg_rating, median_reviews, product_count, avg_funding, avg_revenue)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                success_rate = VALUES(success_rate),
                avg_price = VALUES(avg_price),
                avg_revenue = VALUES(avg_revenue),
                updated_at = CURRENT_TIMESTAMP
            """
            
            params = (
                category,
                float(data.get('category_success_rate', 0)),
                float(data.get('category_avg_price', 0)),
                float(data.get('category_median_price', 0)),
                float(data.get('category_price_std', 0)),
                float(data.get('category_avg_rating', 0)),
                float(data.get('category_median_reviews', 0)),
                int(data.get('category_product_count', 0)),
                float(data.get('category_avg_funding', 0)),
                float(data.get('category_avg_revenue', 0))
            )
            
            cursor.execute(sql, params)
            inserted += 1
            
        except Exception as e:
            print(f"⚠️  Error inserting {category}: {e}")
            failed += 1
    
    conn.commit()
    
    print(f"✅ Inserted {inserted} categories")
    if failed > 0:
        print(f"⚠️  Failed to insert {failed} categories")
    
    # Verify
    cursor.execute("SELECT COUNT(*) FROM category_benchmarks")
    count = cursor.fetchone()[0]
    print(f"✅ Total categories in database: {count}")
    
    # Show sample data
    cursor.execute("SELECT category, avg_revenue, avg_price FROM category_benchmarks LIMIT 3")
    rows = cursor.fetchall()
    print("\n📊 Sample data:")
    for row in rows:
        print(f"  {row[0]}: Revenue=${row[1]}, Price=${row[2]}")
    
    conn.close()


def verify_master_table():
    """Verify master_products_features table exists"""
    
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='June#12345',
        database='prod-iq_db'
    )
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_schema = 'prod-iq_db' 
        AND table_name = 'master_products_features'
    """)
    
    exists = cursor.fetchone()[0]
    
    if exists:
        cursor.execute("SELECT COUNT(*) FROM master_products_features")
        count = cursor.fetchone()[0]
        print(f"✅ master_products_features table exists with {count} products")
        
        # Show schema
        cursor.execute("""
            SELECT COLUMN_NAME, COLUMN_TYPE 
            FROM information_schema.COLUMNS 
            WHERE TABLE_NAME = 'master_products_features'
            AND TABLE_SCHEMA = 'prod-iq_db'
            LIMIT 5
        """)
        print("\n📋 Table columns (first 5):")
        for col_name, col_type in cursor.fetchall():
            print(f"  {col_name}: {col_type}")
    else:
        print("❌ master_products_features table NOT found!")
        print("   Run: python scripts/setup_mysql_db.py")
    
    conn.close()


if __name__ == '__main__':
    print("🚀 Setting up Query System...\n")
    
    print("Step 1: Setting up benchmark table")
    print("-" * 50)
    setup_benchmark_table()
    
    print("\n\nStep 2: Verifying master table")
    print("-" * 50)
    verify_master_table()
    
    print("\n" + "=" * 50)
    print("✅ Setup complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run tests: python tests/test_query_executor.py")
    print("2. Start API: python -m flask --app api/app run")
    print("3. Test API: curl http://localhost:5000/test")