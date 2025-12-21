# scripts/rebuild_chromadb_with_embeddings.py
#!/usr/bin/env python3
"""
Rebuild ChromaDB with PROPER embeddings
Uses sentence-transformers for semantic search
"""
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pandas as pd
import os

def rebuild_collection_with_embeddings(csv_path: str, collection_name: str, client):
    """
    Load CSV and create ChromaDB collection with proper embeddings
    """
    
    print(f"\n{'='*80}")
    print(f"📦 Processing: {collection_name}")
    print(f"   CSV: {csv_path}")
    print(f"{'='*80}")
    
    # Load CSV
    if not os.path.exists(csv_path):
        print(f"   ⚠️  File not found, skipping...")
        return
    
    df = pd.read_csv(csv_path)
    print(f"   Rows: {len(df)}")
    
    # Delete old collection
    try:
        client.delete_collection(collection_name)
        print(f"   ✅ Deleted old collection")
    except:
        pass
    
    # Create embedding function (this is the KEY!)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # Fast, good quality
    )
    
    # Create NEW collection with embedding function
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    
    print(f"   ✅ Created collection with embedding function")
    
    # Prepare data
    ids = []
    documents = []
    metadatas = []
    
    batch_size = 500
    total_added = 0
    
    for idx, row in df.iterrows():
        # Create document text (description is key for semantic search)
        if 'description' in df.columns and pd.notna(row['description']):
            doc_text = str(row['description'])
        elif 'name' in df.columns:
            doc_text = str(row['name'])
        else:
            doc_text = ' '.join([str(row[col]) for col in df.columns[:3]])
        
        # Create minimal metadata (ChromaDB has limits)
        metadata = {
            'row_index': int(idx),
            'source_file': collection_name
        }
        
        # Add name and category if available
        if 'name' in df.columns and pd.notna(row['name']):
            metadata['name'] = str(row['name'])[:500]  # Limit length
        
        if 'category_type' in df.columns and pd.notna(row['category_type']):
            metadata['category_type'] = str(row['category_type'])
        
        ids.append(f"{collection_name}_{idx}")
        documents.append(doc_text[:5000])  # Limit to 5000 chars
        metadatas.append(metadata)
        
        # Batch insert
        if len(ids) >= batch_size:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            total_added += len(ids)
            print(f"   Added {total_added} items...")
            ids, documents, metadatas = [], [], []
    
    # Insert remaining
    if ids:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        total_added += len(ids)
    
    print(f"   ✅ Collection complete: {total_added} items with embeddings")
    
    # Test a quick search
    test_results = collection.query(
        query_texts=["productivity tool"],
        n_results=3
    )
    
    if test_results['distances'] and test_results['distances'][0]:
        avg_distance = sum(test_results['distances'][0]) / len(test_results['distances'][0])
        print(f"   🧪 Test search - Avg distance: {avg_distance:.3f}")
        
        if avg_distance > 1.5:
            print(f"      ⚠️  High distances suggest poor embeddings")
        else:
            print(f"      ✅ Distances look good!")

def main():
    """Rebuild all collections with embeddings"""
    
    print("\n🚀 REBUILDING CHROMADB WITH PROPER EMBEDDINGS")
    print("="*80)
    print("\nThis will:")
    print("1. Use sentence-transformers for semantic embeddings")
    print("2. Re-create all collections from scratch")
    print("3. Enable proper semantic search")
    print("\n" + "="*80)
    
    # Create new ChromaDB client
    client = chromadb.PersistentClient(
        path="database/chroma_db_storage_new",
        settings=Settings(
            anonymized_telemetry=False
        )
    )
    
    # Map CSV files to collection names
    csv_mappings = {
        'data/raw/cleaned raw files/indiehackers_cleaned.csv': 'indiehackers_cleaned',
        'data/raw/cleaned raw files/appstore_cleaned.csv': 'appstore_cleaned',
        'data/raw/cleaned raw files/playstore_cleaned.csv': 'playstore_cleaned',
        'data/raw/cleaned raw files/failory_103_cleaned.csv': 'failory_103_cleaned',
        'data/raw/cleaned raw files/failory_global_cleaned.csv': 'failory_global_cleaned',
        'data/raw/cleaned raw files/failory_industry_cleaned.csv': 'failory_industry_cleaned',
        'data/raw/cleaned raw files/producthunt_cleaned.csv': 'producthunt_cleaned',
        'data/raw/cleaned raw files/india_unicorns_cleaned.csv': 'india_unicorns_cleaned'
    }
    
    for csv_path, collection_name in csv_mappings.items():
        try:
            rebuild_collection_with_embeddings(csv_path, collection_name, client)
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ REBUILD COMPLETE!")
    print("\nNext steps:")
    print("1. Backup old: mv database/chroma_db_storage database/chroma_db_storage_old")
    print("2. Use new: mv database/chroma_db_storage_new database/chroma_db_storage")
    print("3. Re-run tests")
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
