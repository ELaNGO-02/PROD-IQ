# scripts/check_actual_metadata.py
#!/usr/bin/env python3
"""
Check what's ACTUALLY in your ChromaDB metadata
"""
import chromadb
from chromadb.config import Settings
import json

client = chromadb.PersistentClient(
    path="database/chroma_db_storage",
    settings=Settings(anonymized_telemetry=False)
)

collections_to_check = [
    'indiehackers_cleaned',
    'appstore_cleaned',
    'playstore_cleaned'
]

for coll_name in collections_to_check:
    try:
        collection = client.get_collection(coll_name)
        sample = collection.get(limit=3, include=["metadatas", "documents"])
        
        print(f"\n{'='*80}")
        print(f"📦 {coll_name}")
        print(f"{'='*80}")
        
        if sample['metadatas']:
            for i, metadata in enumerate(sample['metadatas'][:3], 1):
                print(f"\nSample {i}:")
                print(json.dumps(metadata, indent=2, default=str))
                
                if sample['documents'] and len(sample['documents']) > i-1:
                    print(f"\nDocument preview:")
                    print(f"{sample['documents'][i-1][:200]}...")
        
    except Exception as e:
        print(f"\n❌ Error with {coll_name}: {e}")

print("\n" + "="*80)
