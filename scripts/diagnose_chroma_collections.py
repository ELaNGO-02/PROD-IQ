# scripts/diagnose_chroma_collections.py
#!/usr/bin/env python3
"""
Diagnose ChromaDB collections - Check actual metadata structure
"""
import chromadb
from chromadb.config import Settings

def diagnose_collections():
    """Check what columns exist in each collection"""
    
    client = chromadb.PersistentClient(
        path="database/chroma_db_storage",
        settings=Settings(anonymized_telemetry=False)
    )
    
    collections = client.list_collections()
    
    print("🔍 CHROMADB COLLECTION DIAGNOSTICS")
    print("="*80)
    
    for collection in collections:
        print(f"\n📦 Collection: {collection.name}")
        print(f"   Total items: {collection.count()}")
        
        # Get a sample item to see metadata structure
        try:
            sample = collection.get(limit=1, include=["metadatas", "documents"])
            
            if sample['metadatas'] and len(sample['metadatas']) > 0:
                metadata = sample['metadatas'][0]
                
                print(f"   Metadata columns ({len(metadata)} fields):")
                for key, value in metadata.items():
                    value_type = type(value).__name__
                    value_preview = str(value)[:50] if value else "None"
                    print(f"      - {key}: {value_type} = {value_preview}")
                
                if sample['documents'] and len(sample['documents']) > 0:
                    doc = sample['documents'][0]
                    print(f"\n   Document preview:")
                    print(f"      {doc[:200]}...")
            else:
                print("   ⚠️  No metadata found")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*80)
    print("\n💡 Now paste the output above and I'll create the correct mapping!")

if __name__ == '__main__':
    diagnose_collections()
