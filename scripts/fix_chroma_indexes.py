# Quick fix script - save as fix_chroma_indexes.py
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="F:\\prodIq-v2\\database\\chroma_db_storage")

EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Delete and recreate broken collections
broken_collections = ['india_unicorns_cleaned', 'producthunt_cleaned']

for name in broken_collections:
    try:
        # Get old collection
        old_col = client.get_collection(name, embedding_function=EMBEDDING_FUNCTION)
        
        # Get all data
        data = old_col.get(include=['embeddings', 'metadatas', 'documents'])
        
        print(f"📦 Backing up {name}: {len(data['ids'])} items")
        
        # Delete old
        client.delete_collection(name)
        
        # Recreate
        new_col = client.create_collection(
            name=name,
            embedding_function=EMBEDDING_FUNCTION,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Add data back
        new_col.add(
            ids=data['ids'],
            embeddings=data['embeddings'],
            metadatas=data['metadatas'],
            documents=data['documents']
        )
        
        print(f"✅ Rebuilt {name}: {new_col.count()} items")
        
    except Exception as e:
        print(f"❌ Error rebuilding {name}: {e}")

print("\n🎉 Done! Run your test again.")
