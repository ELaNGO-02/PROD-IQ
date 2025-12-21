import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import glob
import uuid

# --- Config ---
DB_PATH = "./database/chroma_db_storage"
RAW_FILES_DIR = "data/raw/cleaned raw files/"
TEXT_COLUMN_NAME = "description"
BATCH_SIZE = 200
MODEL_NAME = "all-MiniLM-L6-v2"

os.makedirs(DB_PATH, exist_ok=True)

# Init model ONCE
model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path=DB_PATH)

csv_files = glob.glob(os.path.join(RAW_FILES_DIR, "*.csv"))

for file_path in csv_files:
    collection_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n📂 Processing: {collection_name}")

    df = pd.read_csv(file_path, dtype=str)
    df = df.dropna(subset=[TEXT_COLUMN_NAME])

    collection = client.get_or_create_collection(name=collection_name)

    documents, embeddings, metadatas, ids = [], [], [], []

    for idx, row in enumerate(df.itertuples(), start=1):
        text = getattr(row, TEXT_COLUMN_NAME)

        if not isinstance(text, str) or len(text.strip()) < 10:
            continue

        documents.append(text[:1500])
        metadatas.append({
            "source_file": collection_name,
            "row_index": idx
        })
        ids.append(str(uuid.uuid4()))

        if len(documents) >= BATCH_SIZE:
            vectors = model.encode(
                documents,
                batch_size=32,
                show_progress_bar=False
            )

            collection.add(
                documents=documents,
                embeddings=vectors.tolist(),
                metadatas=metadatas,
                ids=ids
            )

            documents, metadatas, ids = [], [], []

        if idx % 500 == 0:
            print(f"  ⏳ Embedded {idx}/{len(df)} rows")

    # Final flush
    if documents:
        vectors = model.encode(
            documents,
            batch_size=32,
            show_progress_bar=False
        )
        collection.add(
            documents=documents,
            embeddings=vectors.tolist(),
            metadatas=metadatas,
            ids=ids
        )

    print(f"✅ Done: {collection.count()} vectors in '{collection_name}'")
