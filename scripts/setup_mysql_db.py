import pandas as pd
from sqlalchemy import create_engine
import mysql.connector
import os

# --- Configuration ---
DB_USER = 'root'  # !! CHANGE THIS !!
DB_PASSWORD = 'June#12345' # !! CHANGE THIS !!
DB_HOST = 'localhost'
DB_NAME = 'prod-iq_db'
MASTER_CSV_PATH = 'data/raw/data.csv' # Path to your master CSV
TABLE_NAME = 'master_products_features'

# Define the database path (create it first in MySQL CLI)
engine_url = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
engine = create_engine(
    engine_url,
    pool_pre_ping=True,
    pool_recycle=3600
)

df_master = pd.read_csv(MASTER_CSV_PATH, dtype=str, low_memory=False)
df_master = df_master.fillna("0")

CHUNK_SIZE = 1000  # sweet spot for MySQL

try:
    with engine.begin() as connection:
        df_master.to_sql(
            name=TABLE_NAME,
            con=connection,
            if_exists='replace',
            index=False,
            chunksize=CHUNK_SIZE,
            method='multi'
        )

    print(f"✅ Successfully loaded {len(df_master)} rows into '{TABLE_NAME}' 💖")

except Exception as e:
    print(f"❌ MySQL load failed: {e}")
