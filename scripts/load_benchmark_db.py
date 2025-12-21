import json
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text

# --- Configuration ---
DB_USER = 'root'  # !! CHANGE THIS !!
DB_PASSWORD = 'June#12345' # !! CHANGE THIS !!
DB_HOST = 'localhost'
DB_NAME = 'prod-iq_db'
BENCHMARK_JSON_PATH = 'ml_core/artifacts/category_benchmarks.json' 
TABLE_NAME = 'category_benchmarks'

engine_url = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
engine = create_engine(engine_url)

# --- Execution ---
with open(BENCHMARK_JSON_PATH, 'r') as f:
    benchmarks_data = json.load(f)

# Convert the dictionary format to a DataFrame suitable for SQL load
data_list = []
for category, metrics in benchmarks_data.items():
    row = {'main_category': category}
    row.update(metrics)
    data_list.append(row)

df_benchmarks = pd.DataFrame(data_list)

try:
    # Load into MySQL
    df_benchmarks.to_sql(name=TABLE_NAME, con=engine, if_exists='replace', index=False)
    
    # Optional: Add an index on the category column for fast lookups (requires Raw SQL)
    with engine.begin() as connection:
        connection.execute(
        text(
            f"CREATE INDEX idx_category ON {TABLE_NAME} (main_category(100))"
        )
    )
    
    print(f"Successfully loaded {len(df_benchmarks)} benchmarks into '{TABLE_NAME}' table.")
except Exception as e:
    print(f"An error occurred during benchmark load: {e}")

print(df_benchmarks.head())
print("Rows:", len(df_benchmarks))
print(df_benchmarks.columns)
