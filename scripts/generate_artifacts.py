"""
Generate all pre-computed artifacts needed for inference.
Run this ONCE after training models.

Usage:
    python scripts/generate_artifacts.py
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# ============================================================================
# CONFIGURATION
# ============================================================================

RAW_DATA_PATH = "data/raw/propulse.csv"
PROCESSED_DATA_PATH = "data/processed/without_meta/df_without_meta.csv"
ARTIFACTS_DIR = "ml_core/artifacts"
LABEL_ENCODERS_DIR = "ml_core/artifacts/label_encoders"

# Create directories if they don't exist
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(LABEL_ENCODERS_DIR, exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================

print("🚀 Loading data...")
df_raw = pd.read_csv(RAW_DATA_PATH)
df_processed = pd.read_csv(PROCESSED_DATA_PATH)

# Filter to real data only (remove synthetic)
if 'synthetic_data' in df_raw.columns:
    real_df = df_raw[df_raw['synthetic_data'] == False].copy()
    print(f"✅ Loaded {len(real_df)} real products (excluded {len(df_raw) - len(real_df)} synthetic)")
else:
    real_df = df_raw.copy()
    print(f"✅ Loaded {len(real_df)} products")

# ============================================================================
# 1. CATEGORY BENCHMARKS
# ============================================================================

print("\n📊 Creating category_benchmarks.json...")

# Aggregate by category
category_agg = real_df.groupby('main_category').agg({
    'success_label': 'mean',
    'price': ['mean', 'median', 'std'],
    'rating_avg': 'mean',
    'review_count': 'median',
    'product_id': 'count',
    'total_funding': 'mean',
    'revenue_monthly': 'mean',
}).reset_index()

# Flatten columns
category_agg.columns = [
    'main_category', 'category_success_rate', 
    'category_avg_price', 'category_median_price', 'category_price_std',
    'category_avg_rating', 'category_median_reviews', 'category_product_count',
    'category_avg_funding', 'category_avg_revenue'
]

# Convert to dict
benchmarks = {}
for _, row in category_agg.iterrows():
    category = row['main_category']
    benchmarks[category] = {
        'category_success_rate': float(row['category_success_rate']),
        'category_avg_price': float(row['category_avg_price']) if pd.notna(row['category_avg_price']) else 0.0,
        'category_median_price': float(row['category_median_price']) if pd.notna(row['category_median_price']) else 0.0,
        'category_price_std': float(row['category_price_std']) if pd.notna(row['category_price_std']) else 1.0,
        'category_avg_rating': float(row['category_avg_rating']) if pd.notna(row['category_avg_rating']) else 3.0,
        'category_median_reviews': float(row['category_median_reviews']) if pd.notna(row['category_median_reviews']) else 0.0,
        'category_product_count': int(row['category_product_count']),
        'category_avg_funding': float(row['category_avg_funding']) if pd.notna(row['category_avg_funding']) else 0.0,
        'category_avg_revenue': float(row['category_avg_revenue']) if pd.notna(row['category_avg_revenue']) else 0.0
    }

# Add default for unknown categories
benchmarks['default'] = {
    'category_success_rate': float(real_df['success_label'].mean()),
    'category_avg_price': float(real_df['price'].median()),
    'category_median_price': float(real_df['price'].median()),
    'category_price_std': float(real_df['price'].std()),
    'category_avg_rating': float(real_df['rating_avg'].mean()),
    'category_median_reviews': float(real_df['review_count'].median()),
    'category_product_count': int(len(real_df) / real_df['main_category'].nunique()),
    'category_avg_funding': float(real_df['total_funding'].mean()),
    'category_avg_revenue': float(real_df['revenue_monthly'].mean())
}

# Save
with open(f'{ARTIFACTS_DIR}/category_benchmarks.json', 'w') as f:
    json.dump(benchmarks, f, indent=2)

print(f"✅ Saved category_benchmarks.json")
print(f"   Categories: {len(benchmarks)-1}")
print(f"   Sample: {list(benchmarks.keys())[:5]}")

# ============================================================================
# 2. COMPETITOR DOMINANCE
# ============================================================================

print("\n📊 Creating competitor_dominance.json...")

# Top 5 average reviews per category
competitor_dom = {}
for category in real_df['main_category'].unique():
    cat_data = real_df[real_df['main_category'] == category]['review_count'].dropna()
    if len(cat_data) >= 5:
        competitor_dom[category] = float(cat_data.nlargest(5).mean())
    else:
        competitor_dom[category] = float(cat_data.mean()) if len(cat_data) > 0 else 0.0

# Add default
competitor_dom['default'] = float(real_df['review_count'].nlargest(5).mean())

# Save
with open(f'{ARTIFACTS_DIR}/competitor_dominance.json', 'w') as f:
    json.dump(competitor_dom, f, indent=2)

print(f"✅ Saved competitor_dominance.json")

# ============================================================================
# 3. TF-IDF VECTORIZERS (4 files)
# ============================================================================

print("\n🔤 Creating TF-IDF vectorizers...")

# TF-IDF params (same as your training)
tfidf_params = {
    'stop_words': 'english',
    'ngram_range': (1, 2),
    'min_df': 3,
    'max_df': 0.85,
}

# 3.1 Description TF-IDF
print("   - desc_tfidf.pkl...")
desc_tfidf = TfidfVectorizer(max_features=100, **tfidf_params)
desc_tfidf.fit(df_processed['description'].fillna('unknown'))
joblib.dump(desc_tfidf, f'{ARTIFACTS_DIR}/desc_tfidf.pkl')

# 3.2 Success Factors TF-IDF
print("   - sf_tfidf.pkl...")
sf_tfidf = TfidfVectorizer(max_features=50, **tfidf_params)
sf_tfidf.fit(df_processed['success_factors'].fillna('unknown'))
joblib.dump(sf_tfidf, f'{ARTIFACTS_DIR}/sf_tfidf.pkl')

# 3.3 Failure Reason TF-IDF
print("   - fr_tfidf.pkl...")
fr_tfidf = TfidfVectorizer(max_features=30, **tfidf_params)
fr_tfidf.fit(df_processed['failure_reason'].fillna('unknown'))
joblib.dump(fr_tfidf, f'{ARTIFACTS_DIR}/fr_tfidf.pkl')

# 3.4 Success Reason TF-IDF
print("   - sr_tfidf.pkl...")
sr_tfidf = TfidfVectorizer(max_features=20, **tfidf_params)
sr_tfidf.fit(df_processed['success_reason'].fillna('unknown'))
joblib.dump(sr_tfidf, f'{ARTIFACTS_DIR}/sr_tfidf.pkl')

print("✅ Saved all TF-IDF vectorizers")

# ============================================================================
# 4. LABEL ENCODERS (2 files)
# ============================================================================

print("\n🔢 Creating label encoders...")

# 4.1 Country Encoder
print("   - country_encoder.pkl...")
country_encoder = LabelEncoder()
country_encoder.fit(df_processed['country'].fillna('unknown'))
joblib.dump(country_encoder, f'{LABEL_ENCODERS_DIR}/country_encoder.pkl')
print(f"      Countries: {len(country_encoder.classes_)}")

# 4.2 Category Type Encoder
print("   - category_type_encoder.pkl...")
category_type_encoder = LabelEncoder()
category_type_encoder.fit(df_processed['category_type'].fillna('unknown'))
joblib.dump(category_type_encoder, f'{LABEL_ENCODERS_DIR}/category_type_encoder.pkl')
print(f"      Category types: {len(category_type_encoder.classes_)}")

print("✅ Saved all label encoders")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✅ ALL ARTIFACTS GENERATED SUCCESSFULLY!")
print("="*70)
print("\n📦 Created files:")
print(f"   1. {ARTIFACTS_DIR}/category_benchmarks.json")
print(f"   2. {ARTIFACTS_DIR}/competitor_dominance.json")
print(f"   3. {ARTIFACTS_DIR}/desc_tfidf.pkl")
print(f"   4. {ARTIFACTS_DIR}/sf_tfidf.pkl")
print(f"   5. {ARTIFACTS_DIR}/fr_tfidf.pkl")
print(f"   6. {ARTIFACTS_DIR}/sr_tfidf.pkl")
print(f"   7. {LABEL_ENCODERS_DIR}/country_encoder.pkl")
print(f"   8. {LABEL_ENCODERS_DIR}/category_type_encoder.pkl")
print("\n📊 Stats:")
print(f"   - Categories: {len(benchmarks)-1}")
print(f"   - Countries: {len(country_encoder.classes_)}")
print(f"   - TF-IDF features: 100 (desc) + 50 (sf) + 30 (fr) + 20 (sr) = 200")
print("\n🚀 You can now use these in feature_engine.py!")
