"""
Feature Engineering Engine
Converts raw user input (5-15 fields) into 387-feature vector.

This module:
1. Loads pre-computed artifacts (TF-IDF, encoders, benchmarks)
2. Applies the EXACT same transformations as training
3. Returns a 387-column DataFrame ready for model inference

Usage:
    from ml_core.feature_engine import FeatureEngine
    
    engine = FeatureEngine()
    features = engine.engineer_features({
        "name": "MyApp",
        "description": "An AI tool for...",
        "main_category": "saas",
        "price": 50,
        "team_size": 3
    })
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "ml_core" / "artifacts"

with open(ARTIFACTS_DIR / "category_benchmarks.json", "r") as f:
    category_benchmarks = json.load(f)

class FeatureEngine:
    """
    Feature engineering pipeline for startup prediction.
    
    Loads artifacts once at initialization, then reuses them for all requests.
    """
    
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        self.artifacts_dir = Path(artifacts_dir)
        """
        Initialize feature engine by loading all artifacts.
        
        Args:
            artifacts_dir: Path to artifacts directory
        """
        self.artifacts_dir = artifacts_dir
        
        print("🔧 Initializing Feature Engine...")
        
        # Load category benchmarks
        with open(f"{artifacts_dir}/category_benchmarks.json") as f:
            self.category_benchmarks = json.load(f)
        
        # Load competitor dominance
        with open(f"{artifacts_dir}/competitor_dominance.json") as f:
            self.competitor_dominance = json.load(f)
        
        # Load TF-IDF vectorizers
        self.desc_tfidf = joblib.load(f"{artifacts_dir}/desc_tfidf.pkl")
        self.sf_tfidf = joblib.load(f"{artifacts_dir}/sf_tfidf.pkl")
        self.fr_tfidf = joblib.load(f"{artifacts_dir}/fr_tfidf.pkl")
        self.sr_tfidf = joblib.load(f"{artifacts_dir}/sr_tfidf.pkl")
        
        # Load label encoders
        self.country_encoder = joblib.load(f"{artifacts_dir}/label_encoders/country_encoder.pkl")
        self.category_type_encoder = joblib.load(f"{artifacts_dir}/label_encoders/category_type_encoder.pkl")
        
        # Constants (from your training script)
        self.BREAK_EVEN_BENCHMARKS = {
            'saas': 21, 'finance': 30, 'business': 18, 'productivity': 18,
            'shopping': 15, 'education': 24, 'games': 12, 'health & fitness': 20,
            'food & drink': 21, 'entertainment': 18, 'social networking': 30,
            'other': 24,
        }
        
        self.TECH_KEYWORDS = [
            'ai', 'ml', 'machine learning', 'api', 'saas', 'cloud',
            'blockchain', 'analytics', 'automation', 'data', 'developer'
        ]
        
        print("✅ Feature Engine ready!")
        print(f"   - Categories: {len(self.category_benchmarks)-1}")
        print(f"   - Countries: {len(self.country_encoder.classes_)}")
        print(f"   - TF-IDF features: 200 (100+50+30+20)")
    
    def engineer_features(self, raw_input: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert raw user input into 387-feature DataFrame.
        
        Args:
            raw_input: Dictionary with user-provided fields
                Required: main_category
                Optional: name, description, price, team_size, etc.
        
        Returns:
            DataFrame with 1 row and 387 columns
        """
        print("\n🔧 FEATURE ENGINEERING STARTED")
        print(f"   Input keys: {list(raw_input.keys())}")
        # Start with empty row
        row = {}
        
        # ====================================================================
        # SECTION 1: BASE INPUTS (with defaults)
        # ====================================================================
        
        # Around line 45-50 in feature_engine.py

# ====================================================================
# SECTION 1: BASE INPUTS (with defaults)
# ====================================================================

        row['name'] = raw_input.get('name') or 'Unnamed Startup'  # ✅ FIX: Handle None
        desc = raw_input.get('description')
        if not isinstance(desc, str) or not desc or not desc.strip():  # ✅ FIX: Also check None
            desc = "A new startup product"

        row['description'] = desc
        row['main_category'] = raw_input.get('main_category') or 'other'  # ✅ FIX: Handle None

        row['product_type'] = raw_input.get('product_type', 'product')
        row['business_model'] = raw_input.get('business_model', 'subscription')
        row['price'] = float(raw_input.get('price', 0))
        row['currency'] = raw_input.get('currency', 'USD')
        row['team_size'] = int(raw_input.get('team_size', 2))
        row['num_founders'] = int(raw_input.get('num_founders', min(row['team_size'], 2)))
        row['country'] = raw_input.get('country', 'unknown')
        row['category_type'] = raw_input.get('category_type', 'unknown')
        row['total_funding'] = float(raw_input.get('total_funding', 0))
        row['funding_rounds'] = int(raw_input.get('funding_rounds', 0))
        row['num_investors'] = int(raw_input.get('num_investors', 0))
        
        # Traction metrics (default to 0 for new startups)
        row['rating_avg'] = float(raw_input.get('rating_avg', 3.0))
        row['review_count'] = int(raw_input.get('review_count', 0))
        row['sentiment_score'] = float(raw_input.get('sentiment_score', 0.5))
        row['downloads'] = int(raw_input.get('downloads', 0))
        row['upvotes'] = int(raw_input.get('upvotes', 0))
        row['active_users'] = int(raw_input.get('active_users', 0))
        row['revenue_monthly'] = float(raw_input.get('revenue_monthly', 0))
        row['content_rating'] = raw_input.get('content_rating', 'Everyone')
        
        # ====================================================================
        # SECTION 2: TIME FEATURES
        # ====================================================================
        
        # For new startups, assume just launched
        current_date = datetime.now()
        launch_date = raw_input.get("launch_date")

        # Normalize launch_date
        if launch_date is None:
            launch_date = current_date
        elif isinstance(launch_date, str):
            launch_date = pd.to_datetime(launch_date, errors="coerce")
            if pd.isna(launch_date):
                launch_date = current_date
        elif not isinstance(launch_date, (datetime, pd.Timestamp)):
            launch_date = current_date
        
        age_days = (current_date - launch_date).days
        age_months = max(age_days / 30.44, 0.1)  # Avoid division by zero
        
        row['age_days'] = age_days
        row['age_months'] = age_months
        row['age_years'] = age_days / 365.25
        row['is_young'] = int(age_months < 12)
        row['is_established'] = int(age_months >= 24)
        row['is_old'] = int(age_months > 60)
        row['is_dead'] = 0  # New startup is alive
        row['survival_months'] = age_months  # For now, same as age
        
        # ====================================================================
        # SECTION 3: TEAM & FUNDING METRICS
        # ====================================================================
        
        team_size_filled = max(row['team_size'], 1)
        row['team_size_filled'] = team_size_filled
        
        # Burn rate estimation ($5K per person + $2K overhead)
        burn_rate = (team_size_filled * 5000) + 2000
        row['burn_rate_monthly_est'] = burn_rate
        
        # Runway
        row['remaining_investment'] = row['total_funding']
        runway = row['total_funding'] / max(burn_rate, 1)
        row['estimated_runway_months'] = min(runway, 240)
        
        # Funding efficiency
        row['funding_per_employee'] = row['total_funding'] / team_size_filled
        row['funding_per_month_alive'] = row['total_funding'] / max(age_months, 1)
        row['funding_per_round'] = row['total_funding'] / max(row['funding_rounds'], 1)
        
        # Flags
        row['is_bootstrapped'] = int(row['total_funding'] == 0)
        row['is_well_funded'] = int(row['total_funding'] > 100000)
        row['investor_density'] = row['num_investors'] / max(row['funding_rounds'], 1)
        
        # ====================================================================
        # SECTION 4: CATEGORY BENCHMARKS (from JSON)
        # ====================================================================
        
        category = row['main_category']
        benchmarks = self.category_benchmarks.get(category, self.category_benchmarks['default'])
        
        row['category_success_rate'] = benchmarks['category_success_rate']
        row['category_avg_price'] = benchmarks['category_avg_price']
        row['category_median_price'] = benchmarks['category_median_price']
        row['category_price_std'] = benchmarks['category_price_std']
        row['category_avg_rating'] = benchmarks['category_avg_rating']
        row['category_median_reviews'] = benchmarks['category_median_reviews']
        row['category_product_count'] = benchmarks['category_product_count']
        row['category_avg_funding'] = benchmarks['category_avg_funding']
        row['category_avg_revenue'] = benchmarks['category_avg_revenue']
        
        # ====================================================================
        # SECTION 5: RELATIVE POSITIONING
        # ====================================================================
        
        row['price_vs_category'] = row['price'] / max(row['category_median_price'], 1)
        row['rating_vs_category'] = row['rating_avg'] / max(row['category_avg_rating'], 1)
        row['price_deviation'] = (row['price'] - row['category_avg_price']) / max(row['category_price_std'], 1)
        
        # Price tiers
        if row['price'] == 0:
            price_tier = 0
        elif row['price'] <= 10:
            price_tier = 1
        elif row['price'] <= 50:
            price_tier = 2
        elif row['price'] <= 200:
            price_tier = 3
        else:
            price_tier = 4
        
        row['price_tier_numeric'] = price_tier
        row['is_free'] = int(row['price'] == 0)
        row['is_premium'] = int(row['price'] > 100)
        
        # ====================================================================
        # SECTION 6: MARKET SATURATION
        # ====================================================================
        
        row['market_saturation'] = row['category_product_count']
        
        # Median saturation across all categories
        all_counts = [b['category_product_count'] for b in self.category_benchmarks.values() if 'category_product_count' in b]
        median_saturation = np.median(all_counts) if all_counts else 100
        row['is_crowded_market'] = int(row['market_saturation'] > median_saturation)
        
        # Competition level
        if row['market_saturation'] <= 50:
            comp_level = 0
        elif row['market_saturation'] <= 200:
            comp_level = 1
        elif row['market_saturation'] <= 500:
            comp_level = 2
        elif row['market_saturation'] <= 2000:
            comp_level = 3
        else:
            comp_level = 4
        
        row['competition_level'] = comp_level
        
        # Competitor dominance
        row['competitor_dominance'] = self.competitor_dominance.get(category, self.competitor_dominance['default'])
        
        # ====================================================================
        # SECTION 7: TRACTION & ENGAGEMENT
        # ====================================================================
        
        row['reviews_per_month'] = row['review_count'] / max(age_months, 1)
        row['downloads_per_month'] = row['downloads'] / max(age_months, 1)
        row['upvotes_per_month'] = row['upvotes'] / max(age_months, 1)
        
        # Engagement score
        row['engagement_score'] = (
            np.log1p(row['downloads']) +
            np.log1p(row['review_count']) +
            np.log1p(row['upvotes']) +
            np.log1p(row['active_users'])
        )
        
        # Rating metrics
        row['rating_weighted'] = row['rating_avg'] * np.log1p(row['review_count'])
        
        if row['review_count'] > 100:
            rating_conf = 1.0
        elif row['review_count'] > 10:
            rating_conf = 0.5
        else:
            rating_conf = 0.0
        row['rating_confidence'] = rating_conf
        
        # Momentum flags (use defaults for new startups)
        row['has_strong_reviews'] = int(row['review_count'] > 100)  # Conservative
        row['has_strong_rating'] = int(row['rating_avg'] > 4.0)
        row['has_viral_reach'] = int(row['downloads'] > 10000)  # Conservative
        
        # ====================================================================
        # SECTION 8: QUALITY & CREDIBILITY
        # ====================================================================
        
        # Presence flags
        row['has_brand'] = int(raw_input.get('brand') is not None)
        row['has_website_url'] = int(raw_input.get('website_url') is not None)
        row['has_country'] = int(row['country'] != 'unknown')
        row['has_city'] = int(raw_input.get('city') is not None)
        row['has_founders'] = int(row['num_founders'] > 0)
        row['has_marketing_channels'] = int(raw_input.get('marketing_channels') is not None)
        row['has_business_model'] = int(row['business_model'] != 'unknown')
        row['has_case_study_url'] = int(raw_input.get('case_study_url') is not None)
        
        # Info completeness
        row['info_completeness'] = (
            row['has_brand'] +
            row['has_website_url'] +
            row['has_country'] +
            row['has_founders'] +
            int(len(row['description']) > 50 if isinstance(row['description'], str) else 0)
        ) / 5
        
        # Rating tiers
        if row['rating_avg'] <= 0:
            rating_tier = 0
        elif row['rating_avg'] <= 2.5:
            rating_tier = 1
        elif row['rating_avg'] <= 3.5:
            rating_tier = 2
        elif row['rating_avg'] <= 4.0:
            rating_tier = 3
        else:
            rating_tier = 4
        
        row['rating_tier'] = rating_tier
        row['is_highly_rated'] = int(row['rating_avg'] >= 4.0)
        row['is_top_rated'] = int(row['rating_avg'] >= 4.5)
        
        # Text metrics
        row['description_length'] = len(row['description']) if row['description'] else 0
        row['description_word_count'] = len(row['description'].split()) if row['description'] else 0
        row['name_length'] = len(row['name']) if row['name'] else 0  # ← FIX THIS LINE
        row['has_long_description'] = int(row['description_length'] > 200)
        row['has_detailed_name'] = int(row['name_length'] > 20)
        
        # Tech density
        desc_lower = row['description'].lower()
        tech_count = sum(1 for kw in self.TECH_KEYWORDS if kw in desc_lower)
        row['desc_tech_density'] = tech_count
        
        # ====================================================================
        # SECTION 9: REVENUE & PROFITABILITY
        # ====================================================================
        
        row['revenue_monthly_filled'] = row['revenue_monthly']
        
        # Estimate revenue if not provided
        if row['revenue_monthly'] > 0:
            revenue_est = row['revenue_monthly']
        elif row['price'] > 0 and row['downloads'] > 0:
            # Conservative 0.5% conversion
            revenue_est = row['price'] * row['downloads'] * 0.005
        else:
            revenue_est = 0
        
        row['revenue_estimated'] = revenue_est
        
        # Profitability
        row['monthly_profit_est'] = revenue_est - burn_rate
        row['is_profitable'] = int(row['monthly_profit_est'] > 0)
        
        # Break-even calculation
        base_breakeven = self.BREAK_EVEN_BENCHMARKS.get(category, 24)
        
        if row['total_funding'] > 1000000:
            funding_adj = 1.2
        elif row['total_funding'] > 100000:
            funding_adj = 1.0
        else:
            funding_adj = 0.8
        
        if row['team_size'] > 50:
            team_adj = 1.3
        elif row['team_size'] > 10:
            team_adj = 1.1
        else:
            team_adj = 0.9
        
        if row['rating_avg'] >= 4.5:
            rating_adj = 0.8
        elif row['rating_avg'] >= 4.0:
            rating_adj = 0.9
        else:
            rating_adj = 1.1
        
        row['base_break_even_months'] = base_breakeven
        row['funding_adjustment'] = funding_adj
        row['team_adjustment'] = team_adj
        row['rating_adjustment'] = rating_adj
        
        estimated_breakeven = base_breakeven * funding_adj * team_adj * rating_adj
        row['estimated_break_even_months'] = min(max(estimated_breakeven, 6), 72)
        
        # Capital efficiency
        row['capital_efficiency'] = revenue_est / max(row['total_funding'], 1)
        row['revenue_to_burn_ratio'] = revenue_est / max(burn_rate, 1)
        
        # ====================================================================
        # SECTION 10: COMPOSITE SCORES
        # ====================================================================
        
        # Survival score
        survival_score = (
            row['rating_weighted'] * 0.2 +
            row['engagement_score'] * 0.2 +
            (row['estimated_runway_months'] / 60) * 0.2 +
            row['is_highly_rated'] * 0.15 +
            row['info_completeness'] * 0.15
        )
        row['survival_score'] = min(survival_score, 100)
        
        # Growth score
        growth_score = (
            np.log1p(row['reviews_per_month']) * 0.3 +
            np.log1p(row['downloads_per_month']) * 0.3 +
            (rating_tier / 4) * 0.2 +
            row['has_strong_reviews'] * 0.2
        )
        row['growth_score'] = min(growth_score, 100)
        
        # Market fit score
        market_fit = (
            (1 - (comp_level / 5)) * 0.3 +
            row['rating_vs_category'] * 0.3 +
            (1 - min(row['price_vs_category'], 2) / 2) * 0.2 +
            row['category_success_rate'] * 0.2
        )
        row['market_fit_score'] = min(market_fit, 100)
        
        # Risk score
        risk = (
            (1 - min(row['survival_score'] / 100, 1)) * 0.3 +
            (1 - min(row['growth_score'] / 100, 1)) * 0.2 +
            row['is_bootstrapped'] * 0.2 +
            (1 - row['is_established']) * 0.15 +
            (1 - row['info_completeness']) * 0.15
        )
        row['risk_score'] = min(risk * 100, 100)
        
        # ====================================================================
        # SECTION 11: CATEGORICAL ENCODING
        # ====================================================================
        
        # Label encoders
        try:
            row['country_encoded'] = self.country_encoder.transform([row['country']])[0]
        except:
            # Unknown country, use a default
            row['country_encoded'] = 0
        
        try:
            row['category_type_encoded'] = self.category_type_encoder.transform([row['category_type']])[0]
        except:
            row['category_type_encoded'] = 0
        
        # Factorize (simple numeric encoding)
        # Note: For inference, we just use 0 if unknown
        row['category_encoded'] = 0  # Will be handled by one-hot
        row['product_type_encoded'] = 0
        row['source_encoded'] = 0  # New startup = unknown source
        row['business_model_encoded'] = 0
        
        # ====================================================================
        # SECTION 12: INTERACTION FEATURES
        # ====================================================================
        
        row['funding_team_interaction'] = (row['total_funding'] / 1000000) * np.log1p(row['team_size'])
        row['quality_traction_interaction'] = row['rating_avg'] * np.log1p(row['review_count'])
        row['maturity_quality_interaction'] = np.log1p(age_months) * row['rating_avg']
        row['price_reach_interaction'] = row['price'] * np.log1p(row['downloads'])
        
        # ====================================================================
        # SECTION 13: TEXT EMBEDDINGS (TF-IDF)
        # ====================================================================
        
        # Description
        desc_vec = self.desc_tfidf.transform([row['description']]).toarray()[0]
        for i, val in enumerate(desc_vec):
            row[f'desc_{i}'] = val
        
        # Success factors (default to "unknown" for new startups)
        sf_text = raw_input.get('success_factors', 'unknown')
        sf_vec = self.sf_tfidf.transform([sf_text]).toarray()[0]
        for i, val in enumerate(sf_vec):
            row[f'sf_{i}'] = val
        
        # Failure reason (default to "unknown")
        fr_text = raw_input.get('failure_reason', 'unknown')
        fr_vec = self.fr_tfidf.transform([fr_text]).toarray()[0]
        for i, val in enumerate(fr_vec):
            row[f'fr_{i}'] = val
        
        # Success reason (default to "unknown")
        sr_text = raw_input.get('success_reason', 'unknown')
        sr_vec = self.sr_tfidf.transform([sr_text]).toarray()[0]
        for i, val in enumerate(sr_vec):
            row[f'sr_{i}'] = val
        
        # ====================================================================
        # SECTION 14: ONE-HOT ENCODING
        # ====================================================================
        
        # We'll create one-hot columns for categorical variables
        # For now, initialize all as 0, then set the relevant one to 1
        
        # Categories to one-hot encode (from your training)
        categories_list = list(self.category_benchmarks.keys())
        categories_list.remove('default')  # Don't create column for default
        
        for cat in categories_list:
            row[f'main_category_{cat}'] = int(row['main_category'] == cat)
        
        # Product types (common ones from training)
        product_types = ['app', 'digital_app', 'physical', 'product', 'saas', 'software', 'tool', 'unknown']
        for pt in product_types:
            row[f'product_type_{pt}'] = int(row['product_type'] == pt)
        
        # Business models
        business_models = ['marketplace', 'one-time purchase', 'one_time', 'subscription', 'unknown']
        for bm in business_models:
            row[f'business_model_{bm}'] = int(row['business_model'] == bm)
        
        # Data sources (for new startup, set to unknown or leave at 0)
        data_sources = ['failory', 'india_unicorns', 'indiehackers', 'playstore', 'producthunt', 'synthetic', 'synthetic_augmentation']
        for ds in data_sources:
            row[f'data_source_{ds}'] = 0  # New startup has no source
        
        # Exit status (new startup = Active)
        exit_statuses = ['Active', 'Bankruptcy', 'Failed', 'Operating', 'Shut Down', 'Shutdown', 'Still Active', 'Unknown', 'unknown']
        for es in exit_statuses:
            if es == 'Active':
                row[f'exit_status_{es}'] = 1
            else:
                row[f'exit_status_{es}'] = 0
        
        # Failure reasons (all 0 for new startup)
        failure_cats = ['business_model', 'competition', 'funding', 'legal', 'market_fit', 
                       'multiple', 'no_market_fit', 'other', 'ran_out_of_cash', 'team', 
                       'team_issues', 'unknown']
        for fc in failure_cats:
            row[f'failure_reason_cat_{fc}'] = 0
        
        # Success reasons (all 0 for now)
        success_cats = ['revenue', 'unknown']
        for sc in success_cats:
            row[f'success_reason_cat_{sc}'] = 0
        
        # ====================================================================
        # SECTION 15: CONVERT TO DATAFRAME
        # ====================================================================
        
        df = pd.DataFrame([row])
        
        # Fill any remaining NaN with 0
        df = df.fillna(0)
        
        print(f"   ✅ Generated {df.shape[1]} features")
        print(f"   ✅ Feature names (first 10): {list(df.columns)[:10]}")
        
        return df


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

# Global instance (loaded once)
_feature_engine = None

def get_feature_engine() -> FeatureEngine:
    """Get singleton feature engine instance."""
    global _feature_engine
    if _feature_engine is None:
        _feature_engine = FeatureEngine()
    return _feature_engine


def engineer_features(raw_input: Dict[str, Any]) -> pd.DataFrame:
    """
    Convenience function for feature engineering.
    
    Usage:
        from ml_core.feature_engine import engineer_features
        
        features = engineer_features({
            "main_category": "saas",
            "price": 50,
            "team_size": 3
        })
    """
    engine = get_feature_engine()
    return engine.engineer_features(raw_input)
