#!/usr/bin/env python3
"""
Machine Learning Inference Engine - FIXED FEATURE SELECTION
Loads trained models and runs predictions using proper feature alignment
"""
import sys
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

# Feature engineering
from ml_core.feature_engine import engineer_features
from ml_core.input_adapter import adapt_mcp_input


class InferenceEngine:
    """Handles loading models and running predictions with proper feature engineering"""
    
    def __init__(self):
        """Initialize and load all trained models"""
        self.models = {}
        self.feature_names = {}  # Store expected features per model
        
        # ✅ CORRECTED: Models are in project root, not ml_core/artifacts
        self.project_root = Path(__file__).parent.parent
        self.model_path = self.project_root / "models"
        
        print("\n" + "="*60)
        print("🔧 INITIALIZING INFERENCE ENGINE")
        print("="*60)
        print(f"📂 Project root: {self.project_root}")
        print(f"📂 Model path: {self.model_path}")
        print(f"📂 Model path exists: {self.model_path.exists()}")
        
        self.load_models()
    
    def load_models(self):
        """Load all trained models with detailed logging"""
        
        model_configs = {
            'revenue': {
                'path': 'revenue_estimated_model/re_v1/cat_reg.pkl',
                'features_path': 'revenue_estimated_model/re_v1/feature_names.pkl',
                'name': 'Revenue Prediction Model'
            },
            'success': {
                'path': 'success_label_model/sl_v1/xgb_final.pkl',
                'features_path': 'success_label_model/sl_v1/feature_names.pkl',
                'name': 'Success Prediction Model'
            },
            'breakeven': {
                'path': 'breakeven_time_model/bet_v1/breakeven_time_best_stack_model.pkl',
                'features_path': 'breakeven_time_model/bet_v1/feature_names.pkl',
                'name': 'Breakeven Time Model'
            },
            'survival': {
                'path': 'survival_month_model/sm_v1/survival_month_best_stack_model.pkl',
                'features_path': 'survival_month_model/sm_v1/feature_names.pkl',
                'name': 'Survival Month Model'
            },
            'traction': {
                'path': 'traction_time_model/tt_v1/traction_time_best_stack_model.pkl',
                'features_path': 'traction_time_model/tt_v1/feature_names.pkl',
                'name': 'Traction Time Model'
            }
        }
        
        print("\n📦 LOADING MODELS:")
        print("-" * 60)
        
        for key, config in model_configs.items():
            model_file = self.model_path / config['path']
            features_file = self.model_path / config.get('features_path', '')
            
            try:
                if model_file.exists():
                    # Load model
                    try:
                        self.models[key] = joblib.load(model_file)
                        print(f"✅ {config['name']}")
                        print(f"   📄 File: {config['path']}")
                        print(f"   📊 Type: {type(self.models[key]).__name__}")
                        
                        # Load feature names if available
                        if features_file.exists():
                            self.feature_names[key] = joblib.load(features_file)
                            print(f"   🔧 Features loaded: {len(self.feature_names[key])}")
                        else:
                            print(f"   ⚠️  No feature_names.pkl found (will use all numeric features)")
                            self.feature_names[key] = None
                    except:
                        # Fallback to pickle
                        with open(model_file, 'rb') as f:
                            self.models[key] = pickle.load(f)
                        print(f"✅ {config['name']}")
                        print(f"   📄 File: {config['path']}")
                        print(f"   📊 Type: {type(self.models[key]).__name__}")
                        
                        if features_file.exists():
                            self.feature_names[key] = joblib.load(features_file)
                            print(f"   🔧 Features loaded: {len(self.feature_names[key])}")
                else:
                    print(f"❌ {config['name']}")
                    print(f"   📄 Expected: {model_file}")
                    print(f"   ⚠️  File not found!")
            
            except Exception as e:
                print(f"❌ {config['name']}")
                print(f"   📄 File: {config['path']}")
                print(f"   ⚠️  Error: {str(e)[:100]}")
        
        print("-" * 60)
        print(f"✅ LOADED: {len(self.models)}/5 models")
        print("="*60 + "\n")
        
        if len(self.models) == 0:
            print("⚠️  WARNING: No models loaded! Using fallback heuristics.", file=sys.stderr)
    
    def _prepare_features(self, features: pd.DataFrame, model_key: str) -> pd.DataFrame:
        """
        Prepare features for model prediction by:
        1. Removing string/object columns
        2. Selecting only features expected by model
        3. Filling missing features with 0
        """
        # Step 1: Remove object/string columns (keep only numeric)
        numeric_features = features.select_dtypes(include=[np.number]).copy()
        
        print(f"   📊 Features after numeric filter: {numeric_features.shape[1]} columns")
        
        # Step 2: If we have expected feature names, align to them
        if model_key in self.feature_names and self.feature_names[model_key] is not None:
            expected_features = self.feature_names[model_key]
            
            # Add missing columns with 0
            for col in expected_features:
                if col not in numeric_features.columns:
                    numeric_features[col] = 0
            
            # Select only expected features in correct order
            numeric_features = numeric_features[expected_features]
            
            print(f"   📊 Features after alignment: {numeric_features.shape[1]} columns")
        
        return numeric_features
    
    # ================================================================
    # PREDICTION METHODS
    # ================================================================
    
    def predict_success(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict startup success probability"""
        print("\n🔮 PREDICT_SUCCESS CALLED")
        print(f"   Input data keys: {list(data.keys())}")
        
        try:
            # 1. Adapt input
            print("   🔄 Step 1: Adapting input...")
            adapted = adapt_mcp_input(data)
            print(f"      ✅ Adapted keys: {list(adapted.keys())[:10]}...")
            
            # 2. Engineer features
            print("   🔄 Step 2: Engineering features...")
            features = engineer_features(adapted)
            print(f"      ✅ Features shape: {features.shape}")
            
            # 3. Prepare features (remove strings, align to model)
            print("   🔄 Step 3: Preparing features for model...")
            model_features = self._prepare_features(features, 'success')
            
            # 4. Model inference
            if 'success' in self.models:
                print("   🔄 Step 4: Running ML model...")
                model = self.models['success']
                
                # Get probability
                proba = model.predict_proba(model_features)
                prob = float(proba[0][1])
                
                print(f"      ✅ ML Prediction: {prob:.3f}")
                print(f"      📊 Model type: {type(model).__name__}")
                
                return {
                    "probability": round(prob, 3),
                    "category": "High Potential" if prob > 0.7 else "Moderate Potential",
                    "confidence": 0.80,
                    "source": "ml_model"
                }
            else:
                print("   ⚠️  Step 4: ML model not loaded, using heuristic...")
                
                # Fallback heuristic
                has_revenue = data.get('has_revenue', False)
                funding = data.get('funding_raised', 0)
                
                if funding > 1000000 and has_revenue:
                    prob = 0.75
                elif funding > 100000:
                    prob = 0.60
                else:
                    prob = 0.45
                
                print(f"      ✅ Heuristic prediction: {prob:.3f}")
                
                return {
                    "probability": prob,
                    "category": "Moderate Potential",
                    "confidence": 0.50,
                    "source": "heuristic"
                }
        
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def predict_revenue(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict startup revenue"""
        print("\n💰 PREDICT_REVENUE CALLED")
        
        try:
            adapted = adapt_mcp_input(data)
            features = engineer_features(adapted)
            model_features = self._prepare_features(features, 'revenue')
            
            if 'revenue' in self.models:
                print("   ✅ Using ML model")
                model = self.models['revenue']
                prediction = model.predict(model_features)
                revenue = float(prediction[0])
                
                print(f"   📊 Predicted revenue: ${revenue:,.2f}")
                
                return {
                    "revenue": revenue,
                    "confidence": 0.75,
                    "revenue_low": revenue * 0.7,
                    "revenue_high": revenue * 1.5,
                    "source": "ml_model"
                }
            else:
                print("   ⚠️  Using heuristic")
                base_revenue = data.get("funding_raised", 0) * 0.2
                
                return {
                    "revenue": base_revenue,
                    "confidence": 0.50,
                    "revenue_low": base_revenue * 0.7,
                    "revenue_high": base_revenue * 1.5,
                    "source": "heuristic"
                }
        
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return {"error": str(e)}
    
    def predict_breakeven(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict breakeven timeline"""
        print("\n⚖️  PREDICT_BREAKEVEN CALLED")
        
        try:
            adapted = adapt_mcp_input(data)
            features = engineer_features(adapted)
            model_features = self._prepare_features(features, 'breakeven')
            
            if 'breakeven' in self.models:
                print("   ✅ Using ML model")
                model = self.models['breakeven']
                prediction = model.predict(model_features)
                months = int(max(prediction[0], 1))   # At least 1 month
                months = min(months, 120)
                
                print(f"   📊 Predicted breakeven: {months} months")
                
                return {
                    "months": months,
                    "confidence": 0.70,
                    "source": "ml_model"
                }
            else:
                print("   ⚠️  Using heuristic")
                funding = data.get("funding_raised", 0)
                burn_rate = data.get("monthly_burn_rate", 10000)
                revenue = data.get("monthly_revenue", 0)
                
                monthly_gap = burn_rate - revenue
                months = int(abs(funding / monthly_gap)) if monthly_gap != 0 else 12
                
                return {
                    "months": months,
                    "confidence": 0.50,
                    "source": "heuristic"
                }
        
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return {"error": str(e)}
    
    def predict_survival(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict survival probability"""
        print("\n🛡️ PREDICT_SURVIVAL CALLED")
        
        try:
            adapted = adapt_mcp_input(data)
            features = engineer_features(adapted)
            model_features = self._prepare_features(features, 'survival')
            
            if 'survival' in self.models:
                print("   ✅ Using ML model")
                model = self.models['survival']
                prediction = model.predict(model_features)
                survival_months = int(max(prediction[0], 0))
                survival_months = min(survival_months, 120)
                
                print(f"   📊 Predicted survival: {survival_months} months")
                
                return {
                    "survival_months": survival_months,
                    "prob_12m": 0.85 if survival_months >= 12 else 0.60,
                    "risk_level": "Low" if survival_months >= 24 else "High",
                    "source": "ml_model"
                }
            else:
                print("   ⚠️  Using heuristic")
                runway = data.get("runway_months", 12)
                has_revenue = data.get("has_revenue", False)
                
                survival_months = int(runway * 1.5) if has_revenue else int(runway)
                
                return {
                    "survival_months": survival_months,
                    "prob_12m": 0.70,
                    "risk_level": "Medium",
                    "source": "heuristic"
                }
        
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return {"error": str(e)}
    
    def predict_traction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict time to traction"""
        print("\n📈 PREDICT_TRACTION CALLED")
        
        try:
            adapted = adapt_mcp_input(data)
            features = engineer_features(adapted)
            model_features = self._prepare_features(features, 'traction')
            
            if 'traction' in self.models:
                print("   ✅ Using ML model")
                model = self.models['traction']
                prediction = model.predict(model_features)
                months = int(prediction[0])
                
                print(f"   📊 Predicted traction: {months} months")
                
                return {
                    "months": months,
                    "confidence": 0.65,
                    "source": "ml_model"
                }
            else:
                print("   ⚠️  Using heuristic")
                founding_year = data.get("founding_year", 2023)
                age = 2025 - founding_year
                months = max(6, 18 - age * 3)
                
                return {
                    "months": int(months),
                    "confidence": 0.50,
                    "source": "heuristic"
                }
        
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return {"error": str(e)}
