startup-analyzer/
├── data/                                  
│   ├── raw/
│   │   └── propulse.csv
│   └── processed/
│       ├── df_without_meta.csv
│       └── df_encoded.csv
│
├── ml_core/                               
│   ├── artifacts/
│   │   ├── category_benchmarks.json
│   │   ├── competitor_dominance.json
│   │   ├── desc_tfidf.pkl
│   │   ├── sf_tfidf.pkl
│   │   ├── fr_tfidf.pkl
│   │   ├── sr_tfidf.pkl
│   │   └── label_encoders/
│   │       ├── country_encoder.pkl
│   │       └── category_type_encoder.pkl
│   │
│   ├── config.py
│   ├── feature_engine.py
│   ├── preprocessing.py
│   └── inference.py
│
├── models/                                 
│   ├── revenue_estimated_model/
│   ├── survival_month_model/
│   ├── breakeven_time_model/
│   ├── traction_time_model/
│   └── success_label_model/
│
├── mcp_server/                           
│   ├── __init__.py
│   ├── server.py                          
│   ├── tools/                            
│   │   ├── __init__.py
│   │   ├── predict_revenue.py
│   │   ├── predict_survival.py
│   │   ├── predict_breakeven.py
│   │   ├── predict_traction.py
│   │   ├── predict_success.py
│   │   ├── story_weaver.py              
│   │   └── market_scout.py               
│   │
│   ├── schemas/                           
│   │   ├── startup_input.json
│   │   └── prediction_output.json
│   │
│   └── config.json                        
│
├── api/                                  
│   ├── app.py                            
│   └── routes.py
│
├── scripts/
│   ├── generate_artifacts.py
│   └── test_mcp_server.py                
│
├── llm/                                   
│   ├── prompts/
│   │   ├── system_prompt.txt             
│   │   ├── extraction_prompt.txt        
│   │   └── explanation_prompt.txt       
│   │
│   ├── client.py                         
│   └── config.py                         
│
├── tests/
│   ├── test_feature_engine.py
│   ├── test_inference.py
│   └── test_mcp_tools.py
│
├── requirements.txt
├── mcp_requirements.txt                  
├── README.md
└── .env.example                          

