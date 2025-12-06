startup-analyzer/
├── data/                                   # Data files
│   ├── raw/
│   │   └── propulse.csv
│   └── processed/
│       ├── df_without_meta.csv
│       └── df_encoded.csv
│
├── ml_core/                                # ML Backend
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
├── models/                                 # Trained models
│   ├── revenue_estimated_model/
│   ├── survival_month_model/
│   ├── breakeven_time_model/
│   ├── traction_time_model/
│   └── success_label_model/
│
├── mcp_server/                            # ← MCP SERVER (NEW!)
│   ├── __init__.py
│   ├── server.py                          # MCP server implementation
│   ├── tools/                             # Tool definitions
│   │   ├── __init__.py
│   │   ├── predict_revenue.py
│   │   ├── predict_survival.py
│   │   ├── predict_breakeven.py
│   │   ├── predict_traction.py
│   │   ├── predict_success.py
│   │   ├── story_weaver.py               # Story generation
│   │   └── market_scout.py               # Competitor analysis
│   │
│   ├── schemas/                           # JSON schemas for tools
│   │   ├── startup_input.json
│   │   └── prediction_output.json
│   │
│   └── config.json                        # MCP server config
│
├── api/                                   # REST API (optional, for testing)
│   ├── app.py                            # Flask/FastAPI app
│   └── routes.py
│
├── scripts/
│   ├── generate_artifacts.py
│   └── test_mcp_server.py                # Test MCP tools
│
├── llm/                                   # ← LLM INTEGRATION (NEW!)
│   ├── prompts/
│   │   ├── system_prompt.txt             # Main system prompt
│   │   ├── extraction_prompt.txt         # Data extraction
│   │   └── explanation_prompt.txt        # Result explanation
│   │
│   ├── client.py                         # LLM client wrapper
│   └── config.py                         # LLM config (model path, etc.)
│
├── tests/
│   ├── test_feature_engine.py
│   ├── test_inference.py
│   └── test_mcp_tools.py
│
├── requirements.txt
├── mcp_requirements.txt                  # MCP-specific deps
├── README.md
└── .env.example                          # Environment variables

