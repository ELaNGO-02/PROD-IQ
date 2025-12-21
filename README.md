# Prod-IQ: Vertical AI Agent for Pre-Seed Startup Validation

![Status](https://img.shields.io/badge/Status-Proprietary-red)
![Version](https://img.shields.io/badge/BetaVersion-0.0.0-blue)
![Stack](https://img.shields.io/badge/Tech-Llama3%20%7C%20MCP%20%7C%20PyTorch-green)


> **“Data-Driven Foresight for the Pre-Seed Stage.”**

Prod‑IQ is a **vertical AI agent for early-stage founders**. It behaves like a **startup doctor**: a chat-based consultant that analyzes your idea, numbers, and context, then returns **hard, model-backed answers** about revenue, survival, traction, and break-even—backed by a proprietary dataset of ~42.5k real products and 5 specialized ML models.

Unlike generic LLMs that “sound smart but hallucinate numbers”, Prod‑IQ **never trusts the LLM for math**. The LLM is only the *controller*; deterministic ML models and benchmark databases actually compute the numbers.

***

## 📖 Executive Summary

As of 2025, over **11,300 funded startups have shut down in a single year**, with thousands more in India alone by October. Many fail not because the founders are lazy, but because:

- They **don’t know if their numbers are realistic**.
- Financial modeling is **too complex or too expensive**.
- They get advice that is **vague (“focus on product‑market fit”) and not quantitative**.

**Prod‑IQ** is designed to be:

- A **pre-seed validation copilot** that sits between “idea” and “first institutional cheque”.
- A **compound AI system** that:
    - Accepts **natural language** from founders.
    - Converts it into **387 structured features**.
    - Runs **5 honest ML models** trained on **42,500+ real products**.
    - Cross‑validates with **SQL benchmarks** and **RAG over 8 CSV knowledge sources**.
    - Responds in **clear, narrative, human‑friendly language**, but grounded in numbers.

The MVP is a **beta demonstrator** (like an early ChatGPT-style launch), not the full ERP‑scale product you plan for later.

***

## 🎯 MVP Scope (What This Version Does)

The current MVP focuses on:

1. **Pre‑seed / early‑stage startup analysis**, not full ERP.
2. **5 core questions**, each backed by a dedicated model:
    - Will my startup likely succeed?
    - What revenue can I realistically expect?
    - How long can this survive?
    - When might I break even?
    - How fast can I reach initial traction?
3. **Session-based, no accounts yet**:
    - Context preserved within a chat session.
    - MVP is for testing with friendly users / recruiters / mentors.
4. **LLM-Orchestrated, Tool-Driven**:
    - Llama‑3.2B (4‑bit quantized) as the brain.
    - MCP server as the tool gateway.
    - ML models + DBs + RAG as the “muscles”.

The future **ERP model** (multi-tenant, full analytics suite, alerting, internal team workspace, etc.) is explicitly **out of scope for this MVP**.

***

## 🧠 Core Concept: Controller–Worker Compound AI

Prod‑IQ is designed as a **Compound AI System** with a clear separation of responsibilities:

### 1. Controller – Llama‑3 Agent

**Folder:** `llm/`
**Key files:** `orchestrator.py`, `router.py`, `extractor.py`, `mcp_bridge.py`, `mcp_client.py`, `local_inference.py`

**Role:**
The Controller is the **reasoning engine and traffic cop**. It:

- Reads chat messages:
    - Example:
*“I’m building a subscription fitness app for Indian Tier‑2 cities, 3 founders, 6 team members, thinking of ₹499/month. What revenue and survival odds?”*
- Extracts structured fields via prompt‑guided extraction + examples:
    - `main_category`, `product_type`, `team_size`, `price`, `country`, `funding`, etc.
- Decides which tools to call:
    - One or several of the 5 ML models (via MCP tools).
    - SQL benchmark queries.
    - Vector DB search (Chroma).
    - Special tools like **StoryWeaver** (narrative scenarios) and **Competitor Analysis**.
- Cross-validates:
    - Compares model predictions against:
        - Category averages (from SQL benchmarks).
        - Similar products (from Chroma-based RAG).
    - Adjusts messaging if something looks unrealistic (“this is outside typical range, here’s why…”).

**Important:** The Controller **never does the numeric prediction itself**. It delegates to the ML core and DBs, then interprets.

***

### 2. Proprietary Data Engine \& Feature Reconstruction

**Folders:**

- `data/raw/`, `data/processed/`
- `ml_core/` (feature pipeline + artifacts)

**Dataset:**

- ~**42,500 real product/startup rows** across multiple verticals:
    - Digital apps, SaaS, consumables, food, B2B, etc.
- ~**387 engineered features**, including:
    - Raw: `price`, `rating_avg`, `review_count`, `downloads`, `active_users`, `team_size`, `total_funding`, `launch_date`, etc.
    - Derived:
        - `survival_score`, `growth_score`, `market_fit_score`, `risk_score`.
        - Category benchmarks: `category_success_rate`, `category_avg_price`, `category_product_count`.
        - Financial metrics: `burn_rate_monthly_est`, `estimated_runway_months`, `funding_per_employee`, `monthly_profit_est`.
        - Quality/engagement: `engagement_score`, `rating_weighted`, `reviews_per_month`, `downloads_per_month`.
    - Flags and encodings:
        - `is_bootstrapped`, `is_well_funded`, `is_crowded_market`, `is_highly_rated`.
        - Encoders: `category_encoded`, `product_type_encoded`, `business_model_encoded`, `source_encoded`.

**Feature Reconstruction Challenge:**
Users give you ~10–20 inputs in messy language. Your pipeline must reconstruct a **387-feature vector** with:

- Direct mappings (e.g., `price`, `team_size`).
- Derived fields (e.g., `burn_rate_monthly_est`, `estimated_runway_months`).
- Category benchmarks (from `category_benchmarks.json`).
- Text features (TF‑IDF for description, success/failure reasons).
- Encodings \& flags.
- Safe defaults (no information leakage from training).

**Implementation (in `ml_core/`):**

- `input_adapter.py`:
    - Maps extracted fields from LLM into a base row.
- `preprocessing.py` / `feature_engine.py` / `normalizer.py` / `pipeline.py`:
    - Derives time features (`age_months`, `is_young`, `is_established`).
    - Derives funding \& team features:
        - `burn_rate_monthly_est = team_size * k + c`
        - `estimated_runway_months = total_funding / burn_rate`
    - Merges category-level stats from:
        - `category_benchmarks.json`
        - `competitor_dominance.json`
    - Builds traction metrics:
        - `downloads_per_month`, `reviews_per_month`, `upvotes_per_month`.
    - Text embeddings:
        - `desc_tfidf.pkl`, `sf_tfidf.pkl`, `fr_tfidf.pkl`, `sr_tfidf.pkl`.
    - Encodings:
        - Label encoders in `artifacts/label_encoders/`.
    - Fills missing columns with 0 or median; sets flags like `has_brand`, `has_website_url`, `info_completeness`.

**Role:**
This engine **turns partial user reality into the exact feature space** the models were trained on, without cheating or leaking labels.

***

### 3. Expert Worker Nodes – The 5 ML Models

**Folder:** `models/` and `ml_core/inference.py`

Each core question is backed by a dedicated model (often a stack of CatBoost + XGBoost, with separate configs and feature lists):


| Model | Target | Train Shape | Notes |
| :-- | :-- | :-- | :-- |
| **Revenue Estimator** | `revenue_estimated` | (35674, 180) | CatBoost + XGBoost, R² ≈ 0.63 |
| **Survival Predictor** | `target_survival_months` | (35674, 207) | Stack model + XGB + Cat |
| **Break-Even Time** | `target_breakeven_time` | (34612, 188) | Stack model ensemble |
| **Traction Time** | `target_traction_time` | (35520, 191) | Months to reach key traction |
| **Success Classifier** | `success_label` (−1/0/1) | (3708, 15) | 73% accuracy after leakage removal |

**Example questions they answer:**

- **Success Predictor:**
“Will my startup succeed or fail?”
→ Success probability + key risk factors.
- **Revenue Estimator:**
“What revenue can I expect?”
→ Monthly revenue estimate ± confidence.
- **Survival Predictor:**
“How long will this last with my burn and funding?”
→ Expected survival months.
- **Traction Time:**
“How fast will I get traction?” (e.g., 1,000 active users)
→ Months to traction.
- **Break-Even Time:**
“When will I be profitable?”
→ Months to break-even.

All models are trained on real, scraped data with **leakage carefully removed**, and stored with:

- Model binaries (`*.pkl`).
- `feature_names.pkl` for each model to strictly control inputs.

**Role:**
They provide **honest numeric estimates**, and are only called through a controlled inference pipeline (no direct ad-hoc queries from the LLM).

***

### 4. Knowledge \& Benchmarks: SQL + Vector DB

**SQL Benchmarks (Structured)**

**Folder:** `database/`, `scripts/load_benchmark_db.py`, `scripts/setup_mysql_db.py`

- A relational store (e.g., MySQL/Postgres) built from the master CSV used to train the 5 models.
- Provides:
    - Category averages: `category_success_rate`, `category_avg_funding`, `category_avg_revenue`.
    - Market density: `category_product_count`, `market_saturation`, `competition_level`.
    - Benchmarks for pricing, ratings, traction by category \& region.

**Vector DB (Unstructured / RAG)**

**Folder:** `database/chroma_db_storage/`, scripts: `setup_chromadb.py`, `rebuild_chromadb_with_embeddings.py`, `diagnose_chroma_collections.py`, etc.

- Uses **ChromaDB** as an embedded vector DB for:
    - ~8 CSV files of case studies, post‑mortems, success stories, strategic patterns.
- Accessed via:
    - `hybrid_search.py`, `vector_search.py`, `vector_search_text_based.py`, `unified_backend.py`, `backend_executor.py`.
- The LLM uses RAG to:
    - Retrieve similar companies.
    - Borrow phrasing and logic from historical examples.
    - Support features like **Competitor Analysis** and **StoryWeaver**.

**Role:**
These layers provide **grounding and benchmarks** for the LLM:
Predictions are not in isolation—they’re compared to similar products and markets.

***

### 5. MCP Server – Tool Interface Layer

**Folder:** `mcp_server/`

- **`server.py` + `config.json`**
    - MCP server exposing tools to the LLM.
- **Tools (`mcp_server/tools/`)**:
    - `predict_revenue.py`
    - `predict_survival.py`
    - `predict_breakeven.py`
    - `predict_traction.py`
    - `predict_success.py`
    - `journey_simulator.py`
    - `market_scout.py`
- **Schemas (`mcp_server/schemas/`)**:
    - `startup_input.json` – normalized JSON payload from the LLM.
    - `prediction_output.json` – structured result for the LLM to interpret.

**Workflow:**

1. LLM decides: “Need revenue + survival + traction.”
2. It composes a `startup_input` payload.
3. Calls tools via MCP.
4. MCP server:
    - Uses `ml_core` to reconstruct features.
    - Runs the relevant models.
    - Returns structured JSON: predictions + metadata.
5. LLM reads the JSON and responds in natural language.

**Role:**
MCP is the **formal contract** between “words world” (LLM) and “numbers world” (ML + DB).

***

### 6. API \& Frontend Integration

**API:** `api/app.py`, `api/routes.py`

- Exposes HTTP endpoints:
    - `POST /chat` – main chat endpoint.
    - `POST /tools/storyweaver` – manual trigger (optional).
    - `POST /tools/competitor-analysis` – manual trigger (optional).
- Connects frontend, LLM orchestrator, and session store.

**Frontend:** (in progress)

- Chat UI with:
    - Message history (stored as JSON for context).
    - Toggle buttons (like ChatGPT’s “modes”):
        - **StoryWeaver**: scenario storytelling + visualized outcome.
        - **Competitor Analysis**: multi-company benchmark tables.
- Session‑based, no login/accounts in MVP:
    - Good enough to demo to recruiters and beta users.
    - Full auth/workspaces planned for v2.

***

### 7. MLOps \& Automation

**Planned / partial in MVP:**

- **Airflow DAGs:**
    - Weekly retraining of ML models using newly collected user data.
    - Drift detection:
        - Automated test calls with synthetic/baseline inputs.
        - Compare predictions vs training distribution.
    - Scheduled ingestion of fresh web data into:
        - Raw SQL tables.
        - Chroma collections.
- **Deployment:**
    - GCP free tier + Docker.
    - Single-node setup for demo:
        - API + LLM + MCP + DB on one instance (for MVP).

**Role:**
This MLOps layer ties everything into a **living system**, not a static offline model.

***

## 🧩 Special Tools: StoryWeaver \& Competitor Analysis

These are **vertical “modes”** on top of the core system.

### 1. StoryWeaver (Name Suggestion: **StoryWeaver** or **Scenario Studio**)

**Goal:** Explain complex metrics as a story, from multiple points of view.

**Flow:**

1. User:
> “My product is X, customers are saying Y, I plan to change strategy to Z. What will happen?”
2. LLM:
    - Extracts: current metrics + planned strategy.
    - Calls underlying models:
        - New revenue prediction.
        - New survival/traction projections.
    - Uses RAG to retrieve similar strategy changes from history.
3. Output:
    - A **story-style narrative**:
        - Founder POV: how numbers translate into runway \& stress.
        - Market POV: how the market might react.
        - Investor POV: how this strategy affects fundability.
    - Optionally references dashboard visuals (future: embed chart calls).

**Role:**
Make analytics **emotionally understandable** for non-technical founders.

***

### 2. Competitor Analysis

**Goal:** Show the founder where they sit among similar companies.

**Input:**

- Triggered as a separate tool (e.g., button in UI).
- Asks user:
    - Region: `regional / national / global`.
    - Product type: `SaaS / digital app / consumable / food / B2B / etc.`.
    - Comparison context: `valuation / revenue / product reach / customer retention`.

**Flow:**

1. LLM builds a structured query.
2. Backend uses:
    - SQL benchmarks to filter 42k products by:
        - Category, valuation band, funding stage, etc.
    - Optionally vector search for “similar narrative” companies.
3. Returns:
    - A comparison table like:

| Company | Valuation | Revenue | Team | Break-Even | Traction Time |
| :-- | :-- | :-- | :-- | :-- | :-- |
| CompanyX | \$25M | \$150K | 15 | 24 mo | 8 mo |
| CompanyY | \$18M | \$80K | 8 | 30 mo | 12 mo |
| **YOU** | ~\$22M | ~\$35K | 3 | ~26 mo | ~9 mo |

Plus insights:
    - “Your break-even time is faster than 65% of similar startups…”
    - “Your team is leaner than most; consider raising \$X to match growth patterns…”

**Role:**
Ground the founder in **relative reality**, not just absolute numbers.

***

## 🔎 End-to-End Request Flow (MVP)

1. **User types in chat:**
    - “We’re building an organic kids clothing brand, 2-person team, ₹4.8L burn, 180 orders/month at ₹1,899. How bad is this, and how fast can we fix it?”
2. **Frontend sends to API**
`POST /chat` with session ID and message history.
3. **API → LLM Orchestrator**
    - LLM:
        - Extracts:
            - `main_category="kids_products"`, `price=1899`, `team_size=2`, `burn_rate=480000`, etc.
        - Calls MCP tools:
            - `predict_revenue`, `predict_survival`, `predict_traction`, `predict_breakeven`, `predict_success`.
        - Calls DB/RAG:
            - Fetch similar “organic kids clothing” products.
            - Fetch category benchmarks.
4. **MCP Tools → ML Core + Databases**
    - `ml_core` builds 387-feature vector.
    - Models run and return predictions.
    - SQL + Chroma provide benchmarks and similar cases.
5. **MCP → LLM**
    - Receives structured JSON with:
        - Predictions.
        - Benchmarks.
        - Metadata (confidence, where the sample lies in distribution).
6. **LLM crafts answer**
    - Explains:
        - Profitability issues.
        - Runway \& survival risk.
        - Strategy suggestions (e.g., change AOV, switch to subscription).
    - Optionally uses:
        - **StoryWeaver** for narrative.
        - **Competitor Analysis** to show comparative table.
7. **API returns to frontend**
    - Chat UI shows narrative + numbers + optional table/visuals.

***

## 🛠 Technology Stack (MVP)

- **LLM \& Orchestration**
    - Llama‑3.2B (4‑bit quantized, local).
    - Custom prompt set:
        - `prompts/global_prompt.txt`
        - `prompts/guardrails.txt`
        - Task-specific prompts (predict_revenue, survival, traction, etc.).
- **ML Core**
    - CatBoost, XGBoost, stacking pipelines.
    - Artifacts in `ml_core/artifacts/`.
- **Vector DB**
    - ChromaDB for RAG over CSV knowledge.
- **Relational DB**
    - MySQL/Postgres for benchmarks and master product table.
- **Backend**
    - Python, FastAPI.
- **MCP**
    - Custom MCP server in `mcp_server/` with tools for each analytical task.
- **MLOps**
    - Airflow (planned/in-progress) for retraining \& drift scripts.
- **Infra**
    - GCP free tier + Docker for hosting.

***

## 🔒 License \& Usage

**Prod-IQ – Proprietary License**

© 2025 PROD-IQ. All rights reserved.

Authors:
- Hariharasudhan M
- Elango T

For licensing or collaboration inquiries, contact:
- Email: mghariharasudhan@gmail.com, elangothangarasan@gmail.com
- LinkedIn: https://linkedin.com/in/hariharasudhan-mg
           https://www.linkedin.com/in/elango-t/


Permission is hereby granted to view this repository for the sole purpose of
evaluation (e.g., hiring, technical review, or due diligence).

You may NOT, without prior written consent from the owner:
- copy, distribute, or publish this code or any part of it;
- use this project or its models in any commercial product or service;
- use the datasets, feature engineering logic, or model artifacts to train
  competing systems;
- remove or alter any proprietary notices or attribution.

This software is provided “as is”, without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose, and noninfringement. In no event shall the
author be liable for any claim, damages, or other liability, whether in an
action of contract, tort, or otherwise, arising from, out of, or in connection
with the software or its use.


> *Built with precision in love.*

***
