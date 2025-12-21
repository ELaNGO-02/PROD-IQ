def normalize_features(raw: dict) -> dict:
    """
    Normalize extracted LLM output into safe ML-ready input.
    This function GUARANTEES no None values for ML-critical fields.
    """

    normalized = raw.copy()

    # ---- Numeric defaults ----
    normalized["team_size"] = int(raw.get("team_size") or 2)
    normalized["num_founders"] = int(raw.get("num_founders") or min(normalized["team_size"], 2))
    normalized["price"] = float(raw.get("price") or 0.0)
    normalized["total_funding"] = float(raw.get("total_funding") or 0.0)

    # Burn rate inference (VERY IMPORTANT)
    if raw.get("burn_rate_monthly") is None:
        normalized["burn_rate_monthly"] = normalized["team_size"] * 5000 + 2000
    else:
        normalized["burn_rate_monthly"] = float(raw["burn_rate_monthly"])

    # ---- Strings ----
    normalized["main_category"] = raw.get("main_category") or "other"
    normalized["business_model"] = raw.get("business_model") or "subscription"
    normalized["country"] = raw.get("country") or "unknown"
    normalized["description"] = raw.get("description") or "A startup product"

    return normalized
