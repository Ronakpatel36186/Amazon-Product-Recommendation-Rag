import time
from datetime import datetime
from collections import defaultdict


# In-memory storage — resets when server restarts, no external DB needed
request_log = []


def log_request(endpoint, latency_ms, tokens_used=0, cost_usd=0.0):
    """
    Record one API request. Called by every endpoint after it completes.
    """
    request_log.append({
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
        "latency_ms": round(latency_ms, 2),
        "tokens_used": tokens_used,
        "cost_usd": round(cost_usd, 6)
    })


def get_metrics():
    """
    Aggregate all logged requests into summary stats.
    """
    if len(request_log) == 0:
        return {
            "total_requests": 0,
            "avg_latency_ms": 0,
            "total_tokens_used": 0,
            "total_cost_usd": 0,
            "requests_by_endpoint": {}
        }

    total_requests = len(request_log)
    avg_latency = sum(r["latency_ms"] for r in request_log) / total_requests
    total_tokens = sum(r["tokens_used"] for r in request_log)
    total_cost = sum(r["cost_usd"] for r in request_log)

    # Count requests per endpoint
    by_endpoint = defaultdict(int)
    for r in request_log:
        by_endpoint[r["endpoint"]] += 1

    return {
        "total_requests": total_requests,
        "avg_latency_ms": round(avg_latency, 2),
        "total_tokens_used": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "requests_by_endpoint": dict(by_endpoint)
    }