import sys
import os
import time
from fastapi import FastAPI, HTTPException

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from backend.models import RecommendRequest, RecommendResponse, ProductResult, SimilarRequest, SimilarResponse, AskRequest, AskResponse, Citation
from backend.models import ExplainRequest, ExplainResponse, FeatureContribution, ABTestRequest, ABTestResponse, EvaluateRequest, EvaluateResponse
from backend.monitoring import log_request, get_metrics
from rag.evaluator import rag_recommend, retrieve_products, rag_ask, run_ab_test, evaluate_single_query, data as evaluator_df
from rag.explainer import explain_single_prediction, xgb_model, FEATURE_COLS, data as explainer_data

app = FastAPI(title = "Amazon Recommendation API", description= " RAG + LLM powered product recommendation system", version = "1.0.0")

@app.get("/")
def root():
    return { "message": "Amazon Recommendation API is running"}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    """
    RAG pipeline: query → retrieve → LLM explain → structured response
    """
    start_time = time.time()
    try: 
        result = rag_recommend( 
            user_query= request.query,
            user_id=request.user_id,
            user_history= request.user_history,
            k= request.k )
        latency_ms = (time.time() - start_time) * 1000
        log_request("recommend", latency_ms)
        return result
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))
    
@app.post("/similar", response_model=SimilarResponse)
def similar(request: SimilarRequest):
    """
    Find products semantically similar to a given product title.
    Pure FAISS lookup — no LLM, no re-ranking.
    """
    start_time = time.time()
    try:
        results = retrieve_products(request.product_title, k=request.k)
        results["ml_score"] = None  # not applicable for this endpoint
        latency_ms = (time.time() - start_time) * 1000
        log_request("similar", latency_ms)
        return {
            "query_title": request.product_title,
            "similar_products": results[["title", "average_rating", "price", "similarity_score", "ml_score"]].to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Citation-grounded natural language Q&A.
    Every claim in the answer is traced to a specific source product.
    """
    start_time = time.time()
    try:
        result = rag_ask(request.question, k=request.k)
        latency_ms = (time.time() - start_time) * 1000
        log_request("ask", latency_ms)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest):
    """
    SHAP explanation for one specific user-product prediction.
    """
    start_time = time.time()
    try:
        result = explain_single_prediction(
            user_id=request.user_id,
            product_title=request.product_title,
            data=explainer_data,
            xgb_model=xgb_model,
            FEATURE_COLS=FEATURE_COLS
        )
        latency_ms = (time.time() - start_time) * 1000
        log_request("explain", latency_ms)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/ab_test", response_model=ABTestResponse)
def ab_test(request: ABTestRequest):
    """
    A/B test: FAISS-only ranking vs FAISS + XGBoost re-ranking.
    Returns uplift and statistical significance (p-value).
    """
    start_time = time.time()
    try:
        # Sample real user_ids from processed data
        sample_user_ids = evaluator_df["user_id"].drop_duplicates().sample(
            n=min(request.n_users, evaluator_df["user_id"].nunique()),
            random_state=42
        ).tolist()

        result = run_ab_test(request.queries, sample_user_ids, k=5)
        latency_ms = (time.time() - start_time) * 1000
        log_request("ab_test", latency_ms)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_endpoint(request: EvaluateRequest):
    """
    Run RAGAS evaluation on a single query in real-time.
    Slower than other endpoints (5-10s) — diagnostic tool, not high-traffic use.
    """
    start_time = time.time()
    try:
        result = evaluate_single_query(
            question=request.question,
            ground_truth=request.ground_truth,
            k=request.k
        )
        latency_ms = (time.time() - start_time) * 1000
        log_request("evaluate", latency_ms)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/metrics")
def metrics():
    """
    Live aggregated stats: total requests, avg latency, requests by endpoint.
    In-memory tracking — resets when server restarts.
    """
    return get_metrics()