from pydantic import BaseModel
from typing import List, Optional

# Recommend POST
class RecommendRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    user_history: Optional[str] = None
    k: int = 5

class ProductResult(BaseModel):
    title: str
    average_rating: Optional[float] = None
    price: Optional[float] = None 
    similarity_score: float
    ml_score: Optional[float] = None 

class RecommendResponse(BaseModel):
    query: str
    user_id: Optional[str] = None 
    top_pick: str
    explanation: str
    reason: str
    retrieved_products: list[ProductResult]

# Similar POST

class SimilarRequest(BaseModel):
    product_title: str
    k: int = 10

class SimilarResponse(BaseModel):
    query_title: str
    similar_products: List[ProductResult]

# Ask POST

class AskRequest(BaseModel):
    question: str
    k: int = 5

class Citation(BaseModel):
    claim: str
    source_product: str

class AskResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]

# Expalin POST

class ExplainRequest(BaseModel):
    user_id: str
    product_title: str


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_contribution: float


class ExplainResponse(BaseModel):
    product_title: str
    user_id: str
    predicted_rating: float
    base_value: float
    feature_contributions: List[FeatureContribution]

# A/B testing POST

class ABTestRequest(BaseModel):
    queries: List[str] = ["moisturizer for dry skin", "shampoo for curly hair", "face serum"]
    n_users: int = 10

class ABTestResponse(BaseModel):
    strategy_a_mean_score: float
    strategy_b_mean_score: float
    uplift: float
    uplift_pct: float
    t_statistic: float
    p_value: float
    significant: bool
    n_samples: int

# Evaluate POST

class EvaluateRequest(BaseModel):
    question: str
    ground_truth: Optional[str] = None
    k: int = 5

class EvaluateResponse(BaseModel):
    question: str
    answer: str
    faithfulness: float
    answer_relevancy: float
    context_recall: float