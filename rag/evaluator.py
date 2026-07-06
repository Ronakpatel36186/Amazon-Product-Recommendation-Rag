import os
import json
import numpy as np
import pandas as pd
import faiss
import xgboost as xgb
import mlflow
from scipy import stats
from openai import OpenAI
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ADD this after your existing imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# lets load the index and the data sample for evaluation
openai_index = faiss.read_index(os.path.join(BASE_DIR, "rag", "openai_faiss.index"))
data_sample = pd.read_csv(os.path.join(BASE_DIR, "rag", "products_sample.csv"))
data = pd.read_csv(os.path.join(BASE_DIR, "data", "amazon_beauty_processed.csv"))

xgb_model = xgb.XGBRegressor()
xgb_model.load_model(os.path.join(BASE_DIR, "ml", "xgboost_model.json"))


RERANK_FEATURE_COLS = [
    "user_avg_rating",
    "user_review_count",
    "user_rating_std",
    "product_avg_rating",
    "product_review_count",
    "product_rating_std",
    "review_length",
    "is_verified"
]

print(f"FAISS index loaded: {openai_index.ntotal} products")
print(f"Products loaded: {len(data_sample)}")
print(f"Processed data loaded: {len(data)}")

# getting all functions from previous notebook

def embed_text(text):
    """Convert text to vector using OpenAI"""
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    query_embedding = np.array(response.data[0].embedding)
    return query_embedding / np.linalg.norm(query_embedding)


def retrieve_products(query, k=10):
    """Embed query → search FAISS → return top k products"""
    query_norm = embed_text(query)
    distances, indices = openai_index.search(
        query_norm.reshape(1, -1).astype("float32"), k
    )
    # In evaluator.py retrieve_products()
    results = data_sample.iloc[indices[0]][["parent_asin", "title", "description", "average_rating", "price"]].copy()
    results["similarity_score"] = distances[0]
    return results.reset_index(drop=True)


def build_prompt(query, retrieved_products, user_history=None):
    """Format retrieved products into LLM prompt"""
    product_context = ""
    for i, row in retrieved_products.iterrows():
        price_str = f"${row['price']:.2f}" if pd.notna(row['price']) else "price not listed"
        rating_str = f"{row['average_rating']:.1f}/5" if pd.notna(row['average_rating']) else "no rating"
        product_context += f"{i+1}. {row['title']} — {rating_str}, {price_str}\n"

    history_str = user_history if user_history else "No purchase history available"

    prompt = f"""You are a helpful Amazon beauty product advisor.

User is looking for: {query}
User purchase history: {history_str}

Top matching products retrieved:
{product_context}
Based ONLY on the product names and ratings listed above, explain in 3-4 sentences 
why these products match what the user needs.
DO NOT mention any ingredients, benefits, or features not explicitly stated in 
the product names above.
Be specific — mention product names and ratings only.
Highlight the single best pick and explain why.

Return your response as JSON in exactly this format:
{{
    "explanation": "your 3-4 sentence explanation here",
    "top_pick": "exact product name here",
    "reason": "one sentence on why this is the best pick"
}}"""
    return prompt

def build_citation_prompt(question, retrieved_products):
    """
    Prompt designed to force the LLM to cite which specific product
    backs each claim it makes — not just a general explanation.
    """
    product_context = ""
    for i, row in retrieved_products.iterrows():
        rating_str = f"{row['average_rating']:.1f}/5" if pd.notna(row['average_rating']) else "no rating"
        product_context += f"PRODUCT_{i+1}: {row['title']} (Rating: {rating_str})\n"

    prompt = f"""You are a product advisor. Answer the user's question using 
ONLY the products listed below. For every claim you make, you MUST tag it 
with which PRODUCT number it came from.

User question: {question}

Available products:
{product_context}
Write a 3-4 sentence answer. Break it into individual claims, each tagged 
with its source product.

Return ONLY this JSON format:
{{
    "answer": "your full natural-sounding answer here",
    "citations": [
        {{"claim": "specific claim from the answer", "source_product": "exact product title"}},
        {{"claim": "another specific claim", "source_product": "exact product title"}}
    ]
}}"""
    return prompt

def generate_explanation(prompt):
    """Send prompt to GPT-4o-mini, return parsed JSON"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw_text = response.choices[0].message.content
    clean = raw_text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


def get_user_features(user_id, data):
    """Look up real user features. Cold start → dataset averages."""
    user_rows = data[data["user_id"] == user_id]

    if len(user_rows) == 0:
        return {
            "user_avg_rating": data["user_avg_rating"].mean(),
            "user_review_count": data["user_review_count"].mean(),
            "user_rating_std": data["user_rating_std"].mean(),
            "is_verified": 1
        }

    user = user_rows.iloc[0]
    return {
        "user_avg_rating": user["user_avg_rating"],
        "user_review_count": user["user_review_count"],
        "user_rating_std": user["user_rating_std"],
        "is_verified": int(user["is_verified"])
    }


def build_reranking_features(retrieved_products, user_features, data):
    """Combine user + product features for XGBoost scoring."""
    rows = []

    for _, product in retrieved_products.iterrows():
        product_rows = data[data["parent_asin"] == product["parent_asin"]]

        if len(product_rows) > 0:
            prod = product_rows.iloc[0]
            product_avg_rating = prod["product_avg_rating"]
            product_review_count = prod["product_review_count"]
            product_rating_std = prod["product_rating_std"]
        else:
            product_avg_rating = product["average_rating"] if pd.notna(product["average_rating"]) else data["product_avg_rating"].mean()
            product_review_count = data["product_review_count"].mean()
            product_rating_std = data["product_rating_std"].mean()

        rows.append({
            "user_avg_rating": user_features["user_avg_rating"],
            "user_review_count": user_features["user_review_count"],
            "user_rating_std": user_features["user_rating_std"],
            "product_avg_rating": product_avg_rating,
            "product_review_count": product_review_count,
            "product_rating_std": product_rating_std,
            "review_length": 200,
            "is_verified": user_features["is_verified"]
        })

    return pd.DataFrame(rows, columns=RERANK_FEATURE_COLS)


def rerank_with_xgboost(retrieved_products, user_id, data):
    """Score retrieved products with XGBoost, sort by predicted rating."""
    user_features = get_user_features(user_id, data)
    feature_matrix = build_reranking_features(retrieved_products, user_features, data)
    ml_scores = xgb_model.predict(feature_matrix)

    reranked = retrieved_products.copy().reset_index(drop=True)
    reranked["ml_score"] = ml_scores
    reranked = reranked.sort_values("ml_score", ascending=False).reset_index(drop=True)

    return reranked

# lets make A/B testing function

def run_ab_test(test_queries_list, sample_user_ids, k =5):
    """
    here we are going to compare FAISS only vs FAISS + XGBoost re-ranking strategy.
    we had already compared OpenAI vs HuggingFace embedding so no need to do that again.
    for each user + query pair get the #1 predicted rating result under both stategies and then will run a t-test.
    """
    strategy_a_score = []
    strategy_b_score = []

    for query in test_queries_list:
        for user_id in sample_user_ids:
            retrieved = retrieve_products(query, k=k)

            # Strategy A FAISS order, score the #1 result with XGBoost 
            # (Note: we need a score to compare thats why using XGBoost to score but not rerank) 
            user_features = get_user_features(user_id, data)
            feature_matrix_a = build_reranking_features(retrieved.iloc[[0]], user_features, data)
            score_a = xgb_model.predict(feature_matrix_a)[0]
            strategy_a_score.append(score_a)

            # Now startegy B rerank and than take the new #1 results score
            rerank = rerank_with_xgboost(retrieved, user_id, data)
            score_b = rerank.iloc[0]["ml_score"]
            strategy_b_score.append(score_b)

    # statistical test
    t_stat, p_value = stats.ttest_ind(strategy_a_score, strategy_b_score)
    uplift = np.mean(strategy_b_score) - np.mean(strategy_a_score)
    uplift_pct = (uplift/ np.mean(strategy_a_score)) * 100

    return {
        "strategy_a_mean_score": round(float(np.mean(strategy_a_score)), 4),
        "strategy_b_mean_score": round(float(np.mean(strategy_b_score)), 4),
        "uplift": round(float(uplift), 4),
        "uplift_pct": round(float(uplift_pct), 2),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < 0.05),
        "n_samples": len(strategy_a_score)
        }

def rag_recommend(user_query, user_id=None, user_history=None, k=5):
    """
    Full RAG pipeline with XGBoost re-ranking:
    query → retrieve → ML rerank (if user_id given) → LLM explain → JSON
    """
    retrieved = retrieve_products(user_query, k=k)

    if user_id:
        reranked = rerank_with_xgboost(retrieved, user_id, data)
    else:
        reranked = retrieved.copy()
        reranked["ml_score"] = None

    prompt = build_prompt(user_query, reranked, user_history)
    result = generate_explanation(prompt)

    result["retrieved_products"] = reranked[["title", "average_rating", "price", "similarity_score", "ml_score"]].to_dict(orient="records")
    result["query"] = user_query
    result["user_id"] = user_id

    return result

def rag_ask(question, k=5):
    """
    Citation-grounded Q&A — every claim traced to a specific product.
    """
    retrieved = retrieve_products(question, k=k)
    prompt = build_citation_prompt(question, retrieved)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content
    clean = raw.strip().replace("```json", "").replace("```", "").strip()
    result = json.loads(clean)

    result["question"] = question
    return result

def evaluate_single_query(question, ground_truth=None, k=5):
    """
    Run RAGAS evaluation on ONE query in real-time.
    Returns faithfulness, answer relevancy, context recall scores.
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    from datasets import Dataset

    # Run the pipeline for this one query
    result = rag_recommend(question, k=k)

    # Default ground truth if none provided — generic but workable
    if not ground_truth:
        ground_truth = f"A good answer should recommend relevant products that address: {question}"

    eval_data = {
        "question": [question],
        "answer": [result["explanation"]],
        "contexts": [[p["title"] for p in result["retrieved_products"]]],
        "ground_truth": [ground_truth]
    }

    dataset = Dataset.from_dict(eval_data)

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall]
    )

    return {
        "question": question,
        "answer": result["explanation"],
        "faithfulness": round(float(results["faithfulness"]), 4),
        "answer_relevancy": round(float(results["answer_relevancy"]), 4),
        "context_recall": round(float(results["context_recall"]), 4)
    }

# Running test queries with ground truth

test_queries = [
    # Skincare (7 queries)
    {
        "question": "moisturizer for dry skin",
        "ground_truth": "A good answer should recommend products with words like moisturizer, cream, or lotion labeled for dry skin in the product title."
    },
    {
        "question": "face serum with vitamin C",
        "ground_truth": "A good answer should recommend products with serum or vitamin C mentioned in the product title for brightening or skin care."
    },
    {
        "question": "sunscreen for sensitive skin",
        "ground_truth": "A good answer should recommend products with sunscreen, SPF, or sun protection mentioned in the product title suitable for sensitive skin."
    },
    {
        "question": "anti aging cream for wrinkles",
        "ground_truth": "A good answer should recommend products with anti aging, wrinkle, or retinol mentioned in the product title targeting signs of aging."
    },
    {
        "question": "gentle cleanser for oily skin",
        "ground_truth": "A good answer should recommend products with cleanser, wash, or foaming mentioned in the product title suitable for oily skin."
    },
    {
        "question": "eye cream for dark circles",
        "ground_truth": "A good answer should recommend products with eye cream, under eye, or dark circle mentioned in the product title targeting the eye area."
    },
    {
        "question": "face mask for acne prone skin",
        "ground_truth": "A good answer should recommend products with mask, clay, or acne mentioned in the product title targeting acne or pores."
    },

    # Haircare (6 queries)
    {
        "question": "shampoo for curly hair",
        "ground_truth": "A good answer should recommend products with shampoo mentioned in the product title suitable for curly or wavy hair."
    },
    {
        "question": "hair mask for damaged bleached hair",
        "ground_truth": "A good answer should recommend products with hair mask, treatment, or conditioner mentioned in the product title for damaged or bleached hair."
    },
    {
        "question": "volumizing shampoo for fine hair",
        "ground_truth": "A good answer should recommend products with shampoo or volumizing mentioned in the product title suitable for fine or thin hair."
    },
    {
        "question": "natural conditioner for frizzy hair",
        "ground_truth": "A good answer should recommend products with conditioner mentioned in the product title targeting frizzy or dry hair."
    },
    {
        "question": "scalp treatment for dandruff",
        "ground_truth": "A good answer should recommend products with scalp, dandruff, or treatment mentioned in the product title targeting scalp issues."
    },
    {
        "question": "hair growth serum for thinning hair",
        "ground_truth": "A good answer should recommend products with hair growth, serum, or thinning mentioned in the product title targeting hair loss."
    },

    # Makeup (4 queries)
    {
        "question": "long lasting foundation for oily skin",
        "ground_truth": "A good answer should recommend products with foundation mentioned in the product title suitable for oily or combination skin."
    },
    {
        "question": "waterproof mascara for sensitive eyes",
        "ground_truth": "A good answer should recommend products with mascara or waterproof mentioned in the product title suitable for sensitive eyes."
    },
    {
        "question": "natural lip balm for chapped lips",
        "ground_truth": "A good answer should recommend products with lip balm, chapstick, or lip care mentioned in the product title for dry or chapped lips."
    },
    {
        "question": "setting powder for oily skin",
        "ground_truth": "A good answer should recommend products with powder, setting, or finishing mentioned in the product title for oily skin."
    },

    # Fragrance and Body (3 queries)
    {
        "question": "natural deodorant without aluminum",
        "ground_truth": "A good answer should recommend products with deodorant or aluminum free mentioned in the product title using natural ingredients."
    },
    {
        "question": "body lotion with shea butter",
        "ground_truth": "A good answer should recommend products with lotion, body cream, or shea butter mentioned in the product title for moisturizing the body."
    },
    {
        "question": "perfume with floral scent for women",
        "ground_truth": "A good answer should recommend products with perfume, fragrance, or floral mentioned in the product title for women."
    },
]

print(f"Total test queries: {len(test_queries)}")

# running pipeline on test queries

def run_pipeline_on_test_queries(test_queries):

    evaluation_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    for i, item in enumerate(test_queries):
        print(f"Running query {i+1}/20: {item['question']}")
        try:
            # Run full RAG pipeline
            result = rag_recommend(item["question"])

            # Question user asked
            evaluation_data["question"].append(item["question"])

            # Answer by LLM explained
            evaluation_data["answer"].append(result["explanation"])

            # Contexts — the product titles FAISS retrieved
            # RAGAS uses this to check if LLM stayed grounded
            contexts = [f"{p['title']} {p.get('description', '')}" for p in result["retrieved_products"]
]
            evaluation_data["contexts"].append(contexts)

            # Ground truth a correct answer looks like
            evaluation_data["ground_truth"].append(item["ground_truth"])

        except Exception as e:
            print(f"Failed on query '{item['question']}': {e}")
            continue

    return evaluation_data

# lets create ragas evaluation 

def run_ragas_evaluation(evaluation_data):


    # Convert to HuggingFace Dataset — RAGAS requires this format
    dataset = Dataset.from_dict(evaluation_data)
    print(f"Dataset created: {len(dataset)} samples")
    print("Running RAGAS evaluation — this will take 2-3 minutes...")

    # Run evaluation — 0.1.21 picks up OPENAI_API_KEY automatically
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall]
    )

    # Extract scores
    scores = {
        "faithfulness": round(float(results["faithfulness"]), 4),
        "answer_relevancy": round(float(results["answer_relevancy"]), 4),
        "context_recall": round(float(results["context_recall"]), 4)
    }

    return scores, results


# lets log our function into MLflow

def log_to_mlflow(scores, evaluation_data):
    """
    Log RAGAS scores and evaluation details to MLflow.
    """
    with mlflow.start_run(run_name="ragas_evaluation"):
        # Log scores
        mlflow.log_metric("faithfulness", scores["faithfulness"])
        mlflow.log_metric("answer_relevancy", scores["answer_relevancy"])
        mlflow.log_metric("context_recall", scores["context_recall"])

        # Log evaluation setup
        mlflow.log_param("num_test_queries", len(evaluation_data["question"]))
        mlflow.log_param("llm_model", "gpt-4o-mini")
        mlflow.log_param("embedding_model", "text-embedding-3-small")
        mlflow.log_param("retrieval_k", 5)

        # Pass or fail against roadmap targets
        mlflow.log_param("faithfulness_target", 0.85)
        mlflow.log_param("relevancy_target", 0.88)
        mlflow.log_param("context_recall_target", 0.70)

        # Tag whether targets were hit
        mlflow.set_tag("faithfulness_passed", 
                       str(scores["faithfulness"] >= 0.85))
        mlflow.set_tag("relevancy_passed", 
                       str(scores["answer_relevancy"] >= 0.88))
        mlflow.set_tag("context_recall_passed", 
                       str(scores["context_recall"] >= 0.70))

        # Save full evaluation results as artifact
        results_df = pd.DataFrame({
            "question": evaluation_data["question"],
            "answer": evaluation_data["answer"],
            "ground_truth": evaluation_data["ground_truth"],
        })
        results_df.to_csv("ragas_results.csv", index=False)
        mlflow.log_artifact("ragas_results.csv")

    print("\n=== RAGAS SCORES ===")
    print(f"Faithfulness    : {scores['faithfulness']} (target: 0.85+)")
    print(f"Answer Relevancy: {scores['answer_relevancy']} (target: 0.88+)")
    print(f"Context Recall  : {scores['context_recall']} (target: 0.70+)")
    print("\nLogged to MLflow :)")


if __name__ == "__main__":
    print("Starting RAGAS Evaluation Pipeline")
    print("=" * 50)

    # Step 1 — Run pipeline on all 20 queries
    print("\nStep 1: Running RAG pipeline on 20 test queries...")
    evaluation_data = run_pipeline_on_test_queries(test_queries)
    print(f"Collected {len(evaluation_data['question'])} results")

    # Step 2 — Run RAGAS evaluation
    print("\nStep 2: Running RAGAS evaluation...")
    scores, full_results = run_ragas_evaluation(evaluation_data)

    # Step 3 — Log to MLflow
    print("\nStep 3: Logging to MLflow...")
    log_to_mlflow(scores, evaluation_data)

    print("\n" + "=" * 50)
    print("Evaluation complete.")