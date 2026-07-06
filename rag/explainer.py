import os
import numpy as np 
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import mlflow
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = pd.read_csv(os.path.join(BASE_DIR, "data", "amazon_beauty_processed.csv"))
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(os.path.join(BASE_DIR, "ml", "xgboost_model.json"))


FEATURE_COLS = [
    "user_avg_rating",
    "user_review_count",
    "user_rating_std",
    "product_avg_rating",
    "product_review_count",
    "product_rating_std",
    "review_length",
    "is_verified"
]


def explain_single_prediction(user_id, product_title, data, xgb_model, FEATURE_COLS):
    """
    Given a user_id and product_title, build the feature row,
    predict rating, and compute SHAP values for THIS specific prediction.
    """
    # Find the product in processed data
    product_rows = data[data["title_y"].str.contains(product_title, case=False, na=False)]
    
    if len(product_rows) == 0:
        raise ValueError(f"Product '{product_title}' not found in processed data")
    
    product = product_rows.iloc[0]
    
    # Find user features — cold start fallback if not found
    user_rows = data[data["user_id"] == user_id]
    if len(user_rows) == 0:
        user_avg_rating = data["user_avg_rating"].mean()
        user_review_count = data["user_review_count"].mean()
        user_rating_std = data["user_rating_std"].mean()
        is_verified = 1
    else:
        user = user_rows.iloc[0]
        user_avg_rating = user["user_avg_rating"]
        user_review_count = user["user_review_count"]
        user_rating_std = user["user_rating_std"]
        is_verified = int(user["is_verified"])
    
    # Build the single feature row
    feature_row = pd.DataFrame([{
        "user_avg_rating": user_avg_rating,
        "user_review_count": user_review_count,
        "user_rating_std": user_rating_std,
        "product_avg_rating": product["product_avg_rating"],
        "product_review_count": product["product_review_count"],
        "product_rating_std": product["product_rating_std"],
        "review_length": 200,  # neutral placeholder, same logic as Week 4
        "is_verified": is_verified
    }], columns=FEATURE_COLS)
    
    # Predict
    predicted_rating = float(xgb_model.predict(feature_row)[0])
    
    # SHAP for this one prediction
    explainer = shap.TreeExplainer(xgb_model)
    shap_values_single = explainer.shap_values(feature_row)
    
    # Build human-readable breakdown
    feature_contributions = []
    for feature, value, shap_val in zip(FEATURE_COLS, feature_row.iloc[0], shap_values_single[0]):
        feature_contributions.append({
            "feature": feature,
            "value": round(float(value), 4),
            "shap_contribution": round(float(shap_val), 4)
        })
    
    # Sort by absolute contribution — biggest impact first
    feature_contributions.sort(key=lambda x: abs(x["shap_contribution"]), reverse=True)
    
    base_value = explainer.expected_value
    if hasattr(base_value, '__len__'):
        base_value = float(base_value[0])
    else:
        base_value = float(base_value)
        
    return {
        "product_title": product["title_y"],
        "user_id": user_id,
        "predicted_rating": round(predicted_rating, 2),
        "base_value": round(base_value, 4),
        "feature_contributions": feature_contributions
    }


if __name__ == "__main__":
    X = data[FEATURE_COLS]
    y = data["rating"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    print(f"SHAP value shape: {shap_values.shape}")

    # Summary plot show feature importance across all predictions 

    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_test, feature_names= FEATURE_COLS, show = False)
    plt.tight_layout()
    plt.savefig("rag/shap_summary_plot.png", dpi = 150, bbox_inches = "tight")
    plt.close()

    print("Summary plot saved: shap_summary_plot.png")

    # Waterfall plot

    sample_idx = 0

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap.Explanation(values=shap_values[sample_idx],base_values=explainer.expected_value,data=X_test.iloc[sample_idx].values,feature_names=FEATURE_COLS)
            ,show=False)
    plt.tight_layout()
    plt.savefig("rag/shap_waterfall_plot.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Waterfall plot saved: shap_waterfall_plot.png")
    print(f"\nActual rating for this sample: {y_test.iloc[sample_idx]}")
    print(f"Predicted rating: {xgb_model.predict(X_test.iloc[[sample_idx]])[0]:.2f}")

    #let log this to mlflow

    with mlflow.start_run(run_name="shap_explainability"):
        mlflow.log_param("explainer_type", "TreeExplainer")
        mlflow.log_param("test_set_size", len(X_test))
        mlflow.log_param("features_used", FEATURE_COLS)

        # Log feature importance as metrics (mean absolute SHAP value per feature)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        for feature, importance in zip(FEATURE_COLS, mean_abs_shap):
            mlflow.log_metric(f"shap_importance_{feature}", round(float(importance), 4))

        # Log the actual plots as artifacts
        mlflow.log_artifact("rag/shap_summary_plot.png")
        mlflow.log_artifact("rag/shap_waterfall_plot.png")

    print("Logged to MLflow :)")

    # Print feature ranking for quick reference
    print("\n=== FEATURE IMPORTANCE RANKING ===")
    importance_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    print(importance_df.to_string(index=False))

    print("SHAP explainability analysis complete.")