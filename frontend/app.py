import os
import streamlit as st
import requests

# Backend API URL — FastAPI must be running on this address
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Amazon Recommendation System", layout="wide")

st.title("Amazon Beauty Product Recommendation System")
st.caption("RAG + LLM Personalization | XGBoost Re-ranking | RAGAS Evaluated")

# 4 tabs
tab1, tab2, tab3, tab4 = st.tabs(["Recommender", "Product Explorer", "A/B Testing", "Monitoring"])

# ============================================================
# TAB 1 — RECOMMENDER
# ============================================================
with tab1:
    st.header("Get Personalized Recommendations")

    query = st.text_input(
        "What are you looking for?",
        placeholder="e.g. face cream for dry sensitive skin under $40"
    )

    col1, col2 = st.columns(2)
    with col1:
        user_id = st.text_input("User ID (optional)", placeholder="Leave blank for anonymous")
    with col2:
        user_history = st.text_input("Purchase history (optional)", placeholder="e.g. hydrating toner, vitamin C serum")

    if st.button("Get Recommendations", type="primary"):
        if not query:
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching products and generating explanation..."):
                try:
                    response = requests.post(
                        f"{API_URL}/recommend",
                        json={
                            "query": query,
                            "user_id": user_id if user_id else None,
                            "user_history": user_history if user_history else None,
                            "k": 5
                        }
                    )
                    response.raise_for_status()
                    result = response.json()

                    # Display top pick prominently
                    st.success(f"**Top Pick:** {result['top_pick']}")
                    st.write(f"**Why:** {result['reason']}")

                    st.subheader("Explanation")
                    st.write(result['explanation'])

                    st.subheader("All Retrieved Products")
                    for i, product in enumerate(result['retrieved_products']):
                        with st.expander(f"{i+1}. {product['title'][:80]}"):
                            col_a, col_b, col_c = st.columns(3)
                            col_a.metric("Rating", f"{product['average_rating']}/5" if product['average_rating'] else "N/A")
                            col_b.metric("Similarity", f"{product['similarity_score']:.3f}")
                            if product.get('ml_score'):
                                col_c.metric("ML Score", f"{product['ml_score']:.3f}")

                    with st.expander("Raw JSON Response"):
                        st.json(result)

                except requests.exceptions.RequestException as e:
                    st.error(f"API Error: {e}")

# ============================================================
# TAB 2 — Product Explorer
# ============================================================
with tab2:
    st.header("Product Explorer")
    st.write("Search for a product and find similar items with SHAP explanation.")

    product_search = st.text_input(
        "Enter product name",
        placeholder="e.g. Cetaphil Moisturizing Cream"
    )

    user_id_tab2 = st.text_input(
        "User ID for SHAP explanation",
        placeholder="e.g. AGKHLEW2SOWHNMFQIJGBECAF7INQ",
        key="user_id_tab2"
    )

    if st.button("Explore Product", type="primary"):
        if not product_search:
            st.warning("Please enter a product name.")
        else:
            # Section 1 — Similar products
            with st.spinner("Finding similar products..."):
                try:
                    sim_response = requests.post(
                        f"{API_URL}/similar",
                        json={"product_title": product_search, "k": 5}
                    )
                    sim_response.raise_for_status()
                    sim_result = sim_response.json()

                    st.subheader("Similar Products")
                    for i, product in enumerate(sim_result['similar_products']):
                        with st.expander(f"{i+1}. {product['title'][:80]}"):
                            col_a, col_b = st.columns(2)
                            col_a.metric("Rating", f"{product['average_rating']}/5" if product['average_rating'] else "N/A")
                            col_b.metric("Similarity", f"{product['similarity_score']:.3f}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Similar products error: {e}")

            # Section 2 — SHAP explanation (only if user_id provided)
            if user_id_tab2:
                with st.spinner("Generating SHAP explanation..."):
                    try:
                        explain_response = requests.post(
                            f"{API_URL}/explain",
                            json={
                                "user_id": user_id_tab2,
                                "product_title": product_search
                            }
                        )
                        explain_response.raise_for_status()
                        explain_result = explain_response.json()

                        st.subheader("SHAP Explanation")
                        st.write(f"**Product:** {explain_result['product_title']}")

                        col1, col2 = st.columns(2)
                        col1.metric("Predicted Rating", f"{explain_result['predicted_rating']}/5")
                        col2.metric("Base Value", f"{explain_result['base_value']}")

                        st.write("**Feature Contributions:**")
                        for fc in explain_result['feature_contributions']:
                            contribution = fc['shap_contribution']
                            color = "🟢" if contribution > 0 else "🔴"
                            st.write(f"{color} **{fc['feature']}** = {fc['value']} → {contribution:+.4f}")

                    except requests.exceptions.RequestException as e:
                        st.error(f"SHAP explanation error: {e}")
            else:
                st.info("Enter a User ID above to see SHAP explanation for this product.")

# ============================================================
# TAB 3 — A/B Testing
# ============================================================
with tab3:
    st.header("A/B Testing")
    st.write("Compare FAISS-only ranking vs FAISS + XGBoost re-ranking.")

    queries_input = st.text_area(
        "Test queries (one per line)",
        value="moisturizer for dry skin\nshampoo for curly hair\nface serum with vitamin C",
        height=120
    )

    n_users = st.slider("Number of users to test", min_value=5, max_value=50, value=10, step=5)

    if st.button("Run A/B Test", type="primary"):
        with st.spinner("Running A/B test — this may take 30-60 seconds..."):
            try:
                queries_list = [q.strip() for q in queries_input.split("\n") if q.strip()]

                response = requests.post(
                    f"{API_URL}/ab_test",
                    json={"queries": queries_list, "n_users": n_users},
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()

                # Results summary
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Strategy A Score", f"{result['strategy_a_mean_score']:.4f}", help="FAISS only")
                col2.metric("Strategy B Score", f"{result['strategy_b_mean_score']:.4f}", help="FAISS + XGBoost", delta=f"{result['uplift']:+.4f}")
                col3.metric("Uplift", f"{result['uplift_pct']:+.2f}%")
                col4.metric("P-Value", f"{result['p_value']:.4f}")

                # Significance result
                if result['significant']:
                    st.success(f"✅ Statistically significant difference (p < 0.05). XGBoost re-ranking improves recommendations.")
                else:
                    st.warning(f"⚠️ No statistically significant difference (p = {result['p_value']:.4f}). Both strategies perform similarly.")

                # Bar chart comparing strategies
                import pandas as pd
                chart_data = pd.DataFrame({
                    "Strategy": ["A: FAISS Only", "B: FAISS + XGBoost"],
                    "Mean Score": [result['strategy_a_mean_score'], result['strategy_b_mean_score']]
                })
                st.bar_chart(chart_data.set_index("Strategy"))

                st.caption(f"Test ran on {result['n_samples']} user-query pairs")

                with st.expander("Raw JSON Response"):
                    st.json(result)

            except requests.exceptions.RequestException as e:
                st.error(f"A/B test error: {e}")

# ============================================================
# TAB 4 — Monitoring
# ============================================================

with tab4:
    st.header("Live API Monitoring")
    st.write("Real-time stats refreshing every 30 seconds.")

    # Auto refresh every 30 seconds
    import time as time_module

    placeholder = st.empty()

    def fetch_and_display_metrics():
        try:
            response = requests.get(f"{API_URL}/metrics")
            response.raise_for_status()
            metrics = response.json()

            with placeholder.container():
                # Top level metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Requests", metrics["total_requests"])
                col2.metric("Avg Latency", f"{metrics['avg_latency_ms']:.0f}ms")
                col3.metric("Total Tokens", metrics["total_tokens_used"])
                col4.metric("Total Cost", f"${metrics['total_cost_usd']:.4f}")

                # Requests by endpoint
                st.subheader("Requests by Endpoint")
                if metrics["requests_by_endpoint"]:
                    import pandas as pd
                    endpoint_data = pd.DataFrame({
                        "Endpoint": list(metrics["requests_by_endpoint"].keys()),
                        "Requests": list(metrics["requests_by_endpoint"].values())
                    })
                    st.bar_chart(endpoint_data.set_index("Endpoint"))
                    st.dataframe(endpoint_data, use_container_width=True)
                else:
                    st.info("No requests logged yet. Use other tabs to generate traffic.")

                st.caption(f"Last refreshed: {time_module.strftime('%H:%M:%S')}")

        except requests.exceptions.RequestException as e:
            st.error(f"Could not fetch metrics: {e}")

    # Initial load
    fetch_and_display_metrics()

    # Manual refresh button
    if st.button("🔄 Refresh Now"):
        fetch_and_display_metrics()

    # Auto refresh every 30 seconds
    st.info("Page auto-refreshes every 30 seconds")
    time_module.sleep(30)
    st.rerun()