import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

# -----------------
# Fake Data Setup
# -----------------
st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    data = [
        # Hiring decisions – Llama
        ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", "Hiring decisions", "cot", 70.2],
        ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", "Hiring decisions", "toxic", 66.2],
        ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", "Hiring decisions", "nontoxic", 91.9],
        ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", "Hiring decisions", "concise", 66.7],
        ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", "Hiring decisions", "detailed", 70.3],

        # Hiring decisions – DeepSeek
        ["deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", "Hiring decisions", "cot", 72.0],
        ["deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", "Hiring decisions", "toxic", 71.6],
        ["deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", "Hiring decisions", "nontoxic", 71.2],
        ["deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", "Hiring decisions", "concise", 65.2],
        ["deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", "Hiring decisions", "detailed", 69.4],

        # Harmful requests – Llama
        ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", "Harmful requests", "cot", 78.1],
        ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", "Harmful requests", "toxic", 77.7],
        ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", "Harmful requests", "nontoxic", 79.8],
        ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", "Harmful requests", "concise", 80.5],
        ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", "Harmful requests", "detailed", 83.8],

        # Harmful requests – DeepSeek
        ["deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", "Harmful requests", "cot", 80.9],
        ["deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", "Harmful requests", "toxic", 73.2],
        ["deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", "Harmful requests", "nontoxic", 76.8],
        ["deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", "Harmful requests", "concise", 73.7],
        ["deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", "Harmful requests", "detailed", 81.9],
    ]
    return pd.DataFrame(data, columns=["model", "dataset", "explanation_type", "precision"])


df = load_data()

# -----------------
# Sidebar Navigation
# -----------------
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        width: 350px !important;  # Set the desired width (e.g., 400px)
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.sidebar.title("Model Selection")
# page = st.sidebar.radio("Go to", ["Model Comparison", "Leaderboard"])

# -----------------
# Page 1: Model Comparison
# -----------------
# if page == "Model Comparison":
st.title("Theory of Mind for Explainable AI")

# Filters
models = sorted(df["model"].unique().tolist())
datasets = sorted(df["dataset"].unique().tolist())

selected_models = st.sidebar.multiselect("Select Model", models)
selected_datasets = st.sidebar.multiselect("Select Dataset", datasets)

# Apply filters
# filtered_df = df.copy()
if not selected_models:
        selected_models = models
if not selected_datasets:
    selected_datasets = datasets
    
filtered_df = df[
        df["model"].isin(selected_models) & df["dataset"].isin(selected_datasets)
]
   
leaderboard = df.pivot_table(
    index=["model", "dataset"],
    columns="explanation_type",
    values="precision",
    aggfunc="mean",
).reset_index()

# Format only numeric precision columns
numeric_cols = leaderboard.select_dtypes(include="number").columns

styled = (
    leaderboard.style
    .format({col: "{:.2f}" for col in numeric_cols})
    .set_properties(**{
        "font-size": "30pt",       # larger text
        "border": "1px solid black",
        "padding": "6px"
    })
)

# Display with larger width
st.subheader("Overall Leaderboard")
st.dataframe(styled, use_container_width=True)

st.subheader("Graphed Simulation Precision")
if not filtered_df.empty:
    fig = px.bar(
        filtered_df,
        x="explanation_type",
        y="precision",
        color="model",
        barmode="group",   # key for grouped bars
        facet_row="dataset" if len(selected_datasets) > 1 else None,
        title="Precision grouped by Model and Explanation Type",
        height = 800,
        width = 1200,
        color_discrete_sequence=["#dd7272", "#b7b6b4"]
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data matches the filters.")

# -----------------
# Page 2: Leaderboard
# -----------------
# elif page == "Leaderboard":
#     st.title("Simulation Precision Leaderboard")

#     leaderboard = df.pivot_table(
#         index=["model", "dataset"],
#         columns="explanation_type",
#         values="precision",
#         aggfunc="mean",
#     ).reset_index()

#     # Format only numeric precision columns
#     numeric_cols = leaderboard.select_dtypes(include="number").columns
    
#     styled = (
#         leaderboard.style
#         .format({col: "{:.3f}" for col in numeric_cols})
#         .set_properties(**{
#             "font-size": "36pt",       # larger text
#             "border": "1px solid black",
#             "padding": "6px"
#         })
#     )

#     # Display with larger width
#     st.dataframe(styled, use_container_width=True)
