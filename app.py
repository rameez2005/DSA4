import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df_rq1_results = pd.read_csv("df_rq1_results.csv", index_col=0)
df_rq2_results = pd.read_csv("df_rq2_results.csv", index_col=0)
df_rq3_results = pd.read_csv("df_rq3_results.csv", index_col=0)

# -------------------------------
# Title
st.title("Student Marks Prediction Dashboard")
st.markdown("""
This dashboard presents the analysis, evaluation, and model results for predicting student marks.
""")

# -------------------------------
# Workflow Diagram
st.header("Workflow / Pipeline")
st.image("workflow_diagram.png", caption='Prediction Workflow', use_column_width=True)

# -------------------------------
# Load results (replace these with your actual DataFrames)
# Ensure index contains model names
# df_rq1_results, df_rq2_results, df_rq3_results

rqs = {
    "RQ1: Predict Midterm I": df_rq1_results,
    "RQ2: Predict Midterm II": df_rq2_results,
    "RQ3: Predict Final Exam": df_rq3_results
}

# -------------------------------
# Tabs for each RQ
tabs = st.tabs(list(rqs.keys()))

for tab, (rq_name, df) in zip(tabs, rqs.items()):
    with tab:
        st.subheader(f"{rq_name} - Results Table")
        st.dataframe(df)

        # Highlight best model based on R2_test
        best_model_name = df['R2_test'].idxmax()
        st.success(f"Best Model: {best_model_name}")

        # Train vs Test R² bar plot
        plt.figure(figsize=(6,4))
        r2_plot = df[['R2_train','R2_test']].copy()
        r2_plot.index = df.index
        r2_plot.plot(kind='bar', figsize=(6,4))
        plt.title(f"Train vs Test R² - {rq_name}")
        plt.ylabel("R²")
        plt.xticks(rotation=0)
        st.pyplot(plt.gcf())

        # Bootstrap MAE 95% CI plot
        plt.figure(figsize=(6,4))
        mae_boot_mean = df['MAE_boot_mean']
        yerr_lower = df['MAE_boot_mean'] - df['MAE_boot_2.5pct']
        yerr_upper = df['MAE_boot_97.5pct'] - df['MAE_boot_mean']
        yerr = [yerr_lower, yerr_upper]
        mae_boot_mean.plot(kind='bar', yerr=yerr, capsize=5, figsize=(6,4))
        plt.title(f"Bootstrapped MAE 95% CI - {rq_name}")
        plt.ylabel("MAE")
        plt.xticks(rotation=0)
        st.pyplot(plt.gcf())
