import streamlit as st
import pandas as pd
import altair as alt
import os
import sys

# Ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pipeline.orchestrator import run_pipeline
from config import INTERACTIONS_LOG


st.set_page_config(page_title="JAX + LLM Routing Dashboard", layout="wide")


# ------------------------------------------------------------
# ⚡ CHAT / DEMO PAGE
# ------------------------------------------------------------
def page_chat():
    st.title("⚡ JAX + LLM Routing Demo")

    st.write(
        "This demo shows:\n"
        "- JAX-based category router\n"
        "- RAG-enhanced LLM answers\n"
        "- Guardrails for low confidence / quality\n"
        "- Judge scoring"
    )

    user_message = st.text_area(
        "Enter a customer support message:",
        height=120,
        placeholder="I was charged twice this month...",
    )

    if st.button("Run Pipeline 🚀", type="primary"):
        if not user_message.strip():
            st.warning("Please enter a message.")
            return

        with st.spinner("Running full pipeline..."):
            result = run_pipeline(user_message)

        # ---------- Router ----------
        st.subheader("Router")
        st.write(f"**Category:** {result['category']}")
        st.write(f"**Confidence:** {result['router_confidence']:.3f}")

        # ---------- RAG ----------
        st.subheader("Knowledge Base Hits (RAG)")
        if result["kb_hits"]:
            for hit in result["kb_hits"]:
                st.write(
                    f"- `{hit['category']}` – **{hit['title']}** "
                    f"(similarity={hit['similarity']:.3f})"
                )
        else:
            st.write("_No KB hits found._")

        # ---------- LLM ----------
        st.subheader("LLM Answer")
        st.write(result["answer"])

        # ---------- Judge ----------
        st.subheader("Judge")
        st.write(f"**Score:** {result['judge_score']:.3f}")
        st.write(result["judge_reasoning"])

        # ---------- Guardrails ----------
        st.subheader("Guardrails")
        st.write(f"**Action:** `{result['guardrail_action']}`")


# ------------------------------------------------------------
# EVALUATION DASHBOARD
# ------------------------------------------------------------
def page_evaluation():
    st.title("📊 Evaluation Dashboard")

    if not os.path.exists(INTERACTIONS_LOG):
        st.info("No interactions logged yet. Use the Chat / Demo page first.")
        return

    df = pd.read_csv(INTERACTIONS_LOG)

    if df.empty:
        st.info("Log file is empty. Use the demo page first.")
        return

    # =============================
    # 1. High-Level KPIs
    # =============================
    st.subheader("Overview Metrics")

    total = len(df)
    avg_conf = df["router_confidence"].mean()
    avg_score = df["judge_score"].mean()
    escalation_rate = (df["guardrail_action"] != "OK").mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tickets", total)
    c2.metric("Avg Router Confidence", f"{avg_conf:.3f}")
    c3.metric("Avg Judge Score", f"{avg_score:.3f}")
    c4.metric("Escalation Rate", f"{escalation_rate*100:.1f}%")


    st.markdown("---")

    # =============================
    # 2. Per-Category Performance
    # =============================
    st.subheader("Category-Level Performance")

    by_cat = (
        df.groupby("predicted_category")
        .agg(
            tickets=("predicted_category", "size"),
            avg_conf=("router_confidence", "mean"),
            avg_score=("judge_score", "mean"),
            escalation_rate=("guardrail_action", lambda x: (x != "OK").mean()),
        )
        .reset_index()
    )

    # FAANG-style risk categories (low performance)
    risky = by_cat[
        (by_cat["avg_score"] < 0.75) | (by_cat["escalation_rate"] > 0.25)
    ]

    # Tickets per category chart
    st.altair_chart(
        alt.Chart(by_cat)
        .mark_bar()
        .encode(
            x="predicted_category:N",
            y="tickets:Q",
            color="predicted_category:N",
        ),
        use_container_width=True,
    )

    # Avg judge score chart
    st.altair_chart(
        alt.Chart(by_cat)
        .mark_line(point=True)
        .encode(
            x="predicted_category:N",
            y="avg_score:Q",
            color="predicted_category:N",
        ),
        use_container_width=True,
    )

    st.write("Detailed Metrics")
    st.dataframe(
        by_cat.style.format(
            {
                "avg_conf": "{:.3f}",
                "avg_score": "{:.3f}",
                "escalation_rate": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

    if not risky.empty:
        st.warning("⚠️ Categories with low scores or high escalation:")
        st.dataframe(risky)

    st.markdown("---")

    # =============================
    # 3. Judge Score Distribution
    # =============================
    st.subheader("Judge Score Distribution")

    hist = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("judge_score:Q", bin=alt.Bin(step=0.1)),
            y="count():Q",
        )
    )
    st.altair_chart(hist, use_container_width=True)

    # Boxplot by category
    st.altair_chart(
        alt.Chart(df)
        .mark_boxplot()
        .encode(
            x="predicted_category:N",
            y="judge_score:Q",
            color="predicted_category:N",
        ),
        use_container_width=True,
    )

    st.markdown("---")

    # =============================
    # 4. Router Confidence vs Judge Score
    # =============================
    st.subheader("Router Confidence vs Judge Score")

    scatter = (
        alt.Chart(df)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x="router_confidence:Q",
            y="judge_score:Q",
            color="predicted_category:N",
            tooltip=[
                "timestamp",
                "user_message",
                "predicted_category",
                "router_confidence",
                "judge_score",
                "guardrail_action",
            ],
        )
    )
    st.altair_chart(scatter, use_container_width=True)

    st.markdown("---")

    # =============================
    # 5. Lowest-quality tickets
    # =============================
    st.subheader("❗ Worst Tickets (Low Judge Scores)")

    worst = df.sort_values("judge_score").head(10)
    st.dataframe(
        worst[
            [
                "timestamp",
                "user_message",
                "predicted_category",
                "router_confidence",
                "judge_score",
                "guardrail_action",
            ]
        ],
        use_container_width=True,
    )

    st.markdown("---")

    # =============================
    # 6. Download logs
    # =============================
    st.subheader("⬇️ Export Logs")

    st.download_button(
        label="Download interactions.csv",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="interactions.csv",
        mime="text/csv",
    )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    page = st.sidebar.radio(
        "View",
        ["Chat / Demo", "Evaluation Dashboard"],
    )

    if page == "Chat / Demo":
        page_chat()
    else:
        page_evaluation()


if __name__ == "__main__":
    main()
