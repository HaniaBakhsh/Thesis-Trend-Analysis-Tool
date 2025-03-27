
import streamlit as st
from main_complete_final import run_trend_analysis, run_topic_modeling
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Thesis Topic Explorer", layout="wide")

if "run" not in st.session_state:
    st.session_state.run = False
if "reset" not in st.session_state:
    st.session_state.reset = False

def reset():
    st.session_state.run = False
    st.session_state.reset = True

st.title("🎓 Thesis Topic Modeling & Trend Explorer")

st.markdown("""
### 📘 How to Use This App

- **Upload** a CSV file with at least 'Title' and 'Abstract' columns. A 'Year' column is needed for trend analysis.
- **Choose the analysis mode**:
  - **Topic Modeling**: Groups theses into meaningful topics based on abstract content.
  - **Trend Analysis**: Shows how topics evolve over the years and highlights trends.
- **Enter number of topics**: This defines how many clusters or themes you want the system to detect.
- Results will include topic summaries, thesis counts, and downloadable reports.

---

""")

with st.sidebar:
    st.header("🔧 Settings")
    uploaded_file = st.file_uploader("📤 Upload your thesis dataset (.csv)", type=["csv"])
    analysis_mode = st.radio("📊 Choose Analysis Mode", ["Topic Modeling", "Trend Analysis"])
    num_topics = st.number_input("🔢 Number of Topics", min_value=2, max_value=50, step=1)
    if st.button("▶️ Run Analysis"):
        st.session_state.run = True
    if st.button("🔄 Restart Analysis"):
        reset()

if uploaded_file and st.session_state.run:
    st.markdown("---")
    if analysis_mode == "Trend Analysis":
        with st.spinner("Processing trend analysis..."):
            template_path = "trend_report_template_final.html"
            trend_df, insights, html_path, df = run_trend_analysis(uploaded_file, num_topics, template_path)

            st.subheader("📝 Key Insights")
            for insight in insights:
                st.markdown(f"- {insight}", unsafe_allow_html=True)

            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.subheader("🏆 Topics and Thesis Counts")
            topic_counts = df["Topic"].value_counts().sort_values(ascending=False)
            st.dataframe(topic_counts.rename_axis("Topic").reset_index(name="Thesis Count"))

            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.subheader("📈 Topic Trend Chart")
            top_topics = trend_df.groupby("Topic")["Count"].sum().nlargest(num_topics).index
            filtered_df = trend_df[trend_df["Topic"].isin(top_topics)]
            fig = px.line(filtered_df, x="Year", y="Count", color="Topic", markers=True,
                          title="Topic Trends Over Time", color_discrete_sequence=px.colors.qualitative.Set1)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.subheader("📚 Theses by Topic")
            for topic in topic_counts.index:
                with st.expander(topic):
                    for _, row in df[df["Topic"] == topic][["Title", "Year"]].iterrows():
                        year_info = f" ({int(row['Year'])})" if "Year" in row and pd.notna(row["Year"]) else ""
                        st.markdown(f"- {row['Title']}{year_info}")

            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.subheader("🔍 Search Titles or Topics")
            query = st.text_input("Enter keyword:")
            if query:
                result_df = df[df["Title"].str.contains(query, case=False) | df["Topic"].str.contains(query, case=False)]
                st.dataframe(result_df[["Title", "Topic", "Year"]] if "Year" in result_df.columns else result_df[["Title", "Topic"]])

            if "Year" in df.columns:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                st.subheader("🏅 Dominant Topic Per Year")
                dominant = df.groupby("Year")["Topic"].agg(lambda x: x.value_counts().idxmax())
                st.dataframe(dominant)

            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.subheader("📥 Download Report")
            with open(html_path, "rb") as f:
                st.download_button("Download HTML Report", f, file_name="trend_report.html", mime="text/html")

    elif analysis_mode == "Topic Modeling":
        with st.spinner("Processing topic modeling..."):
            template_path = "trend_report_template_final.html"
            df, insights, html_path = run_topic_modeling(uploaded_file, num_topics, template_path)

            st.subheader("📝 Summary")
            for insight in insights:
                st.markdown(f"- {insight}", unsafe_allow_html=True)

            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.subheader("🏆 Topics and Thesis Counts")
            topic_counts = df["Topic"].value_counts().sort_values(ascending=False)
            st.dataframe(topic_counts.rename_axis("Topic").reset_index(name="Thesis Count"))

            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.subheader("📚 Theses by Topic")
            for topic in topic_counts.index:
                with st.expander(topic):
                    for _, row in df[df["Topic"] == topic][["Title", "Year"]].iterrows():
                        year_info = f" ({int(row['Year'])})" if "Year" in row and pd.notna(row["Year"]) else ""
                        st.markdown(f"- {row['Title']}{year_info}")

            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.subheader("🔍 Search Titles or Topics")
            query = st.text_input("Enter keyword:")
            if query:
                result_df = df[df["Title"].str.contains(query, case=False) | df["Topic"].str.contains(query, case=False)]
                st.dataframe(result_df[["Title", "Topic", "Year"]] if "Year" in result_df.columns else result_df[["Title", "Topic"]])

            st.markdown("&nbsp;", unsafe_allow_html=True)
            st.subheader("📥 Download Report")
            with open(html_path, "rb") as f:
                st.download_button("Download HTML Report", f, file_name="topic_modeling_report.html", mime="text/html")
