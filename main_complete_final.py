
import pandas as pd
import numpy as np
import re
#import os
import streamlit as st
from jinja2 import Template
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from dotenv import load_dotenv
import google.generativeai as genai
import nltk
import base64

nltk.data.path.append("nltk_data")
load_dotenv()

def extract_english_abstract(text):
    soup = BeautifulSoup(text, "html.parser")
    paragraphs = soup.find_all("p")
    return paragraphs[0].get_text() if paragraphs else ""

def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english')).union({"thesis", "study", "research", "result", "method", "approach", "process"})
    words = [word for word in words if word not in stop_words and len(word) > 2]
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in words])

def preprocess_csv(file):
    df = pd.read_csv(file)
    df = df.dropna(subset=["Abstract"])
    df["English_Abstract"] = df["Abstract"].apply(extract_english_abstract)
    df = df[df["English_Abstract"].apply(is_english)]
    df["Cleaned_Abstract"] = df["English_Abstract"].apply(clean_text)
    df = df[df["Cleaned_Abstract"].str.strip() != ""]
    return df

def embed(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts)

def kmeans_clustering(embeddings, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings)
    return kmeans.labels_

def generate_topic(texts):
    #api_key = os.getenv('GOOGLE_API_KEY')
    api_key = st.secrets["GOOGLE_API_KEY"]
    client = genai.Client(api_key=api_key)
    prompt = f"""
Given the following thesis abstracts, assign a name for the general topic they represent.
Avoid listing multiple topics.
Only return the name of the topic in 3-6 words, no extra explanation.

Abstracts:
{texts}
"""
    try:
        response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt]
    )
        #return response.text.strip().replace("**", "")
        ''' return response.text.strip().replace("**", "").replace("*", "").strip().split(".")[0][:60]
    except Exception as e:
        return "Unnamed Topic" '''
        topic_name = response.text.strip()
        
                 # Clean: remove asterisks, special characters
        
        #topic_name = topic_name.replace("**", "").replace("*", "").strip()
        #topic_name = re.sub(r"[^\w\s&-]", "", topic_name)
        #topic_name = re.sub(r"[^a-zA-Z0-9\\s&\\-]", "", topic_name)
        

  # Clean: remove *, numbers, punctuation, symbols — keep only letters and spaces
        topic_name = topic_name.replace("**", "").replace("*", "").strip()
        topic_name = re.sub(r"[^a-zA-Z\s]", "", topic_name)
        
        # Truncate to max 8 words
        words = topic_name.split()
        if len(words) > 4:
            topic_name = " ".join(words[:8])
        
        # Fallback if result is empty or generic
        if not topic_name or topic_name.lower() in ["unnamed topic", ""]:
            topic_name = "Unnamed Topic"

        return topic_name

    except Exception:
        return "Unnamed Topic"

def label_topics(df, labels, abstracts, k):
    df["Cluster"] = labels
    topic_names = {}
    for cluster in range(k):
        cluster_abstracts = [abstracts[i] for i in range(len(abstracts)) if labels[i] == cluster][:10]
        topic_name = generate_topic("\n".join(cluster_abstracts))
        topic_name = topic_name.replace("*", "").strip()
        topic_names[cluster] = topic_name
    df["Topic"] = df["Cluster"].apply(lambda x: topic_names.get(x, f"Topic {x+1}"))
    return df, topic_names

def generate_insights(trend_df):
    insights = []
    topic_totals = trend_df.groupby("Topic")["Count"].sum().sort_values(ascending=False)
    if not topic_totals.empty:
        insights.append(f"The most popular topic overall is <b>{topic_totals.index[0]}</b> with {topic_totals.iloc[0]} theses.")
    yearly_diff = trend_df.pivot(index="Year", columns="Topic", values="Count").fillna(0).diff().sum().sort_values(ascending=False)
    if not yearly_diff.empty:
        insights.append(f"The fastest growing topic is <b>{yearly_diff.index[0]}</b> (+{int(yearly_diff.iloc[0])}).")
    return insights

def prepare_trend_data(df):
    trend_data = df.groupby(["Year", "Topic"]).size().reset_index(name="Count")
    return trend_data

def generate_html_report(insights, trend_df, output_path, template_path, theses_by_topic, topic_counts, plot_b64=None):
    latest_year = trend_df["Year"].max() if not trend_df.empty else "N/A"
    sorted_topic_counts = dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True))
    sorted_theses_by_topic = {k: theses_by_topic[k] for k in sorted_topic_counts.keys()}
    with open(template_path, "r", encoding="utf-8") as f:
        template = Template(f.read())
    html_content = template.render(
        insights=insights,
        plot_b64=plot_b64,
        latest_year=latest_year,
        topic_counts=sorted_topic_counts,
        theses_by_topic=sorted_theses_by_topic
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_path

def run_trend_analysis(file, num_topics, template_path):
    df = preprocess_csv(file)
    abstracts = df["Cleaned_Abstract"].tolist()
    embeddings = embed(abstracts)
    labels = kmeans_clustering(embeddings, num_topics)
    df, topic_names = label_topics(df, labels, abstracts, num_topics)
    trend_df = prepare_trend_data(df)
    insights = generate_insights(trend_df)
    html_path = "trend_report.html"
    theses_by_topic = {
        name: df[df["Topic"] == name]["Title"].tolist()
        for name in sorted(topic_names.values())
    }
    topic_counts = {
        name: len(theses_by_topic[name]) for name in theses_by_topic
    }
    filtered_df = trend_df[trend_df["Topic"].isin(topic_counts.keys())]
    fig = px.line(
        filtered_df, x="Year", y="Count", color="Topic", markers=True,
        title="Topic Trends Over Time", color_discrete_sequence=px.colors.qualitative.Set1
    )
    #plotly_json = fig.to_dict()
    #plotly_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    fig_bytes = fig.to_image(format="png")
    plot_b64 = base64.b64encode(fig_bytes).decode("utf-8")
    generate_html_report(insights, trend_df, html_path, template_path, theses_by_topic, topic_counts, plot_b64=plot_b64)
    return trend_df, insights, html_path, df

def run_topic_modeling(file, num_topics, template_path):
    df = preprocess_csv(file)
    abstracts = df["Cleaned_Abstract"].tolist()
    embeddings = embed(abstracts)
    labels = kmeans_clustering(embeddings, num_topics)
    df, topic_names = label_topics(df, labels, abstracts, num_topics)
    trend_df = df.groupby(["Year", "Topic"]).size().reset_index(name="Count") if "Year" in df.columns else pd.DataFrame(columns=["Year", "Topic", "Count"])
    insights = [f"Identified {num_topics} meaningful topics based on thesis abstracts."]
    html_path = "topic_modeling_report.html"
    theses_by_topic = {
        name: df[df["Topic"] == name]["Title"].tolist()
        for name in sorted(topic_names.values())
    }
    topic_counts = {
        name: len(theses_by_topic[name]) for name in theses_by_topic
    }
    #plotly_json = None
    #generate_html_report(insights, trend_df, html_path, template_path, theses_by_topic, topic_counts, plotly_json)
    
    generate_html_report(insights, trend_df, html_path, template_path, theses_by_topic, topic_counts, plot_b64=None)
    return df, insights, html_path
