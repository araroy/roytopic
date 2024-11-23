import streamlit as st
import openai
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from transformers import GPT2TokenizerFast
from difflib import SequenceMatcher


# Set OpenAI API Key
def set_openai_api_key():
    raw_api_key = st.secrets.get("OPENAI_API_KEY")
    if not raw_api_key:
        st.error("OpenAI API Key is missing! Add it to the Streamlit secrets.")
        st.stop()
    clean_api_key = raw_api_key.strip()  # Remove any leading/trailing whitespaces
    openai.api_key = clean_api_key
# Deduplicate topics
def deduplicate_topics(topic_list, similarity_threshold=0.8):
    """
    Deduplicate topics based on text similarity.

    Args:
        topic_list (list): List of topics as strings.
        similarity_threshold (float): Minimum similarity ratio to consider topics as duplicates.

    Returns:
        list: Deduplicated list of topics.
    """
    unique_topics = []
    for topic in topic_list:
        if not topic.strip():
            continue
        is_duplicate = False
        for unique_topic in unique_topics:
            # Compute similarity between the current topic and existing unique topics
            similarity = SequenceMatcher(None, topic, unique_topic).ratio()
            if similarity > similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_topics.append(topic)
    return unique_topics
# Preprocess text
def preprocess_text(text_column):
    return text_column.str.strip()

# Initialize GPT-2 tokenizer to estimate tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def estimate_tokens(text):
    return len(tokenizer.encode(text))

def get_topics_with_loadings_chunked(input_texts, num_topics, max_tokens=8192, reserved_tokens=500):
    topics = []
    chunk = []
    token_count = 0

    for text in input_texts:
        text_tokens = estimate_tokens(text)
        if token_count + text_tokens + reserved_tokens > max_tokens:
            if chunk:
                prompt = f"""
                You are an AI assistant skilled in topic modeling. Analyze the following texts and extract {num_topics} main topics. 
                Provide the topics in this format:
                Topic 1: Main Topic Text - [Keyword1: Loading1, Keyword2: Loading2, ..., Keyword10: Loading10]
                Topic 2: Main Topic Text - [Keyword1: Loading1, Keyword2: Loading2, ..., Keyword10: Loading10]
                Texts: {' '.join(chunk)}
                """
                try:
                    response = openai.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    topics.append(response.choices[0].message.content)
                except Exception as e:
                    topics.append(f"Error: {e}")

            chunk = [text]
            token_count = text_tokens
        else:
            chunk.append(text)
            token_count += text_tokens

    if chunk:
        prompt = f"""
        You are an AI assistant skilled in topic modeling. Analyze the following texts and extract {num_topics} unique main topics. 
        Ensure that the topics do not overlap significantly. Provide the topics in this format:
        Topic 1: Main Topic Text - [Keyword1: Loading1, Keyword2: Loading2, ..., Keyword10: Loading10]
        Topic 2: Main Topic Text - [Keyword1: Loading1, Keyword2: Loading2, ..., Keyword10: Loading10]
        Texts: {' '.join(chunk)}
        """
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            topics.append(response.choices[0].message.content)
        except Exception as e:
            topics.append(f"Error: {e}")
    topics_combined = "\n".join(topics)
    unique_topics = deduplicate_topics(topics_combined.split("\n"))
    return "\n".join(unique_topics)

# Consolidate similar topics
def consolidate_topics(topic_texts, n_clusters):
    vectorizer = TfidfVectorizer(stop_words='english')
    topic_vectors = vectorizer.fit_transform(topic_texts)

    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = clustering_model.fit_predict(topic_vectors.toarray())

    consolidated_topics = {}
    for label, topic in zip(labels, topic_texts):
        consolidated_topics.setdefault(label, []).append(topic)

    consolidated_results = []
    for label, topics in consolidated_topics.items():
        consolidated_results.append(f"Cluster {label + 1}: " + " | ".join(topics))

    return consolidated_results

# Add topic columns
for i, topic in enumerate(raw_topic_list, start=1):
    if ":" in topic:
            topic_name, keywords = topic.split(":", 1)
            keywords = [kw.split(":")[0].strip() for kw in keywords.strip(" -[]").split(",")]
            df[f"Topic {i}"] = df[text_column].apply(
                lambda x: 1 if any(keyword in str(x) for keyword in keywords) else 0
            )

            st.markdown("### Updated Dataset with Topic Columns")
            st.write(df.head())

            # Allow users to download the updated dataset
            output = BytesIO()
            df.to_csv(output, index=False)
            output.seek(0)
            st.download_button(
                label="Download Updated Dataset",
                data=output,
                file_name="updated_dataset_with_topics.csv",
                mime="text/csv"
            )


# Visualize topics as circles
def visualize_topics(topic_text):
    topics = topic_text.split("\n")
    G = nx.Graph()

    for topic in topics:
        if not topic.strip():
            continue
        try:
            topic_name, keywords = topic.split(":", 1)
            keywords = keywords.strip(" -[]").split(",")
            topic_node = topic_name.strip()
            G.add_node(topic_node, size=20)
            for keyword in keywords:
                if ":" in keyword:
                    key, loading = keyword.split(":")
                    key = key.strip()
                    loading = float(loading.strip())
                    G.add_node(key, size=10 + loading * 50)
                    G.add_edge(topic_node, key)
        except ValueError:
            continue

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=[G.nodes[node]["size"] * 50 for node in G.nodes],
        font_size=10,
        node_color="skyblue",
        edge_color="gray",
    )
    plt.title("Topic Modeling Visualization", fontsize=14)
    st.pyplot(plt.gcf())
    plt.clf()
# Propose the number of topics using factor loadings
def propose_number_of_topics(texts, max_topics=20):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=min(max_topics, tfidf_matrix.shape[1] - 1))
    svd.fit(tfidf_matrix)

    explained_variance = svd.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    # Plot Scree Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker="o", label="Cumulative Variance")
    plt.axhline(y=0.8, color="r", linestyle="--", label="80% Variance Threshold")
    plt.title("Scree Plot for Number of Topics")
    plt.xlabel("Number of Topics")
    plt.ylabel("Cumulative Explained Variance")
    plt.legend()
    st.pyplot(plt.gcf())

    # Suggest number of topics
    suggested_topics = next((i + 1 for i, var in enumerate(cumulative_variance) if var >= 0.8), max_topics)
    return suggested_topics
# Streamlit App
st.title("GPT-4 Topic Modeling App with Visualization")
st.markdown("""
Upload a CSV or Excel file, and use GPT-4 for topic modeling:
1. Text preprocessing: Remove whitespaces.
2. Securely access OpenAI API key.
3. Extract main topics with factor loadings in manageable chunks.
4. Consolidate similar topics.
5. Visualize topics as circles.
""")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Load data
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        st.write("Preview of Uploaded Data:")
        st.write(df.head())

        # Select column for topic modeling
        text_column = st.selectbox("Select the column for topic modeling", options=df.columns)

        # Preprocess Text
        if st.button("Preprocess Text"):
            df[text_column] = preprocess_text(df[text_column])
            st.success("Text preprocessed successfully!")
            st.write(df[[text_column]].head())

        # Topic Modeling
        num_topics = st.slider("Number of Topics to Extract", min_value=1, max_value=10, value=5)
        n_clusters = st.slider("Number of Consolidated Topics", min_value=2, max_value=20, value=5)
        if st.button("Run Topic Modeling"):
            set_openai_api_key()
            input_texts = df[text_column].dropna().tolist()
            raw_topics = get_topics_with_loadings_chunked(input_texts, num_topics)

            raw_topic_list = raw_topics.split("\n")
            consolidated_topics = consolidate_topics(raw_topic_list, n_clusters)

            st.markdown("### Extracted Topics with Loadings")
            st.text("\n".join(raw_topic_list))

            st.markdown("### Consolidated Topics")
            st.text("\n".join(consolidated_topics))
             # Add topic columns to the dataset
            for i, topic in enumerate(raw_topic_list, start=1):
                if ":" in topic:
                    topic_name, keywords = topic.split(":", 1)
                    keywords = [kw.split(":")[0].strip()

            # Visualize Topics
            st.markdown("### Topic Visualization")
            visualize_topics("\n".join(raw_topic_list))

            # Download Consolidated Topics
            topics_df = pd.DataFrame({"Consolidated Topics": consolidated_topics})
            output = BytesIO()
            topics_df.to_csv(output, index=False)
            output.seek(0)
            st.download_button(
                label="Download Consolidated Topics",
                data=output,
                file_name="consolidated_topics.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing the file: {e}")
