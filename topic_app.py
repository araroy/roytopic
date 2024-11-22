import streamlit as st
import openai
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
import networkx as nx

# Set OpenAI API Key from Streamlit Secrets and preprocess
def set_openai_api_key():
    raw_api_key = st.secrets.get("OPENAI_API_KEY")
    if not raw_api_key:
        st.error("OpenAI API Key is missing! Add it to the Streamlit secrets.")
        st.stop()
    clean_api_key = raw_api_key.strip()  # Remove any leading/trailing whitespaces
    openai.api_key = clean_api_key

# Function to preprocess text
def preprocess_text(text_column):
    # Remove leading and trailing whitespaces
    return text_column.str.strip()

from transformers import GPT2TokenizerFast

# Initialize GPT-2 tokenizer to estimate tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def estimate_tokens(text):
    """
    Estimate the number of tokens in a given text using GPT-2 tokenizer.
    """
    return len(tokenizer.encode(text))

def get_topics_with_loadings_chunked(input_texts, num_topics, max_tokens=8192, reserved_tokens=500):
    """
    Dynamically process input_texts in chunks to respect token limits for GPT models.
    """
    topics = []
    chunk = []
    token_count = 0

    for text in input_texts:
        text_tokens = estimate_tokens(text)
        if token_count + text_tokens + reserved_tokens > max_tokens:
            # Ensure prompt is defined only when the chunk is not empty
            if chunk:
                prompt = f"""
                You are an AI assistant skilled in topic modeling. Analyze the following texts and extract {num_topics} main topics. 
                Provide the topics in this format:
                Topic 1: Main Topic Text - [Keyword1: Loading1, Keyword2: Loading2, ...]
                Topic 2: Main Topic Text - [Keyword1: Loading1, Keyword2: Loading2, ...]
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

            # Reset for the next chunk
            chunk = [text]
            token_count = text_tokens
        else:
            # Add text to the current chunk
            chunk.append(text)
            token_count += text_tokens

    # Process remaining texts in the last chunk
    if chunk:
        prompt = f"""
        You are an AI assistant skilled in topic modeling. Analyze the following texts and extract {num_topics} main topics. 
        Provide the topics in this format:
        Topic 1: Main Topic Text - [Keyword1: Loading1, Keyword2: Loading2, ...]
        Topic 2: Main Topic Text - [Keyword1: Loading1, Keyword2: Loading2, ...]
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

    return "\n\n".join(topics)


# Function to visualize topics as circles
def visualize_topics(topic_text):
    topics = topic_text.split("\n")
    G = nx.Graph()

    # Parse topics and keywords
    for topic in topics:
        if not topic.strip():
            continue
        topic_name, keywords = topic.split(":", 1)
        keywords = keywords.strip(" -[]").split(",")
        topic_node = topic_name.strip()
        G.add_node(topic_node, size=20)
        for keyword in keywords:
            if ":" in keyword:
                key, loading = keyword.split(":")
                key = key.strip()
                loading = float(loading.strip())
                G.add_node(key, size=10 + loading * 50)  # Size scales with factor loading
                G.add_edge(topic_node, key)

    # Draw the graph
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

# Streamlit App
st.title("GPT-4 Topic Modeling App with Visualization")
st.markdown("""
Upload a CSV or Excel file, and use GPT-4 for topic modeling:
1. Text preprocessing: Remove whitespaces.
2. Securely access OpenAI API key.
3. Extract main topics with factor loadings.
4. Visualize topics as circles.
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

        # Remove whitespaces
        if st.button("Preprocess Text"):
            df[text_column] = preprocess_text(df[text_column])
            st.success("Text preprocessed successfully!")
            st.write("Updated Data:")
            st.write(df[[text_column]].head())

        # Topic Modeling
        if st.button("Run Topic Modeling"):
            set_openai_api_key()  # Set OpenAI API key
            input_texts = df[text_column].dropna().tolist()

    # Ensure chunk_size is defined before calling the function
            num_topics = st.slider("Number of Topics to Extract", min_value=1, max_value=10, value=5)

            chunk_size = st.slider("Chunk Size (Number of Rows per Request)", min_value=100, max_value=2000, value=1000)
        
            set_openai_api_key()  # Set OpenAI API key
            input_texts = df[text_column].dropna().tolist()
            topic_text = get_topics_with_loadings_chunked(input_texts, num_topics, chunk_size)
            st.markdown("### Extracted Topics with Loadings")
            st.text(topic_text)

            # Visualize Topics
            st.markdown("### Topic Visualization")
            visualize_topics(topic_text)

            # Option to download the topics
            topics_df = pd.DataFrame({"Extracted Topics": topic_text.split("\n")})
            output = BytesIO()
            topics_df.to_csv(output, index=False)
            output.seek(0)
            st.download_button(
                label="Download Topics",
                data=output,
                file_name="extracted_topics.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing the file: {e}")
