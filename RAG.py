# app.py
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Models to be used
models = {
    "BAAI/bge-small-en": SentenceTransformer('BAAI/bge-small-en'),
    "thenlper/gte-small": SentenceTransformer('thenlper/gte-small'),
    "paraphrase-MiniLM-L6-v2": SentenceTransformer('paraphrase-MiniLM-L6-v2'),
    "BAAI/bge-large-en": SentenceTransformer('BAAI/bge-large-en'),
    "thenlper/gte-large": SentenceTransformer('thenlper/gte-large')
}

# Streamlit App
st.title('Text Similarity Search')

# Upload input files
uploaded_file_from = st.file_uploader("Choose an Input From file", type="xlsx")
uploaded_file_to = st.file_uploader("Choose an Input To file", type="xlsx")

# Perform similarity search if files are uploaded
if uploaded_file_from and uploaded_file_to:
    df_from = pd.read_excel(uploaded_file_from)
    df_to = pd.read_excel(uploaded_file_to)

    if 'statement' in df_from.columns and 'highlights' in df_to.columns:
        statements = df_from['statement'].tolist()
        highlights = df_to['highlights'].tolist()

        # Perform similarity search
        top_k = 5
        results = []

        for model_name, model in models.items():
            statement_embeddings = model.encode(statements, convert_to_tensor=True)
            highlight_embeddings = model.encode(highlights, convert_to_tensor=True)

            for idx, statement_embedding in enumerate(statement_embeddings):
                similarities = util.pytorch_cos_sim(statement_embedding, highlight_embeddings)
                top_k_results = similarities.topk(top_k)
                for score, index in zip(top_k_results[0], top_k_results[1]):
                    results.append({
                        'statement': statements[idx],
                        'highlight': highlights[index],
                        'similarity_score': score.item(),
                        'model': model_name
                    })

        results_df = pd.DataFrame(results)
        
        st.write("Similarity Results")
        st.dataframe(results_df)
        
        # Option to download the results as an Excel file
        st.download_button(
            label="Download results as Excel",
            data=results_df.to_excel(index=False),
            file_name='similarity_results.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    else:
        st.error("Input files must contain 'statement' and 'highlights' columns.")
