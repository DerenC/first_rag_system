import streamlit as st
import backend as rag
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np

st.title("Vector Visualization")
st.subheader("Visualise your embeddings in a 2D plot using PCA")
st.divider()

query = st.text_input("Enter a query to visualize it")

if st.button("Plot") and query:
    with st.spinner("Reducing 768 dimensions to 2"):
        data = rag.get_vectors_for_visualization(query)

        # Query vector + All relevant knowledge vectors
        vectors = [data["query_vector"]] + [doc_data["vector"] for doc_data in data["docs_data"]]
        labels = ["YOUR QUERY"] + [doc_data["full_text"][:50] + "..." for doc_data in data["docs_data"]]
        types = ["Query"] + ["Result" for _ in data["docs_data"]]

        n_components = min(2, len(data["query_vector"]))
        pca = PCA(n_components=n_components)
        reduced_vectors = pca.fit_transform(np.array(vectors))

        df = pd.DataFrame(reduced_vectors, columns=["x", "y"] if n_components == 2 else ["x"])
        if n_components == 1: df["y"] = 0

        df["label"] = labels
        df["type"] = types
        df["full_text"] = ["User Query"] + [doc_data["full_text"] for doc_data in data["docs_data"]]

        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="type",
            text="label",
            hover_data=["full_text"],
            title="Semantic Distance Map",
            size_max=20,
            color_discrete_map={"Query": "red", "Result": "blue"},
        )

        fig.update_traces(textposition="top center", marker=dict(size=15))
        fig.update_layout(height=600)

        st.plotly_chart(fig, use_container_width=True)
        st.success("Plot generated")
