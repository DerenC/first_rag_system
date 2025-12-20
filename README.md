# My First RAG System

## Features

* **Vector Store**: Uses MongoDB Atlas Vector Search to store and retrieve document embeddings.
* **Embedding Model**: Uses either Google's `model/embeddings-001` or Hugging Face's `sentence-transformers/all-mpnet-base-v2`
* **LLM**: Uses Google's `gemini-2.5-flash` model for generating responses.
* **Framework**: Built using [LangChain](https://www.langchain.com/) and [Streamlit](https://streamlit.io/).

## Prerequisites

Before running the application, ensure you have the following:

1.  **Python 3.8+** installed.
2.  A **MongoDB Atlas** cluster with a vector search index configured.
3.  A **Google AI Studio** with an API key.

## Installation

1.  **Create and activate a virtual environment (recommended):**
    ```bash
    pip install uv # if needed
    uv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2.  **Install dependencies**
    ```bash
    uv pip install -r requirements.txt
    ```

## Configuration & setup

This project uses `streamlit.secrets` for managing sensitive configuration.

1.  **Create a secrets file:**
    Create a folder named `.streamlit` in your project root and add a file named `secrets.toml`:

    ```toml
    # .streamlit/secrets.toml
    MONGO_URI = "mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"
    GEMINI_API_KEY="your-google-api-key"
    ```

3.  **MongoDB Atlas Setup:**

    * **Database Name:** `vector_store_database`
    * **Collection Name:** `vector_store_collection`
    * **Atlas Vector Search Index Name:** `vector_index_ghw_aiml`

    (These 3 names will be used in `backend.py`)

    *Example Index Definition:*
    ```json
    {
      "fields": [
        {
          "numDimensions": 768,
          "path": "embedding",
          "similarity": "cosine",
          "type": "vector"
        }
      ]
    }
    ```

## Usage

The core logic is contained in `backend.py`. You can import these functions into a Streamlit frontend or another Python script.

### Key Functions

* `ingest_text(text_content)`: Takes a string of text, creates a document, calculates its embedding, and stores it in MongoDB.
* `get_rag_response(query)`: Performs a similarity search for the top 3 relevant documents in MongoDB and uses the Gemini LLM to answer the `query` based on that context.

## Future features to be implemented

File uploading:
* [ ] Supports uploading & parsing of pdf and docx files
* [ ] Supports uploading & parsing of html files
* [ ] Supports OCR of uploaded images
* [ ] After OCR, connect to device camera

Other sources of information:
* [ ] Web crawling & web scraping of a user-specified URL
* [ ] Implement a POST API endpoint to allow programmatical uploading

Management of uploaded files & texts (vector store):
* [ ] Keeping track of uploaded texts (might need to involve files' fingerprints), to prevent uploading of duplicated texts
* [ ] Allow users to search the uploaded texts (cloud use Elasticsearch)
* [ ] Allow users to specify which texts to delete from the vector store
