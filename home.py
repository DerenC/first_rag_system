import streamlit as st
import backend as rag

def success_fade(message, seconds=3):
    st.markdown(
        f"""
        <style>
        @keyframes fadeOut {{
            0% {{ opacity: 1; }}
            80% {{ opacity: 1; }}
            100% {{ opacity: 0; }}
        }}

        .fade-success {{
            animation: fadeOut {seconds}s forwards;
            background-color: #d4edda;
            color: #155724;
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            margin-top: 1rem;
        }}
        </style>

        <div class="fade-success">
            ✅ {message}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.header("Simple RAG System")

upload_tab, query_tab = st.tabs(["Upload Knowledge", "Ask Question"])

with upload_tab:
    st.header("Uploading information allows the RAG system to back its responses with sources")

    knowledge_text = st.text_area("Enter text of knowledge here", height=150)

    if st.button("Upload to vector store", key="upper-upload-button"):
        if knowledge_text:
            with st.spinner("Processing"):
                rag.ingest_text(knowledge_text)
                success_fade("Uploaded")
        else:
            st.warning("There is no text. Please enter some text.")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["txt", "md"],
        # type=["pdf", "txt", "docx"],  #TODO
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.write(f"{len(uploaded_files)} file(s) uploaded")

        # for file in uploaded_files:
        #     st.write("Filename:", file.name)
        #     st.write("File type:", file.type)
        #     st.write("File size (bytes):", file.size)

        if st.button("Upload to vector store", key="lower-upload-button"):
            with st.spinner("Processing"):
                texts = []
                for file in uploaded_files:
                    content = file.read().decode("utf-8").strip()
                    texts.append(content)
                rag.ingest_multiple_texts(texts)
                success_fade("Uploaded")


with query_tab:
    st.header("Ask any question, the RAG system will give you evidence-based responses")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask anything you want to know")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("IS THINKING"):
                response_data = rag.get_rag_response(prompt)
                answer = response_data["answer"]
                sources = response_data["sources"]

                st.markdown(answer)

                with st.expander("Sources"):
                    for i, source in enumerate(sources):
                        st.markdown(f"**Source {i+1}:** {source.page_content}")

                st.session_state.messages.append({"role": "assistant", "content": answer})
