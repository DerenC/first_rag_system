import streamlit as st
import backend as rag

st.title("My First RAG System")
st.subheader("")
st.divider()

with st.sidebar:
    st.header("Upload context")
    knowledge_text = st.text_area("Enter text of knowledge here", height=150)

    if st.button("Upload to vector store"):
        if knowledge_text:
            with st.spinner("Processing"):
                rag.ingest_text(knowledge_text)
                st.success("Uploaded")
        else:
            st.warning("There is no text. Please enter some text.")

st.header("Ask anything to the chat from our Knowledge Base")

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
