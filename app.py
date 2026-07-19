import streamlit as st

from retrieve import search
from utils.reranker import rerank
from generator import generate_answer

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Chatbot")
st.caption("Ask questions about your PDF documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
question = st.chat_input("Ask a question about your PDFs...")

if question:

    # User message
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            docs = search(question)
            docs = rerank(question, docs)

            context = "\n\n".join(
                doc.page_content for doc in docs[:3]
            )

            answer = generate_answer(question, context)

        st.markdown(answer)

        with st.expander("📚 Sources"):

            for doc in docs[:3]:

                st.markdown(
                    f"""
**File:** {doc.metadata.get('source')}

**Page:** {doc.metadata.get('page',0)+1}
                    """
                )

    st.session_state.messages.append(
        {"role":"assistant","content":answer}
    )