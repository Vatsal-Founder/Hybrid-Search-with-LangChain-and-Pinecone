

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")    
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]= "Hybrid Search with Langchain"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain  # [!code ++]
# from langchain.chains.combine_documents import create_stuff_documents_chain  # [!code --]
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import create_retrieval_chain  # [!code ++]
# from langchain.chains import create_retrieval_chain  # [!code --]
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever  # [!code ++]
# from langchain.chains.history_aware_retriever import create_history_aware_retriever  # [!code --]
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder


INDEX_NAME = "hybrid-search-research"
MAX_QUESTIONS_PER_SESSION = 10
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 500


api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

if not api_key or not groq_api_key:
    st.error("Missing API keys. Set PINECONE_API_KEY and GROQ_API_KEY in .env")
    st.stop()

pc = Pinecone(api_key=api_key)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)
llm = ChatGroq(groq_api_key=groq_api_key, model=LLM_MODEL)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


st.set_page_config(page_title="Hybrid Search RAG", page_icon="🔎", layout="wide")
st.title("🔎 Hybrid Search RAG — Research Papers")
st.caption("Upload PDFs and ask questions • Powered by LangChain + Pinecone + Groq")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "store" not in st.session_state:
    st.session_state.store = {}
if "retriever_ready" not in st.session_state:
    st.session_state.retriever_ready = False


with st.sidebar:
    st.header("📄 Upload Research Papers")
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type="pdf",
        accept_multiple_files=True,
    )

    session_id = st.text_input("Session ID", value="default_session")

    remaining = MAX_QUESTIONS_PER_SESSION - st.session_state.question_count
    st.metric("Questions remaining", f"{remaining}/{MAX_QUESTIONS_PER_SESSION}")

    if remaining <= 0:
        st.warning("Free demo limit reached. Clone the repo to run unlimited.")

    if uploaded_files:
        with st.spinner("Chunking & indexing..."):
            documents = []
            for uploaded_file in uploaded_files:
                # Safe temp file handling (no overwrites)
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                documents.extend(docs)
                os.unlink(tmp_path)  # cleanup

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(documents)
            split_texts = [doc.page_content for doc in splits]

            # Fit BM25 encoder on the corpus
            bm25_encoder = BM25Encoder()
            bm25_encoder.fit(split_texts)
            bm25_encoder.dump("bm25_values.json")

            # Build retriever & upsert
            retriever = PineconeHybridSearchRetriever(
                embeddings=embeddings,
                sparse_encoder=bm25_encoder,
                index=index,
            )
            retriever.add_texts(split_texts)

            st.session_state.retriever = retriever
            st.session_state.retriever_ready = True

        st.success(f"Indexed {len(split_texts)} chunks from {len(uploaded_files)} PDF(s)")


if st.session_state.retriever_ready:
    retriever = st.session_state.retriever

    # History-aware retriever for follow-up questions
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given a chat history and the latest user question which might "
            "reference context in the chat history, formulate a standalone "
            "question which can be understood without the chat history. "
            "Do NOT answer the question, just reformulate it if needed "
            "and otherwise return it as is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # QA chain
    system_prompt = (
        "You are an intelligent research assistant for answering questions "
        "about research papers. Use the following retrieved context to answer "
        "the question. If you don't know the answer, say so.\n\n"
        "Guidelines:\n"
        "1. Summarize the key information relevant to the query.\n"
        "2. Provide a structured explanation (bullet points if needed).\n"
        "3. Keep language professional, neutral, and clear.\n"
        "4. Suggest follow-up questions if relevant.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Ask a question about your papers..."):
        if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
            st.error("Free demo limit reached. Clone the repo to run unlimited.")
        else:
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching & generating..."):
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": session_id}},
                    )
                    answer = response["answer"]
                    st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.question_count += 1

else:
    st.info("👈 Upload research papers and click **Index Papers** to get started.")
