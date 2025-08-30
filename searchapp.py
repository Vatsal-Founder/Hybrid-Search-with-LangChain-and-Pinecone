import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()


os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import PyPDFDirectoryLoader,PyPDFLoader
from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from pinecone_text.sparse import BM25Encoder





index_name="hybrid-search-research"

api_key=os.getenv("PINECONE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


## initialize the Pinecone client
pc=Pinecone(api_key=api_key)

#create the index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # dimensionality of dense model
        metric="dotproduct",  # sparse values supported only for dotproduct
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index=pc.Index(index_name)




st.title("Hybrid Search on Research Papers: with LangChain and Pinecone")
st.write("Upload PDFs and chat with their research paper")


groq_api_key = st.text_input("Enter your Groq API key:", type="password")

llm=ChatGroq(groq_api_key=groq_api_key, model="llama3-70b-8192")

uploaded_files=st.file_uploader("Choose A Research Paper", type="pdf", accept_multiple_files=True)

if uploaded_files:
    session_id= st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store={}
    
    documents=[]
    for uploaded_file in uploaded_files:
        tempdf=f"./temp.pdf"
        with open(tempdf, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
        
        loader= PyPDFLoader(tempdf)
        docs = loader.load()
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

    splits = [doc.page_content for doc in splits]
    
    bm25_encoder=BM25Encoder()
    bm25_encoder.fit(splits)
    ## store the values to a json file
    bm25_encoder.dump("paper_values.json")
    
    bm25_encoder = BM25Encoder().load("paper_values.json")
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    retriever=PineconeHybridSearchRetriever(embeddings=embeddings,sparse_encoder=bm25_encoder,index=index)
    retriever.add_texts(splits)


    contextualize_q_system_prompt=(
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    
    history_aware_retriever= create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    ## Answer question

    # Answer question
    system_prompt = (
            "You are an intelligent, reliable research assistant for answering questions regarding research paper tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question.  If you don't know the answer, say that you "
            "don't know. Use this format (only if relevant):"
            "1. Summarize the key information relevant to the query."
            "2. Provide a structured explanation or answer (bullet points or numbered steps if needed)"
            "3. Keep language professional, neutral, and clear."
            "4.Add follow-up suggestions if the user may want to explore further."
            #"Use three sentences maximum and keep the "
            #"answer concise."
            "\n\n"
            "{context}"
        )
    qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain=create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]
    

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    user_input = st.text_input("Enter your questions:")
    if user_input:
        session_history=get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id":session_id}
            },  # constructs a key "abc123" in `store`.
        )
        st.write(st.session_state.store)
        st.write("Assistant:  ", response['answer'])
        st.write("Chat History:", session_history.messages)
else:
    st.warning("Please upload the research paper")