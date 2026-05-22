import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Streamlit UI Setup ---
st.set_page_config(page_title="SmartDoc AI", page_icon="🤖", layout="wide")
st.title("🤖 SmartDoc AI: Intelligent PDF Search Assistant")

# --- Sidebar Authentication ---
st.sidebar.header("Authentication")
hf_token = st.sidebar.text_input("Enter HuggingFace API Token", type="password")

if not hf_token:
    st.warning("⚠️ Input your HuggingFace API Token in the left sidebar to unlock execution features.")
else:
    # Set environment variable for LangChain components to access
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

    # --- File Upload Mechanism ---
    uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

    if uploaded_file is not None:
        # Save uploaded file to a temporary directory locally
        temp_file_path = os.path.join("./temp_dir", uploaded_file.name)
        os.makedirs("./temp_dir", exist_ok=True)
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # --- Document Ingestion & Vectorization (ETL Pipeline) ---
        with st.status("🔄 ETL Pipeline Active: Extracting text from PDF...") as status:
            # 1. Load the PDF
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            
            # 2. Split text into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs)
            
            # 3. Download embedding model from Hugging Face
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # 4. Ingest embeddings into a localized FAISS instance
            vectorstore = FAISS.from_documents(final_documents, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            status.update(label="✅ Text indexed into vector database. Model ready.", state="complete")

        # --- Initialize LLM Model ---
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            model_kwargs={"temperature": 0.5, "max_length": 512}
        )

        # --- Standardized LCEL RAG Chain Configuration ---
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        template = """You are an intelligent assistant trained to answer questions based strictly on the provided context.
If the answer cannot be found in the context documents, explicitly state that you don't know.

Context documents:
{context}

Question: {question}
Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)

        # Constructing the RAG chain
        # Added lambda x: x.to_string() to guarantee HuggingFaceHub receives standard text string
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | (lambda x: x.to_string())
            | llm
            | StrOutputParser()
        )

        # --- User Interaction Panel ---
        st.write("---")
        user_query = st.text_input("💬 Ask a question about your uploaded document:")

        if user_query:
            with st.spinner("Analyzing document context and drafting response..."):
                # Run the search-and-generation cycle with a proper dictionary input
                response = rag_chain.invoke({"question": user_query})
                
                # Render output directly on dashboard
                st.subheader("Answer:")
                st.write(response)
