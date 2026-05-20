import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub

# 1. UI Layout Configuration
st.set_page_config(page_title="SmartDoc AI", page_icon="🤖", layout="wide")
st.title("🤖 SmartDoc AI: Intelligent PDF Search Assistant")

# 2. Sidebar Credentials Setup
st.sidebar.header("Authentication")
HF_TOKEN = st.sidebar.text_input("Enter HuggingFace API Token", type="password")

# 3. File Drag-and-Drop Area
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

if uploaded_file is not None and HF_TOKEN:
    # Save the binary stream temporarily onto the disk for parsing
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.info("🔄 ETL Pipeline Active: Extracting text from PDF...")
    
    # 4. Document Extraction (Load)
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    
    # 5. Document Transformation (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    final_documents = text_splitter.split_documents(docs)
    
    # 6. Vector Generation & Local Database Storage
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(final_documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 7. Language Model & Guardrail Prompt Initialization
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
        model_kwargs={"temperature": 0.1, "max_length": 512}
    )
    
    system_prompt = (
        "You are an expert research assistant. Answer the user's question using only the provided context.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 8. Execution Chains Linkage
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    st.success("✅ Text indexed into vector database. Model ready.")
    st.divider()
    
    # 9. Query Interface Segment
    user_query = st.text_input("💬 Ask a question about your uploaded document:")
    
    if user_query:
        with st.spinner("Searching vectors..."):
            response = rag_chain.invoke({"input": user_query})
            
            st.markdown("### 🤖 Answer:")
            st.write(response["answer"])
            
            # 10. Display Data Provenance
            with st.expander("🔍 View Text Segments Used by AI"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', 0) + 1}):**")
                    st.caption(doc.page_content)
                    st.divider()
                    
    # Clean up disk memory
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
        
elif not HF_TOKEN:
    st.warning("⚠️ Input your HuggingFace API Token in the left sidebar to unlock execution features.")