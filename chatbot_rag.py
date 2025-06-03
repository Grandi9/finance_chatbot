import streamlit as st
import os
import time
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Configure page
st.set_page_config(
    page_title="Document Analysis Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .chat-container {
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .user-msg {
        background-color: #E3F2FD;
    }
    .ai-msg {
        background-color: #F5F5F5;
    }
    .metadata {
        font-size: 0.8rem;
        color: #757575;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about your documents using AI</div>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Index selection
    index_from_env = os.environ.get("PINECONE_INDEX_NAME", "docs")
    index_name = st.text_input("Pinecone Index Name", value=index_from_env)
    
    # Model selection
    model_name = st.selectbox(
        "Model",
        ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
        index=0
    )
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)

    # Retrieval settings
    st.subheader("Retrieval Settings")
    k_value = st.slider("Number of documents to retrieve", 1, 10, 3)
    threshold = st.slider("Similarity threshold", 0.1, 1.0, 0.5, 0.05)
    
    st.divider()
    st.markdown("**About**")
    st.markdown("This chatbot analyzes documents stored in your vector database.")

# Main content area (divide into two columns)
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("Retrieved Documents")
    document_container = st.container()
    document_container.markdown("*No documents retrieved yet*")

with col1:
    # Check for API keys
    api_errors = []
    if not os.environ.get("PINECONE_API_KEY"):
        api_errors.append("PINECONE_API_KEY not found in .env file")
    if not os.environ.get("OPENAI_API_KEY"):
        api_errors.append("OPENAI_API_KEY not found in .env file")
    
    if api_errors:
        for error in api_errors:
            st.error(error)
        st.stop()
    
    # initialize pinecone database
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index(index_name)
    except Exception as e:
        st.error(f"Error connecting to Pinecone: {str(e)}")
        st.stop()
    
    # Initialize embeddings model + vector store
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.environ.get("OPENAI_API_KEY"))
        vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        st.stop()
    
    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(SystemMessage(content="You are a document analysis expert who can answer questions about documents in the database."))
    
    # Display chat messages
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user", avatar="👤"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message.content)
        elif isinstance(message, SystemMessage):
            # Don't display system messages to the user
            pass
    
# User input
prompt = st.chat_input("Ask a question about your documents")
    
# Process user input
if prompt:
    # Add user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # Show loading indicator
    with st.spinner("Searching for relevant documents..."):
        # Create and invoke the retriever
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k_value, "score_threshold": threshold},
        )
        
        docs_with_scores = vector_store.similarity_search_with_score(prompt, k=k_value)
        docs = []
        for doc, score in docs_with_scores:
            doc.metadata['score'] = score
            docs.append(doc)
    
    #Retrieved documents in the sidebar
    with document_container:
        if docs:
            document_container.markdown(f"**Found {len(docs)} relevant documents:**")
            for i, doc in enumerate(docs, 1):
                score = doc.metadata.get('score', None)
                source = doc.metadata.get('source', 'Unknown')
                
                if isinstance(score, (int, float)):
                    score_display = f"{score:.2f}"
                else:
                    score_display = "N/A"
                
                document_container.markdown(f"**Doc {i}** (Score: {score_display})")
                document_container.markdown(f"*Source: {source}*")
                document_container.text_area(f"Content {i}", doc.page_content[:150] + "...", height=100, key=f"doc_{i}")
        else:
            document_container.markdown("*No relevant documents found*")
    
    docs_text = "".join(d.page_content for d in docs)
    
    # Create system prompt
    system_prompt = """You are an AI assistant specialized in analyzing documents. 
    Your task is to provide accurate and helpful answers based on the information in the documents.
    
    If no relevant documents are found or the question is outside the scope of the provided documents, 
    politely inform the user that you don't have enough information to answer accurately.
    
    When answering, refer to specific parts of the documents when appropriate.
    
    Context from documents:
    {context}"""
    
    # Populate system prompt with context
    system_prompt_fmt = system_prompt.format(context=docs_text)
    
    # Add system message with context
    context_message = SystemMessage(content=system_prompt_fmt)
    
    # Create message list for the LLM
    llm_messages = [msg for msg in st.session_state.messages if isinstance(msg, (HumanMessage, AIMessage))]
    llm_messages.insert(0, context_message)
    
    # Show thinking indicator
    with st.spinner("Thinking..."):
        # Initialize the LLM
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Generate response
        result = llm.invoke(llm_messages).content
    
    # Display assistant response
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(result)
    
    # Save the assistant message
    st.session_state.messages.append(AIMessage(content=result))
