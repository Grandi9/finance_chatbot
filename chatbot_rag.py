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
    page_title="AI Healthcare Advisor",
    page_icon="⚕️",
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
    /* Ensures the main content area fills the available space */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- State Initialization ---
if "age_range" not in st.session_state:
    st.session_state.age_range = None
if "docs" not in st.session_state:
    st.session_state.docs = []
if "messages" not in st.session_state:
    st.session_state.messages = []


# Header
st.markdown('<div class="main-header">AI Healthcare Advisor for New Parents</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Select your baby\'s age in the sidebar, then ask questions about their development and care.</div>', unsafe_allow_html=True)


# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")

    # Age selection
    st.subheader("1. Select Your Baby's Age")
    age_options = ["Please select...", "Less than 3 months", "4-7 months", "8-12 months"]
    st.session_state.age_range = st.selectbox(
        "Age Range:",
        options=age_options,
        index=0 # Default to "Please select..."
    )

    st.divider()

    # AI and Retrieval Settings
    st.subheader("2. AI & Retrieval Settings")

    response_style_map = {
        "Precise": 0.2,
        "Balanced": 0.5,
        "Creative": 0.8
    }
    response_style = st.select_slider(
        "Response Style",
        options=list(response_style_map.keys()),
        value="Balanced"
    )
    temperature = response_style_map[response_style]
    st.caption("Temprature")

    k_value = st.slider("Number of documents to retrieve", 1, 10, 3)
    threshold = st.slider("Similarity threshold", 0.1, 1.0, 0.5, 0.05)
    
    st.divider()
    st.markdown("**About**")
    st.markdown("This advisor provides insights based on pediatric health documents.")
    st.info("This tool is for informational purposes only and is not a substitute for professional medical advice.")


# --- Main Content Area ---
col1, col2 = st.columns([3, 1])

with col2:
    # Set the height of this container to match the chat history container
    document_container = st.container(height=500)
    if st.session_state.docs:
        document_container.markdown(f"**Found {len(st.session_state.docs)} relevant sources:**")
        for i, doc in enumerate(st.session_state.docs, 1):
            score = doc.metadata.get('score', None)
            source = doc.metadata.get('source', 'Unknown')
            score_display = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"

            document_container.markdown(f"**Source {i}** (Relevance: {score_display})")
            document_container.markdown(f"*Origin: {source}*")
            unique_key = f"doc_{i}_{hash(doc.page_content)}"
            document_container.text_area(f"Content {i}", doc.page_content[:200] + "...", height=120, key=unique_key)
    else:
        document_container.markdown("*Retrieved sources will appear here.*")


with col1:
    # Initialize services first to avoid re-initialization on every run
    try:
        if 'vector_store' not in st.session_state:
            if not os.environ.get("PINECONE_API_KEY") or not os.environ.get("OPENAI_API_KEY"):
                st.error("API keys for Pinecone or OpenAI not found. Please check your .env file.")
                st.stop()
            
            with st.spinner("Connecting to services..."):
                index_name = os.environ.get("PINECONE_INDEX_NAME", "docs")
                pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
                index = pc.Index(index_name)
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.environ.get("OPENAI_API_KEY"))
                st.session_state.vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    except Exception as e:
        st.error(f"Error initializing services: {str(e)}")
        st.stop()

    # Initialize chat history
    if not st.session_state.messages:
        st.session_state.messages.append(SystemMessage(content="You are a helpful AI healthcare advisor for new parents..."))

    # Create a container with a fixed height for the chat history
    chat_history_container = st.container(height=500)
    with chat_history_container:
        for message in st.session_state.messages:
            if isinstance(message, SystemMessage): # Don't display system messages
                continue
            avatar = "⚕️" if isinstance(message, AIMessage) else "👤"
            with st.chat_message(message.type, avatar=avatar):
                st.markdown(message.content)

    # User input and processing logic is now managed in a two-step process
    # Step 1: A new prompt is submitted
    if prompt := st.chat_input("Ask a question about your baby's health..."):
        # Age Check
        if not st.session_state.age_range or st.session_state.age_range == "Please select...":
            st.warning("Please select your baby's age from the sidebar before asking a question.")
            st.stop()

        # Add user message to state and immediately rerun to display it
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.rerun()

    # Step 2: If the last message is from the user, trigger the AI response
    # ----------------- REPLACE THE BLOCK ABOVE WITH THIS -----------------

    if st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage):
        last_user_prompt = st.session_state.messages[-1].content
        
        # Use a spinner within the chat container for a better UX
        with chat_history_container:
            with st.spinner("Searching for relevant information..."):
                
                # --- START: METADATA FILTERING LOGIC ---
                search_kwargs = {"k": k_value, "score_threshold": threshold}

                # Map user-friendly age selection from the sidebar to the folder names used as metadata
                age_metadata_map = {
                    "Less than 3 months": "birth-3months",
                    "4-7 months": "4-7months",
                    "8-12 months": "8-12months" # Ensure you have a folder named '8-12months' for this to work
                }
                
                selected_age = st.session_state.age_range
                
                # If a valid age is selected, add the metadata filter to the search
                if selected_age in age_metadata_map:
                    metadata_value = age_metadata_map[selected_age]
                    st.info(f"Filtering search for age group: **{metadata_value}**") # Optional: show user the filter is active
                    search_kwargs['filter'] = {'age_range': metadata_value}
                # --- END: METADATA FILTERING LOGIC ---

                retriever = st.session_state.vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs=search_kwargs, # Pass the dynamically created search_kwargs
                )
                
                docs_with_scores = retriever.invoke(last_user_prompt)
                st.session_state.docs = docs_with_scores
    # ----------------- END OF REPLACEMENT -----------------

            if st.session_state.docs:
                with st.spinner("Thinking..."):
                    docs_text = "".join(d.page_content for d in st.session_state.docs)
                    system_prompt = f"""You are an AI assistant specialized in analyzing healthcare documents for new parents.
                    The user is asking about their baby who is in the **{st.session_state.age_range}** age range. Your answer must be tailored to this specific age group.
                    Your task is to provide accurate, supportive, and helpful answers based on the information in the documents.
                    You are not a medical professional. If a user asks for a diagnosis or medical advice that is not directly supported by the text, you must state that you cannot provide medical advice and recommend they consult a pediatrician or healthcare provider.
                    If no relevant documents are found or the question is outside the scope of the provided documents, 
                    politely inform the user that you don't have enough information to answer accurately for a baby that is {st.session_state.age_range}.
                    When answering, refer to specific parts of the documents when appropriate.
                    Context from documents:
                    {{context}}"""
                    
                    system_prompt_fmt = system_prompt.format(context=docs_text)
                    context_message = SystemMessage(content=system_prompt_fmt)
                    llm_messages = [context_message, HumanMessage(content=last_user_prompt)]
                    
                    llm = ChatOpenAI(model="gpt-4o", temperature=temperature, api_key=os.environ.get("OPENAI_API_KEY"))
                    result = llm.invoke(llm_messages).content
            else:
                result = "I couldn't find any relevant information in the documents to answer your question. Please try rephrasing your query."

        # Append the AI's response and rerun to display everything cleanly
        st.session_state.messages.append(AIMessage(content=result))
        st.rerun()
