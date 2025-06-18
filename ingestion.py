# import basics
import os
import time
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec, PineconeException

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# documents
# CHANGED: Swapped PyPDFDirectoryLoader for the more general DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("--- Starting Pinecone Ingestion Script ---")

# 1. Load Environment Variables and Basic Checks
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1") # Default region

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables. Please set it in your .env file.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME not found in environment variables. Please set it in your .env file.")

print(f"Environment variables loaded. Index Name: {PINECONE_INDEX_NAME}, Region: {PINECONE_REGION}")

# 2. Initialize Pinecone Connection
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("Pinecone client initialized successfully.")
except PineconeException as e:
    print(f"Error initializing Pinecone client: {e}")
    print("Please check your PINECONE_API_KEY and network connection.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during Pinecone client initialization: {e}")
    exit()

# 3. Check and Create Pinecone Index
print(f"Checking for existing index: {PINECONE_INDEX_NAME}...")
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating a new one...")
    try:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=3072, # Dimension for text-embedding-3-large
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
        )
        print(f"Index creation initiated for '{PINECONE_INDEX_NAME}'. Waiting for it to become ready...")
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            print("Index not ready yet, waiting...")
            time.sleep(5)
        print(f"Index '{PINECONE_INDEX_NAME}' is ready.")
    except PineconeException as e:
        print(f"Error creating Pinecone index: {e}")
        print("Please check your Pinecone project limits, index name, and region.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during Pinecone index creation: {e}")
        exit()
else:
    print(f"Index '{PINECONE_INDEX_NAME}' already exists and is ready.")

index = pc.Index(PINECONE_INDEX_NAME)

# 4. Initialize Embeddings Model and Vector Store
print("Initializing OpenAI embeddings model and Pinecone vector store...")
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    print("Embeddings model and vector store initialized.")
except Exception as e:
    print(f"Error initializing embeddings or vector store: {e}")
    print("Please check your OPENAI_API_KEY and ensure 'text-embedding-3-large' model is accessible.")
    exit()

# 5. Load Documents
documents_folder = "documents/"
# CHANGED: Updated print statement to be generic for any document type.
print(f"Loading text documents from '{documents_folder}' and its subdirectories...")
try:
    # CHANGED: Replaced PyPDFDirectoryLoader with DirectoryLoader.
    # The glob="**/*.txt" pattern recursively finds all .txt files in all subfolders.
    loader = DirectoryLoader(documents_folder, glob="**/*.txt", show_progress=True, use_multithreading=True)
    raw_documents = loader.load()
    
    if not raw_documents:
        # CHANGED: Updated warning message for .txt files.
        print(f"No .txt documents found in '{documents_folder}' or its subdirectories. Please ensure text files are present.")
        exit()
    else:
        # CHANGED: Updated success message.
        print(f"Successfully loaded {len(raw_documents)} raw text documents.")
        
        # --- CRITICAL DIAGNOSTIC STEP ---
        # CHANGED: Modified inspection log to remove "Page" since .txt files don't have pages.
        print("\n--- Inspecting first 3 raw document contents ---")
        for i, doc in enumerate(raw_documents[:3]):
            print(f"Document {i+1} (Source: {doc.metadata.get('source', 'N/A')}):")
            print(f"  Content length: {len(doc.page_content)} characters")
            snippet = doc.page_content.replace('\n', ' ').strip()
            print(f"  Content snippet: \"{snippet[:500]}...\"")
            if len(doc.page_content.strip()) == 0:
                print("  WARNING: This document's page_content appears empty or only whitespace!")
            print("-" * 50)
        print("--- End of raw document inspection ---\n")

except Exception as e:
    print(f"Error loading documents: {e}")
    # CHANGED: Updated error message for .txt files.
    print(f"Please ensure the '{documents_folder}' directory exists and contains valid .txt files.")
    exit()

# ----------------- REPLACE THIS SECTION -----------------

# 6. Add Folder-Based Metadata and Split Documents
print("Adding metadata based on folder structure...")

# This loop adds the parent folder name as metadata to each document
for doc in raw_documents:
    # The 'source' metadata field contains the full path to the file
    file_path = doc.metadata.get('source', '')
    
    # We normalize the path to handle different OS separators (e.g., / vs \)
    normalized_path = os.path.normpath(file_path)
    
    # Split the path into components
    path_parts = normalized_path.split(os.sep)
    
    # The parent folder name should be the second-to-last part
    # e.g., ['documents', 'birth-3months', 'some-file.txt'] -> 'birth-3months'
    if len(path_parts) > 1:
        age_category = path_parts[-2]
        doc.metadata['age_range'] = age_category
        
print("Metadata added. Example from first document:", raw_documents[0].metadata)

print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

documents = text_splitter.split_documents(raw_documents)

if not documents:
    print("No chunks were generated from the documents. This indicates an issue with the extracted text or chunking parameters.")
    exit()
print(f"Generated {len(documents)} document chunks.")

# ----------------- END OF REPLACEMENT -----------------

# 7. Generate Unique and Informative IDs
print("Generating unique IDs for document chunks...")
uuids = []
for i, doc in enumerate(documents):
    file_path = doc.metadata.get('source', 'unknown_file')
    file_name = os.path.basename(file_path)
    
    # CHANGED: Removed page_number as .txt files do not have page metadata.
    # page_number = doc.metadata.get('page', 'no_page')
    
    clean_file_name = "".join(c if c.isalnum() else "_" for c in file_name).strip("_")

    # CHANGED: Simplified the unique_id format to exclude the non-existent page number.
    unique_id = f"{clean_file_name}_chunk_{i}"
    uuids.append(unique_id)

print(f"Generated {len(uuids)} unique IDs. Example ID: {uuids[0] if uuids else 'N/A'}")

# 8. Add Documents to Pinecone Database
print("Adding document chunks to Pinecone database. This may take a while for many documents...")
try:
    vector_store.add_documents(documents=documents, ids=uuids)
    print(f"Successfully added {len(documents)} document chunks to Pinecone index '{PINECONE_INDEX_NAME}'.")
except PineconeException as e:
    print(f"Error adding documents to Pinecone (PineconeException): {e}")
    print("This could be due to API rate limits, network issues, or issues with embeddings generation or Pinecone service.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while adding documents to Pinecone: {e}")
    print("This might indicate an issue with the embeddings model, data format, or connectivity.")
    exit()

# 9. Verify Ingestion by Checking Index Stats
print(f"Verifying ingestion by checking index stats for '{PINECONE_INDEX_NAME}'...")
try:
    # Adding a small delay to allow the index stats to update.
    time.sleep(10) 
    index_stats = index.describe_index_stats()
    print("\n--- Pinecone Index Stats ---")
    print(f"Index Name: {PINECONE_INDEX_NAME}")
    print(f"Dimension: {index_stats.dimension}")
    print(f"Total Vector Count: {index_stats.total_vector_count}")
    
    if index_stats.namespaces:
        print("Namespaces found:")
        for namespace, stats in index_stats.namespaces.items():
            print(f"  Namespace: '{namespace}', Vector Count: {stats.vector_count}")
    else:
        print("No specific namespaces found (likely using the default implicit namespace).")

    if index_stats.total_vector_count > 0:
        print("\nVerification successful! Records should now be visible in your Pinecone UI.")
    else:
        print("\nVerification failed: No vectors reported in the index stats. Please review previous logs for errors.")

except PineconeException as e:
    print(f"Error fetching Pinecone index stats: {e}")
    print("Could not verify ingestion programmatically. Please check your Pinecone connection and API key.")
except Exception as e:
    print(f"An unexpected error occurred during stats check: {e}")

print("--- Script Finished ---")
