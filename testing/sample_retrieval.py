# import basics
import os
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

# initialize pinecone database
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# set the pinecone index

#index_name = "sample-index"
index_name = "nvidia-transcript"
index = pc.Index(index_name)

# initialize embeddings model + vector store

embeddings = OpenAIEmbeddings(model="text-embedding-3-large",api_key=os.environ.get("OPENAI_API_KEY"))
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# retrieval
# Testing similarity search with cosine similarity
'''

###### add docs to db ##############################
results = vector_store.similarity_search_with_score(
    "what did you have for breakfast?",
    #k=2,
    filter={"source": "tweet"},
)

print("RESULTS:")

for res in results:
    print(f"* {res[0].page_content} [{res[0].metadata}] -- {res[1]}")

'''

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.3},
)
results = retriever.invoke("My name is Krista, and I will be")

print("RESULTS:")

for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

#'''

