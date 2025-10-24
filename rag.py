from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import login
import os

# ------------------- Config -------------------
load_dotenv()
CHUNK_SIZE = 2000
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
VECTOR_STORE_DIR = Path(__file__).parent / "resources/vector_store"
COLLECTION_NAME = "real_estate"
hf_token = os.getenv("HUGGINGFACE_API_TOKEN")

if not hf_token:
    raise ValueError("Hugging Face API token not found. Set HUGGINGFACE_API_TOKEN in .env file.")

login(token=hf_token)

llm = None
vector_store = None

# ------------------- Acronym Expansion -------------------
ACRONYM_MAP = {
    "AIML": "Artificial Intelligence and Machine Learning",
    "AI": "Artificial Intelligence",
    "ML": "Machine Learning",
}

def expand_acronyms(query: str) -> str:
    """Expand common acronyms so retriever can match better."""
    for short, full in ACRONYM_MAP.items():
        if short.lower() in query.lower():
            query = query.replace(short, full)
    return query

# ------------------- Initialization -------------------
def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,   # factual answers
            max_tokens=500
        )

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTOR_STORE_DIR)
        )

# ------------------- Process URLs -------------------
def process_urls(urls):
    """Scrape data from the URLs and store in the vector DB."""

    yield "Initializing components...✅"
    initialize_components()
    vector_store.reset_collection()

    yield "Resetting vector store...✅"
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    if not data:
        yield "⚠️ No data could be extracted from the provided URLs."
        return

    yield "Splitting data into chunks...✅"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
    )
    docs = text_splitter.split_documents(data)

    if not docs:
        yield "⚠️ No valid documents after splitting. Nothing to add to vector store."
        return

    yield f"Adding {len(docs)} chunks to vector store...✅"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield "Done adding docs to vector database...✅"

# ------------------- QA -------------------
def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is not initialized")

    # Expand acronyms like AIML → Artificial Intelligence and Machine Learning
    query = expand_acronyms(query)

    retriever = vector_store.as_retriever(
        search_type="mmr",                # Maximal Marginal Relevance
        search_kwargs={"k": 8, "fetch_k": 20}
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"   # directly stuff docs into prompt
    )

    result = chain.invoke({"query": query})
    answer = result.get("result", "No answer found.")
    sources = []

    if "source_documents" in result:
        sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]

    return answer, sources

# ------------------- Main -------------------
if __name__ == "__main__":
    urls = ["https://en.wikipedia.org/wiki/Artificial_intelligence"]

    for status in process_urls(urls):
        print(status)

    answer, sources = generate_answer("AIML introduction")
    print("\nAnswer:", answer)
    print("Sources:", sources)
