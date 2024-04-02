from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma

DATA_PATH = "./documents"
CHROMA_PATH = "JuniaLLM"
MODEL_NAME = "openlm-research/open_llama_3b_v2"

model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"\nSplit {len(documents)} documents into {len(chunks)} chunks.")

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, hf, persist_directory=CHROMA_PATH
    )

    # Save the DB to disk.
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

    # Query the DB.
    query = "Suivre une formation en informatique."
    print(db.similarity_search(query))


generate_data_store()