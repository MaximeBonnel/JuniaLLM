from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil

DATA_PATH_PDF = "./Documents/pdf"
DATA_PATH_MD = "./Documents/md"
CHROMA_PATH = "ChromaDB"
EMBEDDINGG_PATH = "./Models/paraphrase-multilingual-MiniLM-L12-v2" # Multilangage model qui supporte le français

# Model pour les embeddings
hf = HuggingFaceEmbeddings(
    model_name=EMBEDDINGG_PATH
)

# Supprimer / Nettoyer la base de données Chroma si elle existe
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)
    print(f"Delete {CHROMA_PATH}.")

def generate_data_store():
    for pdf in os.listdir(DATA_PATH_PDF):
        # Vérifier si le fichier est un document .pdf
        if pdf.endswith(".pdf"):
            print(f"\nProcessing {pdf}...")
            documents = load_pdf(DATA_PATH_PDF + "/" + pdf)
            chunks = split_text(documents)
            save_to_chroma(chunks)
    
    print(f"\nProcessing {DATA_PATH_MD}...")
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

# Lire les pdf
def load_pdf(path):
    loader = PyPDFLoader(path)
    documents = loader.load()
    for document in documents:
        document.page_content = document.page_content.replace("\n", " ")
    return documents

# Lire les fichiers .md
def load_documents():
    loader = DirectoryLoader(DATA_PATH_MD, glob="*.md")
    documents = loader.load()
    return documents

# Diviser le texte en morceaux
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks

# Les sauvegarder dans la base de données Chroma
def save_to_chroma(chunks: list[Document]):
    db = Chroma.from_documents(chunks, hf, persist_directory=CHROMA_PATH)
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

generate_data_store()