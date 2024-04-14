from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
import gradio as gr
import ollama
import os

# Paramètres
CHROMA_PATH = "ChromaDB"
EMBEDDINGG_PATH = "./Models/paraphrase-multilingual-MiniLM-L12-v2" # Multilangage model qui supporte le français
OLLAMAMODEL = "JuniaMistral" # Mistral sur ollama
PROMPT_TEMPLATE = """
Répondez à la question en vous basant uniquement sur le contexte suivant :

{context}

---

Répondez à la question en vous basant sur le contexte ci-dessus : {question}
"""

# Initialisationn du LLM
os.system("ollama pull mistral")
os.system("ollama create JuniaMistral -f ./Modelfile")
os.system("cls")
print("Adresse IP de la machine :")
os.system("ipconfig | findstr IPv4") # Afficher l'adresse IP de la machine

# Model pour les embeddings
hf = HuggingFaceEmbeddings(
    model_name=EMBEDDINGG_PATH
)

# Initialisation de la base de données Chroma
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=hf, collection_metadata={"hnsw:space": "l2"})

def RAG(query):
    # Recherche de similarité dans la base de données
    context = "" # Contexte de la réponse

    results = db.similarity_search_with_score(query)
    for result in results:
        context += result[0].page_content + "\n"
    print(context)

    # Génération de réponse
    response = ollama.generate(model=OLLAMAMODEL, prompt=PROMPT_TEMPLATE.format(context=context, question=query))
    return response['response']


# Créer une interface Gradio
gr.Interface(
    fn=RAG,
    inputs=gr.Textbox(lines=2, label="Question", placeholder="Quel est votre question ?"),
    outputs=gr.Textbox(lines=2, label="Réponse"),
    description="Posez vos questions à JUNIA LLM pour tout savoir sur Junia !",
    title="Junia LLM",
    allow_flagging="never"
).launch(share=False, server_name="0.0.0.0", server_port=8080)