from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
import gradio as gr
import ollama
import os

# Paramètres
CHROMA_PATH = "ChromaDB"
EMBEDDINGG_PATH = "./Models/paraphrase-multilingual-MiniLM-L12-v2" # Multilangage model qui supporte le français
PROMPT_TEMPLATE = """
Contexte :
{context}
---
Réponds à la question en Français en te basant sur le contexte ci-dessus : {question}"""

# Initialisationn du LLM
models = ollama.list()
mistral = False
for model in models['models']:
    if model['name'] == 'mistral-latest':
        mistral = True

if not mistral:
    print('Downloading mistral model...')
    os.system("ollama pull mistral")

# Model pour les embeddings
hf = HuggingFaceEmbeddings(
    model_name=EMBEDDINGG_PATH
)

# Initialisation de la base de données Chroma
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=hf, collection_metadata={"hnsw:space": "cosine"})

def RAG(query):
    # Recherche de similarité dans la base de données
    context = "" # Contexte de la réponse

    result = db.similarity_search_with_score(query)[0]
    if result[1] > 9:
        context = "Aucun résultat correspondant trouvé"
        print(f"Aucune donnée trouvée, similarité : {result[1]}")
    else:
        context = result[0].page_content

    # Génération de réponse
    response = ollama.generate(model='mistral', prompt=PROMPT_TEMPLATE.format(context=context, question=query))
    return response['response']


# Créer une interface Gradio
gr.Interface(
    fn=RAG,
    inputs=gr.Textbox(lines=2, label="Question", placeholder="Quel est votre question ?"),
    outputs=gr.Textbox(lines=2, label="Réponse"),
    description="Posez vos questions à JUNIA LLM pour tout savoir sur Junia !",
    title="Junia LLM",
    allow_flagging="never"
).launch()