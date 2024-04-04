from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
import ollama

CHROMA_PATH = "ChromaDB"
EMBEDDINGG_PATH = "./Models/paraphrase-multilingual-MiniLM-L12-v2" # Multilangage model qui supporte le français
PROMPT_TEMPLATE = """
Réponds à la question uniquement en fonction du contexte suivant
{context}
---
Réponds à la question en te basant sur le contexte ci-dessus : {question}"""

# Initialisationn du LLM
models = ollama.list()
for model in models['models']:
    if model['name'] == 'mistral-latest':
        print('Downloading mistral model...')
        ollama.pull('mistral')

# Model pour les embeddings
hf = HuggingFaceEmbeddings(
    model_name=EMBEDDINGG_PATH
)

# Initialisation de la base de données Chroma
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=hf)

# Recherche de similarité dans la base de données
query = "Que font les ISA ?" # Question de l'utilisateur
context = "" # Contexte de la réponse

results = db.similarity_search(query)
for r in results:
    context += r.page_content + "\n"

# Génération de réponse
response = ollama.generate(model='mistral', prompt=PROMPT_TEMPLATE.format(context=context, question=query))
print(response['response'])