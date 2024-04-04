# JuniaLLM

## Setup
1)
    ```pip install -r requirements.txt```
2) Install Ollama
    https://ollama.com/download
3) Install the embedding model
    Clone "paraphrase-multilingual-MiniLM-L12-v2" from HuggingFace in "models" folder
    https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    ```git clone https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2```

---

## Run
#### Creation of the DB
```python ./CreateDB.py```

#### Run the RAG
```python ./RAG.py```