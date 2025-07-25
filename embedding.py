import requests
import numpy as np
import faiss
import ollama

def get_query_embedding(query):
    result = ollama.embed(
        model="nomic-embed-text:v1.5",
        input=query,
    )

    if "embeddings" not in result:
        raise KeyError(f"'embeddings' key not found in Ollama response. Response: {result}")

    return np.array(result["embeddings"][0], dtype=np.float32)

def retrieve_relevant_context(query, index, document_text):
    query_emb = get_query_embedding(query)
    D, I = index.search(np.expand_dims(query_emb, axis=0), k=1)
    return document_text

def text_to_vdb(text):
    chunks = text.split("\n\n")
    
    embeddings = []
    for chunk in chunks:
        res = ollama.embed(
            model="nomic-embed-text:v1.5",
            input=chunk,
        )
        if "embeddings" not in res:
            continue
        embeddings.append(res["embeddings"][0])

    if not embeddings:
        raise RuntimeError("No embeddings were generated.")
    
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))

    return index, chunks

if __name__ == "__main__":
    index = text_to_vdb("hi")