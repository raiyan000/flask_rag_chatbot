import chromadb
from sentence_transformers import SentenceTransformer
model=SentenceTransformer('all-MiniLM-L6-v2')

# chroma_client=chromadb.Client()
chroma_client=chromadb.PersistentClient(path="vector_db")
collection=chroma_client.get_or_create_collection(name="documents",metadata={"hnsw:space":"cosine"})
from config.config import client

# Debug: show number of docs stored on startup
print("📦 ChromaDB startup check:")
try:
    print("Total documents loaded:", collection.count())
except:
    print("⚠️ Could not count documents")
def store_in_chunks(chunks):
    try:
        existing = collection.get()
        existing_texts = set(existing['documents']) if existing and 'documents' in existing else set()
    except Exception as e:
        print("Warning: Could not fetch existing documents:", e)
        existing_texts = set()
    
    ids = []
    texts = []
    embeddings = []
    metadatas = []

    chunk_id = 1
    for chunk in chunks:
        text = chunk['chunks']
        filename = chunk['filename']

        if text in existing_texts:
            print(f"Skipping duplicate chunk from {filename}")
            continue

        embedding = model.encode(text).tolist()
        ids.append(f"chunk_{chunk_id}")
        texts.append(text)
        embeddings.append(embedding)
        metadatas.append({"filename": filename})
        chunk_id += 1

    if texts:
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"Stored {len(texts)} new chunks in ChromaDB.")
    else:
        print("No new chunks to add (all were duplicates).")

def retrieve_relevant_chunks(query, top_k=3, strong_match_threshold=0.40):
    print(f"\n💬 [INFO] Received query: {query}")

    if not query or not query.strip():
        return []

    try:
        query_embedding = model.encode(query).tolist()
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 5,
        include=["documents", "metadatas", "distances"]
    )

    if not results or not results["documents"][0]:
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    combined = list(zip(docs, metas, distances))
    combined_sorted = sorted(combined, key=lambda x: x[2])

    best_doc, best_meta, best_distance = combined_sorted[0]
    best_filename = best_meta.get("filename", "Unknown")

    print("📏 Distances returned:", distances)

    # Strong match → ONLY ONE
    if best_distance < strong_match_threshold:
        print("🎯 Strong match — Returning ONLY best chunk")
        return [{
            "filename": best_filename,
            "distance": best_distance,
            "content": best_doc
        }]

    # Filter by similar content or same file
    filtered = [
        (doc, meta, dist)
        for doc, meta, dist in combined_sorted
        if meta.get("filename") == best_filename
        or dist < best_distance + 0.40
    ]

    if not filtered:
        filtered = combined_sorted[:top_k]

    final = []
    for doc, meta, dist in filtered[:top_k]:
        final.append({
            "filename": meta.get("filename", "Unknown"),
            "distance": dist,
            "content": doc
        })

    return final

# def retrieve_relevant_chunks(query, top_k=3, strong_match_threshold=0.80):
#     print(f"\n💬 [INFO] Received query: {query}")

#     if not query or not query.strip():
#         return []

#     try:
#         query_embedding = model.encode(query).tolist()
#     except Exception as e:
#         print(f"❌ Embedding error: {e}")
#         return []

#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k * 5,
#         include=["documents", "metadatas", "distances"]
#     )

#     if not results or not results["documents"][0]:
#         return []

#     docs = results["documents"][0]
#     metas = results["metadatas"][0]
#     distances = results["distances"][0]

#     combined = list(zip(docs, metas, distances))
#     combined_sorted = sorted(combined, key=lambda x: x[2])

#     best_doc, best_meta, best_distance = combined_sorted[0]
#     best_filename = best_meta.get("filename", "Unknown")

#     print("📏 Distances returned:", distances)

#     # STRONG MATCH → ONLY ONE
#     if best_distance < strong_match_threshold:
#         return [{
#             "text": best_doc,
#             "filename": best_filename,
#             "distance": best_distance
#         }]

#     # FILTER BY FILE OR SIMILARITY WINDOW
#     filtered = [
#         (doc, meta, dist)
#         for doc, meta, dist in combined_sorted
#         if meta.get("filename") == best_filename
#            or dist < best_distance + 0.40
#     ]

#     if not filtered:
#         filtered = combined_sorted[:top_k]

#     final = []
#     for doc, meta, dist in filtered[:top_k]:
#         final.append({
#             "text": doc,
#             "filename": meta.get("filename", "Unknown"),
#             "distance": dist
#         })

#     return final



def generate_answer_with_gpt_4o(question, relevant_chunks):
    # Build clean context
    context_text = "\n\n".join(
        chunk.get("content", "") for chunk in relevant_chunks
    )

    prompt = f"""
        Use ONLY the following context to answer the question.
        If the answer is not present, reply: "Information not available in the documents."

        ### Context:
        {context_text}

        ### Question:
        {question}

        ### Answer:
        """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    #  Groq uses .message.content (dot notation)
    return response.choices[0].message.content
