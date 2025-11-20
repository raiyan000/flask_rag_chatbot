# 🧠 RAG Chatbot using Flask + ChromaDB + OpenAI

A complete Retrieval-Augmented Generation (RAG) chatbot built with **Flask**, **ChromaDB**, **OpenAI GPT**, and **Sentence Transformers**. The chatbot supports document uploads, persistent vector storage, semantic search, and intelligent question answering.

---

## 🚀 Project Overview

This project allows users to upload documents (TXT/PDF etc.), which are then chunked, embedded, and stored in a **persistent vector database** using ChromaDB. When a question is asked, the system retrieves the most relevant chunks and uses GPT to generate the final answer.

---

## 🏗️ Tech Stack

* **Flask** – Backend API
* **ChromaDB** – Vector database (persistent mode)
* **SentenceTransformers** – Embedding model (`all-MiniLM-L6-v2`)
* **OpenAI GPT (4o/4.1)** – For answer generation
* **HTML + JS + Bootstrap** – UI for file upload & chatbot

---

## 📌 Features

* Upload documents and auto-process them into chunks
* Store embeddings in a persistent vector database
* Prevent duplicate chunks
* Retrieve highly relevant chunks with improved filtering
* GPT-powered answer generation
* Chat UI with clean message bubbles

---

# 🧩 Challenges Faced & Solutions

## **1. ChromaDB Was Empty on Every Server Restart**

### Problem

After restarting Flask, the chatbot kept saying:

> "Information not available in the documents."
> Until a new file was uploaded.

### Cause

Using `chromadb.Client()` → stored vectors **in memory**, erased on restart.

### Solution

Use persistent client:

```python
chroma_client = chromadb.PersistentClient(path="vector_db")
```

Now documents auto-load when the server starts.

---

## **2. Duplicate Chunks Stored in Vector DB**

### Problem

Uploading the same file again produced **duplicate chunks**, affecting retrieval accuracy.

### Solution

Before storing chunks:

```python
existing_texts = set(existing['documents'])
if text in existing_texts:
    continue
```

This prevents duplicates.

---

## **3. Wrong Chunks Returned During Retrieval**

### Problem

Query:

> "17 Nov updates"
> Returned text from **other files** like "30 October.txt".

### Cause

Basic top-k similarity retrieval with no context filtering.

### Solution

Implemented:

* **Similarity threshold**
* **Filename-based filtering**
* **Windowed chunk grouping**
* **Fallback logic**

This significantly improved accuracy.

---

# 📚 Lessons Learned

* Persistent vector stores are essential for production RAG apps.
* Duplicate data causes major problems—always dedupe embeddings.
* Naive KNN retrieval is not enough; custom logic improves reliability.
* Always validate `None`-type results when interacting with GPT responses.
* UI/UX matters: make chatbot error-free and intuitive.

---

# 🛠️ Setup Instructions

### **1. Clone the Repository**

```bash
git clone <your-repo-url>
cd flask_rag_chatbot
```

### **2. Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Run Flask App**

```bash
flask --app app.py run --debug
```

---

# 🧪 API Endpoints

### **POST /upload_documents**

Upload document → automatic chunking & storage.

### **POST /query**

Ask chatbot a question.

---

# 📁 Project Structure

```
flask_rag_chatbot/
│── app.py
│── vector_store.py
│── templates/
│── static/
│── uploads/
│── vector_db/   # Chroma persistent data
└── requirements.txt
```

---

# 🙌 Future Improvements

* Add PDF OCR support
* Add user chat history in UI
* Add authentication & user profiles
* Deploy on Render / Railway / EC2

---

# 📝 Author

Developed by **Raiyan** — aims to build practical AI solutions using RAG and LLMs.

---

If you like this project, ⭐ the repo on GitHub!
