from flask import Flask,request,jsonify,render_template
app=Flask(__name__)
import os
from vector_store import store_in_chunks,retrieve_relevant_chunks,generate_answer_with_gpt_4o

UPLOAD_FOLDER='uploads'
os.makedirs(UPLOAD_FOLDER,exist_ok=True)
from utils.document_loader import load_documents
from utils.text_splitter import splits_into_chunks

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload-documents')
def upload_documents():
    return render_template('upload_documents.html')

# @app.route('/upload',methods=['POST'])
# def upload_file():
#     files=request.files.getlist('files')
#     file_paths = []
#     for file in files:
#         path=os.path.join(UPLOAD_FOLDER,file.filename)
#         file.save(path)
#         file_paths.append(path)
#     return jsonify({"message": "Files uploaded successfully", "files": file_paths})



@app.route('/upload', methods=['POST'])
def upload_file():
    files = request.files.getlist('files')
    file_paths = []

    for file in files:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        file_paths.append(path)

    print(f"\n📁 [INFO] Uploaded files: {file_paths}")

    folder = UPLOAD_FOLDER
    print("\n⚙️ [INFO] Starting automatic document processing...")

    documents = load_documents(folder)
    print("📄 Documents loaded:", len(documents))

    all_chunks = []
    for doc in documents:
        chunks = splits_into_chunks(doc['content'])
        for chunk in chunks:
            all_chunks.append({"filename": doc['filename'], "chunks": chunk})

    if all_chunks:
        store_in_chunks(all_chunks)
        print(f"[INFO] Stored {len(all_chunks)} chunks in ChromaDB.")
    else:
        print("⚠️ [WARN] No chunks created (maybe files were empty).")

    return jsonify({
        "message": "Files uploaded and processed successfully.",
        "total_files": len(documents),
        "total_chunks": len(all_chunks),
        "sample_chunk": all_chunks[0] if all_chunks else None
    })




# @app.route('/query',methods=['POST'])
# def ask_question():
#     data=request.get_json()
#     question=data.get('question','')

#     if not question:
#         return jsonify({"error":"Question is required"}),400
    
#     relevant_chunks=retrieve_relevant_chunks(question,top_k=3)
    
#     print(f"🔍 [DEBUG] Retrieved {len(relevant_chunks)} relevant chunks.")


#     return jsonify({
#         "question":question,            
#             "relevant_chunks":relevant_chunks,                
#                     })




@app.route('/query',methods=['POST'])
def ask_question():
    data=request.get_json()
    question=data.get('question','')

    if not question:
        return jsonify({"error":"Question is required"}),400
    
    relevant_chunks=retrieve_relevant_chunks(question)
    
    print(f"🔍 [DEBUG] Retrieved {len(relevant_chunks)} relevant chunks.")


    if not relevant_chunks:
        return jsonify({
            "question":question,            
                "relevant_chunks":relevant_chunks,   
                            "source": None
                
                        })
    final_answer=generate_answer_with_gpt_4o(question,relevant_chunks)


    best=relevant_chunks[0]

    return jsonify({
        "question":question,            
        "answer": final_answer,
        "source": best["filename"],
        "distance": best["distance"],
        "chunks_used":len(relevant_chunks)
    })





if __name__ == '__main__':
    app.run(debug=True)