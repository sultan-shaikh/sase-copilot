import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from google import genai
from google.genai import types

# Initialize FastAPI App
app = FastAPI(title="SASE Copilot API")

# Enable CORS so the HTML frontend can communicate with this backend securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, restrict this to your actual frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state to keep track of files uploaded to Gemini's File API
global_gemini_files = []

class QueryRequest(BaseModel):
    question: str
    api_key: str

@app.get("/")
async def serve_frontend():
    """Serves the frontend HTML interface."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"error": "index.html not found. Please place the HTML file in the same directory as backend.py."}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Saves uploaded SLA and Service Description files to the local 'data' directory."""
    os.makedirs("data", exist_ok=True)
    
    # Clear out old files for this demo to keep the context fresh
    for filename in os.listdir("data"):
        file_path = os.path.join("data", filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)

    saved_files = []
    for file in files:
        file_path = f"data/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file.filename)
        
    return {"message": "Files uploaded successfully.", "files": saved_files}

@app.post("/ingest")
async def ingest_documents(api_key: str = Form(...)):
    """Uploads files from 'data' directory directly to Gemini's File API."""
    global global_gemini_files
    
    if not os.path.exists("data") or not os.listdir("data"):
        raise HTTPException(status_code=400, detail="No files found to ingest. Please upload files first.")
    
    if not api_key:
        raise HTTPException(status_code=400, detail="Gemini API Key is required.")

    try:
        client = genai.Client(api_key=api_key)
        
        # Clear out previously uploaded files from memory
        global_gemini_files.clear()
        
        # Upload each document directly to Gemini (handles up to 50MB PDFs natively)
        count = 0
        for filename in os.listdir("data"):
            file_path = os.path.join("data", filename)
            if os.path.isfile(file_path):
                # Uploading gives us a File object we can pass directly into the prompt
                g_file = client.files.upload(
                    file=file_path, 
                    config=types.UploadFileConfig(display_name=filename)
                )
                global_gemini_files.append(g_file)
                count += 1
                
        return {"message": f"Successfully uploaded {count} documents to Gemini."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion Error: {str(e)}")

@app.post("/query")
async def query_documents(req: QueryRequest):
    """Answers a question based ONLY on the Gemini uploaded files."""
    global global_gemini_files
    
    if not global_gemini_files:
        raise HTTPException(status_code=400, detail="Documents have not been ingested yet. Please upload and ingest first.")
    
    try:
        client = genai.Client(api_key=req.api_key)

        system_instruction = (
            "You are an expert technical assistant for a Managed SASE product. "
            "Answer the user's question using ONLY the provided context from the uploaded documents. "
            "If the answer is not contained in the provided documents, explicitly state: "
            "'I cannot find the answer to this in the uploaded documents.'"
        )
        
        # Combine the user's question and the raw Gemini file objects
        contents = [req.question] + global_gemini_files
        
        # Generate the response using Gemini's massive context window
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1
            )
        )
        
        # Return the display names of the files we used as context
        unique_sources = [f.display_name for f in global_gemini_files if f.display_name]
        
        return {
            "answer": response.text,
            "sources": unique_sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query Error: {str(e)}")
