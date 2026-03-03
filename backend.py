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
    allow_origins=["*"], 
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, "index.html")
    
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": f"index.html not found at {index_path}. Please make sure index.html is committed."}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Saves uploaded files to the local 'data' directory. Files now accumulate!"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    saved_files = []
    for file in files:
        file_path = os.path.join(data_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file.filename)
        
    return {"message": "Files uploaded successfully.", "files": saved_files}

@app.post("/clear")
async def clear_documents():
    """Wipes the current document context so the user can start a fresh comparison."""
    global global_gemini_files
    global_gemini_files.clear()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
                
    return {"message": "All documents cleared."}

@app.post("/ingest")
async def ingest_documents(api_key: str = Form(...)):
    """Uploads accumulated files from 'data' directory directly to Gemini's File API."""
    global global_gemini_files
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        raise HTTPException(status_code=400, detail="No files found to ingest. Please upload files first.")
    
    if not api_key:
        raise HTTPException(status_code=400, detail="Gemini API Key is required.")

    try:
        client = genai.Client(api_key=api_key)
        
        # Clear out previously uploaded files from memory to refresh context
        global_gemini_files.clear()
        
        # Upload each document directly to Gemini
        count = 0
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path):
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

        # Updated prompt to explicitly encourage cross-document comparisons and structured outputs
        system_instruction = (
            "You are an expert technical assistant for a Managed SASE product. "
            "You have been provided with one or multiple documents (e.g., SLAs, service descriptions). "
            "Answer the user's question using ONLY the provided context from the uploaded documents. "
            "If the user asks for a comparison across multiple products or documents, meticulously extract the relevant metrics "
            "(such as uptime, changes per month, SLAs) from EACH document and present a clear, structured comparison (e.g., using bullet points or a markdown table). "
            "If the answer is not contained in the provided documents, explicitly state: "
            "'I cannot find the answer to this in the uploaded documents.'"
        )
        
        contents = [req.question] + global_gemini_files
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1
            )
        )
        
        unique_sources = [f.display_name for f in global_gemini_files if f.display_name]
        
        return {
            "answer": response.text,
            "sources": unique_sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query Error: {str(e)}")
