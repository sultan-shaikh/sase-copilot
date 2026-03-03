import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from google import genai
from google.genai import types

# Initialize FastAPI App
app = FastAPI(title="SASE Agent API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for internal context
global_gemini_files = []

class QueryRequest(BaseModel):
    question: str
    api_key: str
    target_product: str
    enable_search: bool

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
    """Saves uploaded internal documents to the local 'data' directory."""
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
    """Wipes the internal document context."""
    global global_gemini_files
    global_gemini_files.clear()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
                
    return {"message": "All internal documents cleared."}

@app.post("/ingest")
async def ingest_documents(api_key: str = Form(...)):
    """Uploads accumulated files to Gemini."""
    global global_gemini_files
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        raise HTTPException(status_code=400, detail="No internal files found to ingest.")
    
    if not api_key:
        raise HTTPException(status_code=400, detail="Gemini API Key is required.")

    try:
        client = genai.Client(api_key=api_key)
        global_gemini_files.clear()
        
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
                
        return {"message": f"Successfully ingested {count} internal documents."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion Error: {str(e)}")

@app.post("/query")
async def query_documents(req: QueryRequest):
    """Answers a question using both internal documents and external live web search."""
    global global_gemini_files
    
    try:
        client = genai.Client(api_key=req.api_key)

        # The new dynamic persona blending internal and external knowledge
        system_instruction = (
            f"You are a master-level Business and Technical Expert specializing in {req.target_product}. "
            "Your role is to act as a consolidated internal and external product expert for a Managed Service Provider (MSP). "
            "You must synthesize information from two primary sources: "
            "1. The uploaded internal documents (which define your company's specific managed wrapper, SLAs, and service descriptions). "
            "2. Your vast external knowledge and real-time internet searches regarding the core vendor product, its competitors, and the broader market. "
            "When answering, clearly distinguish between the base vendor's capabilities and your company's specific managed offering. "
            "Format responses professionally for Sales/Presales, using markdown tables for competitive comparisons and bullet points for key differentiators."
        )
        
        # If the user hasn't uploaded internal docs yet, we still allow external queries
        contents = [req.question]
        if global_gemini_files:
            contents.extend(global_gemini_files)
            
        # Enable Google Search Grounding if requested by the frontend
        tools = [{"google_search": {}}] if req.enable_search else None
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2, # slightly higher to allow creative synthesis
                tools=tools
            )
        )
        
        unique_sources = [f.display_name for f in global_gemini_files if f.display_name]
        if req.enable_search:
            unique_sources.append("Live Internet Search")
        
        return {
            "answer": response.text,
            "sources": unique_sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query Error: {str(e)}")
