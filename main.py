# main.py
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from components.data_processor import DataProcessor
from components.rag_chain import RAGChain

# Load environment variables
load_dotenv()

# Verify environment variables
# required_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
required_vars = ['OPENAI_API_KEY', 'OPENROUTER_API_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize components
data_processor = DataProcessor()
rag_chain = RAGChain()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting RAG system initialization...")
    if not rag_chain.collection_exists():
        documents = data_processor.process_documents()
        rag_chain.initialize(documents)
    else:
        rag_chain.initialize(None)
    print("RAG system initialized successfully!")
    yield
    # Shutdown
    if rag_chain.db:
        rag_chain.db.persist()
    print("Shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(title="RAG System API", lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Create static directory if it doesn't exist
Path("static").mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
async def ui():
    """Redirect to the UI"""
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

# Define request model
class Question(BaseModel):
    text: str

@app.post("/query")
async def query_rag(question: Question):
    """Query the RAG system"""
    try:
        response = rag_chain.query(question.text)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "RAG System API is running. Visit /docs for documentation."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)