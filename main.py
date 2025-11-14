from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from github_fetcher import RepoFetcher
from processor import RepoProcessor
from tools import RepoTools
from agent import RepoAgent
import os
from dotenv import load_dotenv
import re
import asyncio
import json

load_dotenv()

app = FastAPI()
github_token = os.getenv("GITHUB_TOKEN")
if not github_token:
    print("ERROR: GITHUB_TOKEN not found in environment!")
else:
    print(f"Token found: {github_token[:10]}...")

# Store indexed repos
indexed_repos = {}

class IndexRequest(BaseModel):
    repo_url: str

class QueryRequest(BaseModel):
    repo_url: str
    question: str

def extract_repo_name(repo_url: str) -> str:
    """Extract owner/repo from GitHub URL or return as-is if already in that format"""
    match = re.search(r'github\.com/([^/]+/[^/]+)', repo_url)
    if match:
        return match.group(1).rstrip('/')
    return repo_url.strip('/')

async def index_with_progress(repo_url: str):
    """Generator that yields progress updates during indexing"""
    try:
        github_token = os.getenv("GITHUB_TOKEN")
        repo_name = extract_repo_name(repo_url)
        
        yield f"data: {json.dumps({'status': 'progress', 'message': 'Starting indexing...', 'percent': 0})}\n\n"
        
        loop = asyncio.get_event_loop()
        
        yield f"data: {json.dumps({'status': 'progress', 'message': 'Fetching repository data...', 'percent': 20})}\n\n"
        fetcher = RepoFetcher(github_token)
        repo_data = await loop.run_in_executor(None, fetcher.fetch_repo_data, repo_name)
        
        yield f"data: {json.dumps({'status': 'progress', 'message': 'Processing and embedding documents...', 'percent': 60})}\n\n"
        processor = RepoProcessor()
        vectorstore = await loop.run_in_executor(
            None, processor.process_and_store, repo_data, repo_name
        )
        
        yield f"data: {json.dumps({'status': 'progress', 'message': 'Initializing agent...', 'percent': 90})}\n\n"
        tools = RepoTools(repo_name, github_token)
        agent = RepoAgent(tools)
        
        indexed_repos[repo_name] = agent
        
        yield f"data: {json.dumps({'status': 'complete', 'message': f'Successfully indexed {repo_name}', 'percent': 100, 'repo_name': repo_name})}\n\n"
    
    except Exception as e:
        yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

@app.post("/index")
async def index_repository(request: IndexRequest):
    """Index a GitHub repository with progress updates"""
    return StreamingResponse(
        index_with_progress(request.repo_url),
        media_type="text/event-stream"
    )

@app.post("/query")
async def query_repository(request: QueryRequest):
    """Query an indexed repository"""
    repo_name = extract_repo_name(request.repo_url)
    
    if repo_name not in indexed_repos:
        raise HTTPException(
            status_code=404,
            detail=f"Repository {repo_name} not indexed. Please index it first."
        )
    
    agent = indexed_repos[repo_name]
    answer = agent.query(request.question)
    
    return {"answer": answer, "repo_name": repo_name}

@app.get("/indexed")
async def get_indexed():
    """Get list of indexed repositories"""
    return {
        "indexed_repos": list(indexed_repos.keys()),
        "count": len(indexed_repos)
    }

@app.get("/")
async def root():
    return {"message": "Repository Analyzer API", "indexed_repos": list(indexed_repos.keys())}