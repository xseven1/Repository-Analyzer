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
        
        # Initialize tools and agent - CRITICAL: This must complete
        try:
            tools = RepoTools(repo_name, github_token)
            agent = RepoAgent(tools)
            
            # Store in global dict
            indexed_repos[repo_name] = agent
            print(f"✅ Agent initialized and stored for {repo_name}")
            print(f"📊 Currently indexed repos: {list(indexed_repos.keys())}")
            
        except Exception as e:
            error_msg = f"Failed to initialize agent: {str(e)}"
            print(f"❌ {error_msg}")
            yield f"data: {json.dumps({'status': 'error', 'message': error_msg})}\n\n"
            return
        
        yield f"data: {json.dumps({'status': 'complete', 'message': f'Successfully indexed {repo_name}', 'percent': 100, 'repo_name': repo_name})}\n\n"
    
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Indexing error: {error_msg}")
        yield f"data: {json.dumps({'status': 'error', 'message': error_msg})}\n\n"

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
    
    print(f"🔍 Query request for: {repo_name}")
    print(f"📊 Currently indexed repos: {list(indexed_repos.keys())}")
    
    if repo_name not in indexed_repos:
        raise HTTPException(
            status_code=404,
            detail=f"Repository {repo_name} not indexed. Please index it first. Currently indexed: {list(indexed_repos.keys())}"
        )
    
    agent = indexed_repos[repo_name]
    print(f"✅ Found agent for {repo_name}, executing query...")
    answer = agent.query(request.question)
    
    return {"answer": answer, "repo_name": repo_name}

@app.get("/indexed")
async def get_indexed():
    """Get list of indexed repositories"""
    repos = list(indexed_repos.keys())
    print(f"📋 Indexed repos requested: {repos}")
    return {
        "indexed_repos": repos,
        "count": len(repos)
    }

@app.get("/")
async def root():
    return {"message": "Repository Analyzer API", "indexed_repos": list(indexed_repos.keys())}