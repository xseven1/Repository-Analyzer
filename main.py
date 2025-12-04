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
import glob
from typing import List, Dict, Optional
import uuid

load_dotenv()

app = FastAPI()
github_token = os.getenv("GITHUB_TOKEN")
if not github_token:
    print("ERROR: GITHUB_TOKEN not found in environment!")
else:
    print(f"Token found: {github_token[:10]}...")

# Store indexed repos and conversations
indexed_repos = {}
conversations = {}  # conversation_id -> list of messages

class IndexRequest(BaseModel):
    repo_url: str

class QueryRequest(BaseModel):
    repo_url: str
    question: str
    conversation_id: Optional[str] = None

class ConversationResponse(BaseModel):
    answer: str
    repo_name: str
    conversation_id: str

def extract_repo_name(repo_url: str) -> str:
    """Extract owner/repo from GitHub URL or return as-is if already in that format"""
    match = re.search(r'github\.com/([^/]+/[^/]+)', repo_url)
    if match:
        return match.group(1).rstrip('/')
    return repo_url.strip('/')

def load_existing_repos():
    """Load all existing ChromaDB collections on startup"""
    print("\n🔍 Scanning for existing indexed repositories...")
    
    chroma_dirs = glob.glob("./chroma_db_*")
    
    for chroma_dir in chroma_dirs:
        try:
            dir_name = os.path.basename(chroma_dir)
            repo_name = dir_name.replace('chroma_db_', '').replace('_', '/', 1)
            
            print(f"  📂 Found: {repo_name}")
            
            tools = RepoTools(repo_name, github_token)
            agent = RepoAgent(tools)
            
            indexed_repos[repo_name] = agent
            print(f"  ✅ Loaded: {repo_name}")
            
        except Exception as e:
            print(f"  ⚠️  Failed to load {chroma_dir}: {str(e)}")
            continue
    
    if indexed_repos:
        print(f"\n✅ Successfully loaded {len(indexed_repos)} indexed repositories")
        print(f"📋 Available repos: {list(indexed_repos.keys())}")
    else:
        print("\n📭 No previously indexed repositories found")

@app.on_event("startup")
async def startup_event():
    load_existing_repos()

async def index_with_progress(repo_url: str):
    """Generator that yields progress updates during indexing"""
    try:
        github_token = os.getenv("GITHUB_TOKEN")
        repo_name = extract_repo_name(repo_url)
        
        if repo_name in indexed_repos:
            yield f"data: {json.dumps({'status': 'info', 'message': f'{repo_name} is already indexed. Re-indexing...', 'percent': 0})}\n\n"
        
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
        
        try:
            tools = RepoTools(repo_name, github_token)
            agent = RepoAgent(tools)
            
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
    """Query an indexed repository with conversation history"""
    repo_name = extract_repo_name(request.repo_url)
    
    print(f"🔍 Query request for: {repo_name}")
    print(f"📊 Currently indexed repos: {list(indexed_repos.keys())}")
    
    if repo_name not in indexed_repos:
        raise HTTPException(
            status_code=404,
            detail=f"Repository {repo_name} not indexed. Please index it first. Currently indexed: {list(indexed_repos.keys())}"
        )
    
    # Get or create conversation ID
    conversation_id = request.conversation_id
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        conversations[conversation_id] = []
        print(f"📝 Created new conversation: {conversation_id}")
    else:
        print(f"📖 Continuing conversation: {conversation_id}")
    
    # Get conversation history
    conversation_history = conversations.get(conversation_id, [])
    
    agent = indexed_repos[repo_name]
    print(f"✅ Found agent for {repo_name}, executing query with {len(conversation_history)} previous messages...")
    
    # Query with history
    answer = agent.query_with_history(request.question, conversation_history)
    
    # Store the exchange in conversation history
    conversations[conversation_id].append({
        "role": "user",
        "content": request.question
    })
    conversations[conversation_id].append({
        "role": "assistant", 
        "content": answer
    })
    
    # Limit conversation history to last 10 exchanges (20 messages)
    if len(conversations[conversation_id]) > 20:
        conversations[conversation_id] = conversations[conversation_id][-20:]
    
    return {
        "answer": answer,
        "repo_name": repo_name,
        "conversation_id": conversation_id
    }

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    if conversation_id not in conversations:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id} not found"
        )
    
    return {
        "conversation_id": conversation_id,
        "messages": conversations[conversation_id],
        "message_count": len(conversations[conversation_id])
    }

@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    if conversation_id not in conversations:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation {conversation_id} not found"
        )
    
    del conversations[conversation_id]
    return {"message": f"Deleted conversation {conversation_id}"}

@app.get("/conversations")
async def list_conversations():
    """List all active conversations"""
    return {
        "conversations": [
            {
                "id": conv_id,
                "message_count": len(messages),
                "last_message": messages[-1]["content"][:100] if messages else ""
            }
            for conv_id, messages in conversations.items()
        ],
        "total": len(conversations)
    }

@app.get("/indexed")
async def get_indexed():
    """Get list of indexed repositories"""
    repos = list(indexed_repos.keys())
    print(f"📋 Indexed repos requested: {repos}")
    return {
        "indexed_repos": repos,
        "count": len(repos)
    }

@app.delete("/indexed/{owner}/{repo}")
async def delete_indexed(owner: str, repo: str):
    """Delete an indexed repository"""
    repo_name = f"{owner}/{repo}"
    
    if repo_name not in indexed_repos:
        raise HTTPException(
            status_code=404,
            detail=f"Repository {repo_name} not found in indexed repos"
        )
    
    try:
        del indexed_repos[repo_name]
        
        import shutil
        chroma_dir = f"./chroma_db_{repo_name.replace('/', '_')}"
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)
            print(f"🗑️  Deleted ChromaDB directory: {chroma_dir}")
        
        print(f"✅ Deleted indexed repo: {repo_name}")
        return {"message": f"Successfully deleted {repo_name}"}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting repository: {str(e)}"
        )

@app.get("/")
async def root():
    return {
        "message": "Repository Analyzer API",
        "indexed_repos": list(indexed_repos.keys()),
        "active_conversations": len(conversations)
    }