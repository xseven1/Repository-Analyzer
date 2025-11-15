# Repository Analyzer

A semantic search-powered GitHub repository analysis tool that uses GPT-4 Mini and vector embeddings to understand and query codebases.

## Overview

Repository Analyzer indexes GitHub repositories and allows you to ask natural language questions about commits, pull requests, code implementations, and repository history. It uses Sentence Transformers for semantic search and GPT-4 Mini as an intelligent agent to interpret queries and synthesize information.

## Features

- **Semantic Commit Search**: Find commits using natural language queries about features, bug fixes, or authors
- **Detailed PR Analysis**: Get comprehensive pull request information with impact assessment and file analysis
- **Code Search**: Search for implementations, functions, and patterns even without exact keyword matches
- **Timeline Views**: Understand repository evolution through chronological event timelines
- **Repository Statistics**: Access detailed stats with popularity metrics and contributor breakdowns
- **Smart Context Management**: Automatic conversation history trimming to handle long interactions

## Architecture

### Components

- **FastAPI Backend** (`main.py`): REST API for indexing and querying repositories
- **Streamlit Frontend** (`app.py`): Web interface for user interactions
- **GitHub Fetcher** (`github_fetcher.py`): Retrieves repository data via GitHub API
- **Processor** (`processor.py`): Chunks and embeds code/commits/PRs using Sentence Transformers
- **Vector Store** (`tools.py`): ChromaDB-based semantic search with custom tooling
- **Agent** (`agent.py`): GPT-4 Mini powered conversational agent with tool calling

### Technology Stack

- **LLM**: OpenAI GPT-4 Mini (gpt-4o-mini)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database**: ChromaDB 0.4.24
- **API Framework**: FastAPI 0.104.1
- **UI**: Streamlit 1.28.2
- **GitHub Integration**: PyGithub 2.1.1

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- GitHub Personal Access Token

### Setup

1. Clone the repository:
```bash
git clone 
cd repository-analyzer
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file in project root:
```bash
OPENAI_API_KEY=sk-your-openai-key-here
GITHUB_TOKEN=ghp_your-github-token-here
```

## Usage

### Starting the Application

1. Start the FastAPI backend:
```bash
uvicorn main:app --reload
```

2. In a separate terminal, start the Streamlit frontend:
```bash
streamlit run app.py
```

3. Open your browser to `http://localhost:8501`

### Indexing a Repository

1. Enter a GitHub repository URL (e.g., `https://github.com/owner/repo`)
2. Click "Index Repository"
3. Wait for the indexing process to complete (progress shown in real-time)

### Querying a Repository

1. Select or enter a repository URL
2. Type your question in natural language
3. Click "Ask Question"
4. View the detailed analysis

### Example Questions

**Detailed Analysis:**
- "Tell me the complete history of this repository"
- "What is in PR #43 with all details?"
- "Give me a comprehensive overview of recent changes"

**Code Investigation:**
- "Where is the authentication middleware implemented?"
- "Find all database connection logic"
- "Show me the API endpoint implementations"

**Timeline & History:**
- "Tell me the journey of the user authentication module"
- "What changed in the last 6 months?"
- "Who are the main contributors and what did they work on?"

**Statistics:**
- "Give me detailed repository statistics"
- "What are the most active areas of the codebase?"



## Configuration

### Indexing Limits

Configure in `github_fetcher.py`:
- **Commits**: 50 (default)
- **Pull Requests**: 100 (default)
- **Files**: 500 (default)
- **Max File Size**: 100KB per file

### Context Window Management

Configure in `agent.py`:
- **Max Tokens**: 120,000 (GPT-4 Mini has 128K context window)
- **Conversation Trimming**: Keeps last 8 messages when context exceeds limit
- **Tool Result Truncation**: 15,000 tokens per tool result

### Chunking Parameters

Configure in `processor.py`:
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Max Chunk Size**: 1500 characters

## GitHub API Rate Limits

With an authenticated GitHub token:
- **5,000 requests per hour**
- Indexing a typical repository uses approximately 150-200 requests

Recommended repository sizes:
- **Small** (20-50 commits, 10-20 PRs): ~50 requests
- **Medium** (100-200 commits, 50-100 PRs): 200-500 requests
- **Large** (500+ commits, 200+ PRs): 1000+ requests

## Smart Chunking

The processor uses intelligent chunking strategies:

- **Python Files**: Respects class and function boundaries
- **JavaScript/TypeScript**: Tracks brace scope and function declarations
- **Generic Text**: Splits by paragraphs with overlap
- **Context Preservation**: Each chunk includes file path context

## Tool Capabilities

The agent has access to five specialized tools:

1. **search_commits**: Semantic search through commit history with pattern analysis
2. **get_pr_details**: Comprehensive PR analysis with impact assessment
3. **search_code**: Code search with file type detection and location context
4. **get_timeline**: Chronological event timeline with statistics
5. **get_repository_stats**: Repository overview with popularity metrics

## Project Structure
```
repository-analyzer/
├── agent.py              # GPT-4 Mini agent with tool calling
├── app.py                # Streamlit frontend
├── github_fetcher.py     # GitHub API data retrieval
├── main.py               # FastAPI backend server
├── processor.py          # Document chunking and embedding
├── tools.py              # Repository analysis tools
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (not in repo)
└── chroma_db_*/          # Vector database storage (generated)
```
