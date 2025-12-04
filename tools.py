from github import Github
import chromadb
from typing import List, Optional
import os
import re
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
import dateparser

class RepoTools:
    def __init__(self, repo_name: str, github_token: str):
        self.repo_name = repo_name
        self.github = Github(github_token)
        self.repo = self.github.get_repo(repo_name)
        
        # Initialize ChromaDB with Sentence Transformers (must match processor)
        persist_dir = f"./chroma_db_{repo_name.replace('/', '_')}"
        
        if not os.path.exists(persist_dir):
            raise ValueError(f"ChromaDB directory not found: {persist_dir}. Please index the repository first.")
        
        # Initialize Sentence Transformer model directly
        print(f"Loading Sentence Transformer model for {repo_name}...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize ChromaDB client (ChromaDB 0.4.x compatible)
        client = chromadb.PersistentClient(path=persist_dir)

        # Create embedding function with proper ChromaDB interface
        from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

        model = self.model

        class SentenceTransformerEF(EmbeddingFunction):
            def __call__(self, input: Documents) -> Embeddings:
                embeddings = model.encode(
                    list(input), 
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                return embeddings.tolist()

        self.collection = client.get_collection(
            name="repo_data",
            embedding_function=SentenceTransformerEF()
        )
        print(f"✅ Collection loaded successfully")
    
    def _parse_date_query(self, query: str) -> Optional[tuple]:
        """
        Extract date range from natural language query
        Returns (start_date, end_date) tuple or None
        """
        query_lower = query.lower()
        now = datetime.now()
        
        # Predefined patterns
        patterns = {
            'last week': timedelta(days=7),
            'past week': timedelta(days=7),
            'last month': timedelta(days=30),
            'past month': timedelta(days=30),
            'last 2 months': timedelta(days=60),
            'last 3 months': timedelta(days=90),
            'last 6 months': timedelta(days=180),
            'last year': timedelta(days=365),
            'past year': timedelta(days=365),
            'this week': timedelta(days=7),
            'this month': timedelta(days=30),
            'this year': timedelta(days=365),
        }
        
        for pattern, delta in patterns.items():
            if pattern in query_lower:
                start_date = now - delta
                return (start_date.isoformat(), now.isoformat())
        
        # Try to parse specific dates using dateparser
        try:
            date_phrases = ['since', 'after', 'from', 'between', 'before', 'until']
            
            for phrase in date_phrases:
                if phrase in query_lower:
                    parts = query_lower.split(phrase)
                    if len(parts) > 1:
                        date_str = parts[1].strip().split()[0:3]
                        parsed = dateparser.parse(' '.join(date_str))
                        if parsed:
                            if phrase in ['since', 'after', 'from']:
                                return (parsed.isoformat(), now.isoformat())
                            elif phrase in ['before', 'until']:
                                return (None, parsed.isoformat())
        except:
            pass
        
        return None
    
    def _filter_by_date(self, metadatas: list, documents: list, date_range: tuple) -> tuple:
        """
        Filter results by date range
        Returns filtered (documents, metadatas)
        """
        if not date_range:
            return documents, metadatas
        
        start_date, end_date = date_range
        filtered_docs = []
        filtered_metas = []
        
        for doc, meta in zip(documents, metadatas):
            doc_date = meta.get('date', '')
            if not doc_date:
                continue
            
            try:
                doc_datetime = datetime.fromisoformat(doc_date.replace('Z', '+00:00'))
                
                in_range = True
                if start_date:
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    if doc_datetime < start_dt:
                        in_range = False
                
                if end_date and in_range:
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    if doc_datetime > end_dt:
                        in_range = False
                
                if in_range:
                    filtered_docs.append(doc)
                    filtered_metas.append(meta)
            except:
                continue
        
        return filtered_docs, filtered_metas
    
    def _analyze_commit_patterns(self, metadatas: list) -> str:
        """Analyze patterns in commit metadata"""
        if not metadatas:
            return ""
        
        output = "📊 PATTERN ANALYSIS:\n"
        
        # Author distribution
        authors = {}
        total_additions = 0
        total_deletions = 0
        
        for meta in metadatas:
            author = meta.get('author', 'Unknown')
            authors[author] = authors.get(author, 0) + 1
            total_additions += meta.get('additions', 0)
            total_deletions += meta.get('deletions', 0)
        
        output += f"   • {len(authors)} unique contributors\n"
        if authors:
            top_author = max(authors.items(), key=lambda x: x[1])
            output += f"   • Most active: {top_author[0]} ({top_author[1]} commits)\n"
        
        output += f"   • Total changes: +{total_additions} -{total_deletions} lines\n"
        
        return output
    
    def search_commits(self, query: str, date_range: Optional[tuple] = None) -> str:
        """Search through commit history with semantic understanding and optional date filtering"""
        try:
            if not date_range:
                date_range = self._parse_date_query(query)
            
            results = self.collection.query(
                query_texts=[query],
                n_results=30,
                where={"type": "commit"}
            )
            
            if not results['documents'][0]:
                return "No commits found matching the query."
            
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            
            if date_range:
                docs, metas = self._filter_by_date(metas, docs, date_range)
                
                if not docs:
                    start_str = date_range[0][:10] if date_range[0] else "beginning"
                    end_str = date_range[1][:10] if date_range[1] else "now"
                    return f"No commits found matching the query in the date range {start_str} to {end_str}."
            
            docs = docs[:15]
            metas = metas[:15]
            
            output = f"=== COMMIT SEARCH RESULTS ===\n"
            output += f"Query: '{query}'\n"
            
            if date_range:
                start_str = date_range[0][:10] if date_range[0] else "beginning"
                end_str = date_range[1][:10] if date_range[1] else "now"
                output += f"📅 Date Range: {start_str} to {end_str}\n"
            
            output += f"Found {len(docs)} relevant commits\n\n"
            
            output += self._analyze_commit_patterns(metas)
            output += "\n"
            
            for i, (doc, metadata) in enumerate(zip(docs, metas), 1):
                output += f"{'='*60}\n"
                output += f"COMMIT #{i}\n"
                output += f"{'='*60}\n"
                output += f"SHA: {metadata.get('sha', 'N/A')[:7]}\n"
                output += f"Author: {metadata.get('author', 'Unknown')}\n"
                output += f"Date: {metadata.get('date', 'Unknown')[:10]}\n"
                
                if 'additions' in metadata and 'deletions' in metadata:
                    changes = metadata['additions'] + metadata['deletions']
                    output += f"Changes: +{metadata['additions']} -{metadata['deletions']} (total: {changes} lines)\n"
                    
                    if changes < 10:
                        output += "🔹 Size: Small change (minor fix or tweak)\n"
                    elif changes < 100:
                        output += "📄 Size: Medium change (feature addition or refactor)\n"
                    else:
                        output += "📚 Size: Large change (major feature or significant refactor)\n"
                
                output += f"\n{doc}\n\n"
            
            return output
        except Exception as e:
            return f"Error searching commits: {str(e)}"
    
    def get_pr_details(self, pr_number: str) -> str:
        """Get detailed PR information with impact analysis"""
        try:
            pr_num = int(pr_number)
            
            results = self.collection.query(
                query_texts=[f"pull request {pr_number}"],
                n_results=5,
                where={"type": "pr", "number": pr_num}
            )
            
            if not results['documents'][0]:
                return f"No PR found with number {pr_number}"
            
            doc = results['documents'][0][0]
            metadata = results['metadatas'][0][0]
            
            output = f"=== PULL REQUEST #{pr_number} DETAILS ===\n\n"
            output += f"Title: {metadata.get('title', 'N/A')}\n"
            output += f"State: {metadata.get('state', 'unknown').upper()}\n"
            output += f"Author: {metadata.get('author', 'Unknown')}\n"
            output += f"Created: {metadata.get('date', 'Unknown')[:10]}\n\n"
            
            output += "📋 FULL CONTENT:\n"
            output += f"{doc}\n\n"
            
            # Try to get live PR data for additional context
            try:
                pr = self.repo.get_pull(pr_num)
                output += "💡 ADDITIONAL CONTEXT:\n"
                output += f"   • Commits: {pr.commits}\n"
                output += f"   • Changed files: {pr.changed_files}\n"
                output += f"   • Additions: +{pr.additions}\n"
                output += f"   • Deletions: -{pr.deletions}\n"
                output += f"   • Comments: {pr.comments}\n"
                
                if pr.merged:
                    output += f"   • ✅ Merged by {pr.merged_by.login if pr.merged_by else 'Unknown'}\n"
                elif pr.state == 'closed':
                    output += "   • ❌ Closed without merging\n"
                else:
                    output += "   • ⏳ Still open\n"
            except:
                pass
            
            return output
        except ValueError:
            return f"Invalid PR number: {pr_number}. Please provide a numeric PR number."
        except Exception as e:
            return f"Error getting PR details: {str(e)}"
    
    def search_code(self, query: str) -> str:
        """Search for code implementations with context"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=10,
                where={"type": "code"}
            )
            
            if not results['documents'][0]:
                return "No code found matching the query."
            
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            
            output = f"=== CODE SEARCH RESULTS ===\n"
            output += f"Query: '{query}'\n"
            output += f"Found {len(docs)} relevant code sections\n\n"
            
            # File type analysis
            file_types = {}
            for meta in metas:
                path = meta.get('file_path', '')
                ext = os.path.splitext(path)[1]
                file_types[ext] = file_types.get(ext, 0) + 1
            
            output += "📁 FILE TYPE DISTRIBUTION:\n"
            for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
                output += f"   • {ext or 'no extension'}: {count} files\n"
            output += "\n"
            
            for i, (doc, metadata) in enumerate(zip(docs, metas), 1):
                output += f"{'='*70}\n"
                output += f"RESULT #{i}\n"
                output += f"{'='*70}\n"
                output += f"📄 File: {metadata.get('file_path', 'Unknown')}\n"
                
                if 'chunk_index' in metadata:
                    output += f"📍 Section: {metadata['chunk_index'] + 1}/{metadata.get('total_chunks', '?')}\n"
                
                output += f"💾 Size: {metadata.get('file_size', 0)} bytes\n\n"
                output += f"{doc[:800]}\n"
                
                if len(doc) > 800:
                    output += "\n[... content truncated ...]\n"
                output += "\n"
            
            return output
        except Exception as e:
            return f"Error searching code: {str(e)}"
    
    def get_timeline(self, query: str, date_range: Optional[tuple] = None) -> str:
        """Get timeline of changes with enhanced context and analysis"""
        try:
            if not date_range:
                date_range = self._parse_date_query(query)
            
            results = self.collection.query(
                query_texts=[query],
                n_results=40
            )
            
            if not results['documents'][0]:
                return "No timeline information found."
            
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            
            if date_range:
                docs, metas = self._filter_by_date(metas, docs, date_range)
                
                if not docs:
                    start_str = date_range[0][:10] if date_range[0] else "beginning"
                    end_str = date_range[1][:10] if date_range[1] else "now"
                    return f"No timeline events found in the date range {start_str} to {end_str}."
            
            combined = list(zip(docs, metas))
            sorted_results = sorted(
                combined,
                key=lambda x: x[1].get('date', ''),
                reverse=True
            )
            
            sorted_results = sorted_results[:15]
            
            output = f"=== REPOSITORY TIMELINE ===\n"
            output += f"Query: '{query}'\n"
            
            if date_range:
                start_str = date_range[0][:10] if date_range[0] else "beginning"
                end_str = date_range[1][:10] if date_range[1] else "now"
                output += f"📅 Date Range: {start_str} to {end_str}\n"
            
            output += f"Showing {len(sorted_results)} most recent relevant events\n\n"
            
            event_types = {}
            for _, metadata in sorted_results:
                event_type = metadata.get('type', 'unknown')
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            output += "📊 Timeline Overview:\n"
            if 'commit' in event_types:
                output += f"   • {event_types['commit']} commits\n"
            if 'pr' in event_types:
                output += f"   • {event_types['pr']} pull requests\n"
            if 'code' in event_types:
                output += f"   • {event_types['code']} code snapshots\n"
            output += "\n"
            
            for idx, (doc, metadata) in enumerate(sorted_results, 1):
                date = metadata.get('date', 'Unknown date')
                doc_type = metadata.get('type', 'unknown')
                
                output += f"{'='*70}\n"
                output += f"[{date[:10]}] EVENT #{idx}\n"
                
                if doc_type == 'pr':
                    output += f"📋 Pull Request #{metadata.get('number')} - {metadata.get('state', 'unknown').upper()}\n"
                    output += f"   Title: {metadata.get('title', 'N/A')}\n"
                    output += f"   Author: {metadata.get('author', 'Unknown')}\n"
                elif doc_type == 'commit':
                    output += f"💾 Commit by {metadata.get('author', 'Unknown')}\n"
                    if 'additions' in metadata and 'deletions' in metadata:
                        output += f"   Changes: +{metadata['additions']} -{metadata['deletions']} lines\n"
                elif doc_type == 'code':
                    output += f"📄 Code: {metadata.get('file_path', 'Unknown')}\n"
                
                excerpt = doc[:250].replace('\n', ' ').strip()
                output += f"\n🔍 Summary: {excerpt}...\n\n"
            
            return output
        except Exception as e:
            return f"Error getting timeline: {str(e)}"
    
    def get_repository_stats(self, query: str) -> str:
        """Get comprehensive repository statistics and analysis"""
        try:
            output = f"=== REPOSITORY STATISTICS ===\n"
            output += f"Repository: {self.repo_name}\n\n"
            
            # Basic repo info
            output += "📊 OVERVIEW:\n"
            output += f"   • Full Name: {self.repo.full_name}\n"
            output += f"   • Description: {self.repo.description or 'No description'}\n"
            output += f"   • Language: {self.repo.language or 'Not specified'}\n"
            output += f"   • Created: {self.repo.created_at.strftime('%Y-%m-%d')}\n"
            output += f"   • Last Updated: {self.repo.updated_at.strftime('%Y-%m-%d')}\n"
            output += f"   • License: {self.repo.license.name if self.repo.license else 'No license'}\n\n"
            
            # Popularity metrics
            output += "⭐ POPULARITY:\n"
            output += f"   • Stars: {self.repo.stargazers_count:,}\n"
            output += f"   • Watchers: {self.repo.watchers_count:,}\n"
            output += f"   • Forks: {self.repo.forks_count:,}\n"
            output += f"   • Open Issues: {self.repo.open_issues_count:,}\n\n"
            
            # Size and activity
            output += "💾 SIZE & ACTIVITY:\n"
            output += f"   • Size: {self.repo.size:,} KB\n"
            output += f"   • Default Branch: {self.repo.default_branch}\n"
            
            try:
                branches = self.repo.get_branches()
                output += f"   • Total Branches: {branches.totalCount}\n"
            except:
                pass
            
            try:
                contributors = list(self.repo.get_contributors())[:10]
                output += f"\n👥 TOP CONTRIBUTORS:\n"
                for i, contrib in enumerate(contributors[:5], 1):
                    output += f"   {i}. {contrib.login} - {contrib.contributions} contributions\n"
            except:
                pass
            
            # Vector store stats
            try:
                count = self.collection.count()
                output += f"\n📚 INDEXED DATA:\n"
                output += f"   • Total Documents: {count:,}\n"
                
                # Sample to get type distribution
                sample = self.collection.get(limit=1000)
                if sample and 'metadatas' in sample:
                    type_counts = {}
                    for meta in sample['metadatas']:
                        doc_type = meta.get('type', 'unknown')
                        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                    
                    output += "   • Document Types:\n"
                    for doc_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                        output += f"     - {doc_type}: {count}\n"
            except:
                pass
            
            return output
        except Exception as e:
            return f"Error getting repository stats: {str(e)}"
    
    def get_openai_tools(self):
        """Return tool definitions in OpenAI format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_commits",
                    "description": """Search commit history with analytical insights and optional temporal filtering.
                    
Supports natural language date queries like:
- "last week", "past month", "last 6 months"
- "since January", "after 2024-01-01"
- "this year", "recent changes"

Beyond listing commits, provides:
- Development patterns and coding practices
- Team structure and collaboration analysis
- Architectural shifts and refactoring trends
- Change type analysis (features, bugs, cleanup)

Use for understanding repository evolution over time.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language search query, may include temporal phrases like 'last month' or 'since January'"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_timeline",
                    "description": """Get chronological timeline with analytical insights and optional temporal filtering.

Supports natural language date queries like:
- "last week", "past month", "last 6 months"  
- "changes this year", "recent activity"
- "since January", "after 2024-01-01"

Shows:
- Evolution of features/modules over time
- Development phases and velocity patterns
- Correlation between commits, PRs, and code changes
- Activity trends and patterns

Use for understanding how the repository evolved during specific time periods.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Query to filter timeline, may include temporal phrases"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_pr_details",
                    "description": """Get comprehensive PR analysis with deep insights including:
- Impact assessment (scope, risk, reviewability)
- File change patterns and architectural implications
- Testing and documentation status
- Review discussion quality and depth
- Merge decision rationale
- Recommendations for similar future PRs

Provides context that helps understand not just the PR content, but its quality, risk level, and impact on the codebase. Input should be ONLY the PR number.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pr_number": {
                                "type": "string",
                                "description": "The PR number (just the number, no symbols)"
                            }
                        },
                        "required": ["pr_number"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_code",
                    "description": """Search for code with semantic understanding and rich context:
- Finds implementations even without exact keyword matches
- Provides file type analysis and architectural context
- Shows code location and structure within files
- Identifies related code patterns across the codebase
- Suggests connections to related functionality

Use for understanding WHERE code lives, HOW it's organized, and WHY certain patterns exist. Can find implementations based on purpose, not just names.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language search query for code (e.g., 'authentication middleware', 'database connection logic', 'API endpoints')"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_repository_stats",
                    "description": """Get comprehensive repository analysis with insights:
- Popularity assessment and community health indicators
- Contributor distribution analysis with sustainability implications
- Development activity patterns and velocity
- License and maintenance status
- Project maturity assessment

Provides context beyond raw numbers - helps understand project health, sustainability, community engagement, and long-term viability.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Query parameter (can be empty string)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]