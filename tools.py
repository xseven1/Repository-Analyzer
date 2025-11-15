from github import Github
import chromadb
from typing import List
import os
import re
from sentence_transformers import SentenceTransformer

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
    
    def _analyze_commit_patterns(self, commits_data: list) -> str:
        """Analyze patterns in commits for contextual insights"""
        if not commits_data:
            return ""
        
        # Extract patterns
        authors = {}
        file_patterns = {}
        total_additions = 0
        total_deletions = 0
        
        for metadata in commits_data:
            author = metadata.get('author', 'Unknown')
            authors[author] = authors.get(author, 0) + 1
            total_additions += metadata.get('additions', 0)
            total_deletions += metadata.get('deletions', 0)
        
        analysis = "\n=== COMMIT ANALYSIS ===\n"
        analysis += f"Total changes: +{total_additions:,} additions, -{total_deletions:,} deletions\n"
        analysis += f"Active contributors: {', '.join(authors.keys())}\n"
        
        if total_additions > total_deletions * 2:
            analysis += "📈 Trend: Primarily adding new features/code\n"
        elif total_deletions > total_additions * 2:
            analysis += "🧹 Trend: Significant code cleanup or refactoring\n"
        else:
            analysis += "⚖️ Trend: Balanced mix of additions and modifications\n"
        
        return analysis
    
    def search_commits(self, query: str) -> str:
        """Search through commit history with semantic understanding"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=15,
                where={"type": "commit"}
            )
            
            if not results['documents'][0]:
                return "No commits found matching the query."
            
            output = f"=== COMMIT SEARCH RESULTS ===\n"
            output += f"Query: '{query}'\n"
            output += f"Found {len(results['documents'][0])} relevant commits\n\n"
            
            # Add pattern analysis
            output += self._analyze_commit_patterns(results['metadatas'][0])
            output += "\n"
            
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                output += f"{'='*60}\n"
                output += f"COMMIT #{i}\n"
                output += f"{'='*60}\n"
                output += f"SHA: {metadata.get('sha', 'N/A')[:7]}\n"
                output += f"Author: {metadata.get('author', 'Unknown')}\n"
                output += f"Date: {metadata.get('date', 'Unknown')[:10]}\n"
                
                if 'additions' in metadata and 'deletions' in metadata:
                    changes = metadata['additions'] + metadata['deletions']
                    output += f"Changes: +{metadata['additions']} -{metadata['deletions']} (total: {changes} lines)\n"
                    
                    # Add size context
                    if changes < 10:
                        output += "📝 Size: Small change (minor fix or tweak)\n"
                    elif changes < 100:
                        output += "📄 Size: Medium change (feature addition or refactor)\n"
                    else:
                        output += "📚 Size: Large change (major feature or significant refactor)\n"
                
                output += f"\n{doc}\n\n"
            
            return output
        except Exception as e:
            return f"Error searching commits: {str(e)}"
    
    def _analyze_pr_impact(self, pr, files) -> str:
        """Analyze PR impact and provide context"""
        analysis = "\n=== IMPACT ANALYSIS ===\n"
        
        # Size analysis
        total_changes = pr.additions + pr.deletions
        if total_changes < 50:
            analysis += "📦 Scope: Small PR - Quick review recommended\n"
        elif total_changes < 300:
            analysis += "📦 Scope: Medium PR - Thorough review needed\n"
        else:
            analysis += "📦 Scope: Large PR - Consider breaking into smaller PRs\n"
        
        # File analysis
        if files:
            file_types = {}
            for f in files:
                ext = f.filename.split('.')[-1] if '.' in f.filename else 'no-ext'
                file_types[ext] = file_types.get(ext, 0) + 1
            
            analysis += f"📁 Files affected: {len(files)} files across {len(file_types)} file types\n"
            
            # Identify change patterns
            if any('test' in f.filename.lower() for f in files):
                analysis += "✅ Testing: Includes test file changes\n"
            else:
                analysis += "⚠️  Testing: No test files modified - consider adding tests\n"
            
            if any(f.filename.endswith('.md') or f.filename.endswith('.txt') for f in files):
                analysis += "📖 Documentation: Includes documentation updates\n"
        
        # Review status
        if pr.comments > 0 or pr.review_comments > 0:
            analysis += f"💬 Discussion: {pr.comments + pr.review_comments} comments - Active review process\n"
        else:
            analysis += "💬 Discussion: No comments yet\n"
        
        # State analysis
        if pr.merged:
            analysis += f"✅ Status: Merged successfully"
            if pr.merged_by:
                analysis += f" by {pr.merged_by.login}"
            analysis += "\n"
        elif pr.state == "open":
            analysis += "🔄 Status: Open and awaiting review\n"
        else:
            analysis += "❌ Status: Closed without merging\n"
        
        return analysis
    
    def get_pr_details(self, pr_number: str) -> str:
        """Get details about a specific PR with enhanced context"""
        try:
            # Clean input
            pr_num_str = str(pr_number).strip().strip("'\"#")
            match = re.search(r'\d+', pr_num_str)
            if match:
                pr_num = int(match.group())
            else:
                return f"Error: Could not extract PR number from '{pr_number}'. Please provide just the number."
            
            # Search vector store for related context
            try:
                vector_results = self.collection.query(
                    query_texts=[f"pull request {pr_num}"],
                    n_results=3,
                    where={"type": "pr", "number": pr_num}
                )
            except:
                vector_results = None
            
            # Get fresh GitHub data
            pr = self.repo.get_pull(pr_num)
            
            output = f"{'='*70}\n"
            output += f"PULL REQUEST #{pr.number}: {pr.title}\n"
            output += f"{'='*70}\n\n"
            
            # Basic info
            output += f"👤 Author: {pr.user.login if pr.user else 'Unknown'}\n"
            output += f"📅 Created: {pr.created_at.strftime('%Y-%m-%d %H:%M UTC')}\n"
            output += f"🏷️  State: {pr.state.upper()}\n"
            
            if pr.merged:
                output += f"✅ Merged: {pr.merged_at.strftime('%Y-%m-%d %H:%M UTC')}\n"
                if pr.merged_by:
                    output += f"   Merged by: {pr.merged_by.login}\n"
            
            # Get files for analysis
            files = []
            try:
                files = list(pr.get_files())
            except:
                pass
            
            # Add impact analysis
            output += self._analyze_pr_impact(pr, files)
            
            # Description
            output += f"\n{'='*70}\n"
            output += f"DESCRIPTION\n"
            output += f"{'='*70}\n"
            if pr.body and pr.body.strip():
                output += f"{pr.body}\n"
            else:
                output += "No description provided.\n"
            
            # Files changed with details
            if files:
                output += f"\n{'='*70}\n"
                output += f"FILES CHANGED ({len(files)})\n"
                output += f"{'='*70}\n"
                
                # Group by directory
                dirs = {}
                for f in files:
                    dir_name = '/'.join(f.filename.split('/')[:-1]) or 'root'
                    if dir_name not in dirs:
                        dirs[dir_name] = []
                    dirs[dir_name].append(f)
                
                for dir_name, dir_files in sorted(dirs.items()):
                    output += f"\n📁 {dir_name}/\n"
                    for f in dir_files[:10]:  # Limit per directory
                        output += f"   • {f.filename.split('/')[-1]}"
                        output += f" (+{f.additions} -{f.deletions})"
                        
                        # Add change type indicator
                        if f.additions > 0 and f.deletions == 0:
                            output += " [NEW]"
                        elif f.deletions > f.additions * 2:
                            output += " [MAJOR REFACTOR]"
                        
                        output += "\n"
                    
                    if len(dir_files) > 10:
                        output += f"   ... and {len(dir_files) - 10} more files\n"
            
            # Statistics
            output += f"\n{'='*70}\n"
            output += f"STATISTICS\n"
            output += f"{'='*70}\n"
            output += f"📊 Lines changed: +{pr.additions:,} additions, -{pr.deletions:,} deletions\n"
            output += f"📝 Commits: {pr.commits}\n"
            output += f"💬 Comments: {pr.comments}\n"
            output += f"🔍 Review comments: {pr.review_comments}\n"
            
            # Comments with context
            try:
                comments = list(pr.get_comments())[:8]
                if comments:
                    output += f"\n{'='*70}\n"
                    output += f"REVIEW DISCUSSION (showing {len(comments)} comments)\n"
                    output += f"{'='*70}\n"
                    
                    for idx, c in enumerate(comments, 1):
                        output += f"\n💬 Comment #{idx} by {c.user.login} on {c.created_at.strftime('%Y-%m-%d %H:%M')}\n"
                        comment_preview = c.body[:500]
                        if len(c.body) > 500:
                            comment_preview += "..."
                        output += f"{comment_preview}\n"
                        output += f"{'-'*70}\n"
            except:
                pass
            
            # Related context from vector store
            if vector_results and vector_results['documents'][0]:
                output += f"\n{'='*70}\n"
                output += f"RELATED CONTEXT FROM REPOSITORY\n"
                output += f"{'='*70}\n"
                context = vector_results['documents'][0][0][:600]
                output += f"{context}...\n"
            
            output += f"\n🔗 GitHub URL: {pr.html_url}\n"
            
            return output
            
        except ValueError:
            return f"Error: Invalid PR number '{pr_number}'. Please provide just the number (e.g., 43)."
        except Exception as e:
            return f"Error fetching PR details: {str(e)}"
    
    def search_code(self, query: str) -> str:
        """Search for code snippets with semantic understanding and context"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=8,
                where={"type": "code"}
            )
            
            if not results['documents'][0]:
                return "No code found matching the query."
            
            output = f"=== CODE SEARCH RESULTS ===\n"
            output += f"Query: '{query}'\n"
            output += f"Found {len(results['documents'][0])} relevant code snippets\n\n"
            
            # Analyze code results
            files_found = set()
            for metadata in results['metadatas'][0]:
                files_found.add(metadata.get('file_path', 'Unknown'))
            
            output += f"📂 Spans {len(files_found)} files across the repository\n"
            output += f"💡 Tip: Use get_timeline to see when these were last modified\n\n"
            
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                output += f"{'='*70}\n"
                output += f"RESULT #{i}\n"
                output += f"{'='*70}\n"
                
                file_path = metadata.get('file_path', 'Unknown')
                output += f"📄 File: {file_path}\n"
                
                # Add file type context
                if file_path.endswith('.py'):
                    output += f"🐍 Type: Python module\n"
                elif file_path.endswith(('.js', '.jsx', '.ts', '.tsx')):
                    output += f"⚛️  Type: JavaScript/TypeScript module\n"
                elif file_path.endswith(('.md', '.txt', '.rst')):
                    output += f"📖 Type: Documentation\n"
                
                chunk_idx = metadata.get('chunk_index', 0)
                total_chunks = metadata.get('total_chunks', 1)
                if total_chunks > 1:
                    output += f"📍 Location: Chunk {chunk_idx + 1} of {total_chunks} in this file\n"
                
                file_size = metadata.get('file_size', 0)
                if file_size:
                    output += f"📏 File size: {file_size:,} bytes\n"
                
                output += f"\n💻 Code:\n"
                output += f"```\n{doc[:1000]}"
                if len(doc) > 1000:
                    output += "\n... [truncated, full context available in file]"
                output += f"\n```\n\n"
            
            return output
        except Exception as e:
            return f"Error searching code: {str(e)}"
    
    def get_timeline(self, query: str) -> str:
        """Get timeline of changes with enhanced context and analysis"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=20
            )
            
            if not results['documents'][0]:
                return "No timeline information found."
            
            # Sort by date
            combined = list(zip(results['documents'][0], results['metadatas'][0]))
            sorted_results = sorted(
                combined,
                key=lambda x: x[1].get('date', ''),
                reverse=True
            )
            
            output = f"=== REPOSITORY TIMELINE ===\n"
            output += f"Query: '{query}'\n"
            output += f"Showing {min(15, len(sorted_results))} most recent relevant events\n\n"
            
            # Timeline analysis
            event_types = {}
            for _, metadata in sorted_results[:15]:
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
            
            for idx, (doc, metadata) in enumerate(sorted_results[:15], 1):
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
                
                # Show meaningful excerpt
                excerpt = doc[:250].replace('\n', ' ').strip()
                output += f"\n📝 Summary: {excerpt}...\n\n"
            
            return output
        except Exception as e:
            return f"Error getting timeline: {str(e)}"
    
    def get_repository_stats(self, query: str) -> str:
        """Get comprehensive repository statistics with analysis"""
        try:
            output = f"{'='*70}\n"
            output += f"REPOSITORY OVERVIEW: {self.repo.full_name}\n"
            output += f"{'='*70}\n\n"
            
            # Basic info
            output += f"📄 Description: {self.repo.description or 'No description'}\n"
            output += f"💻 Primary Language: {self.repo.language}\n"
            output += f"📅 Created: {self.repo.created_at.strftime('%Y-%m-%d')}\n"
            output += f"🔄 Last Updated: {self.repo.updated_at.strftime('%Y-%m-%d')}\n"
            
            if self.repo.license:
                output += f"⚖️  License: {self.repo.license.name}\n"
            
            # Popularity metrics
            output += f"\n{'='*70}\n"
            output += f"POPULARITY METRICS\n"
            output += f"{'='*70}\n"
            output += f"⭐ Stars: {self.repo.stargazers_count:,}\n"
            output += f"🍴 Forks: {self.repo.forks_count:,}\n"
            output += f"👁️  Watchers: {self.repo.watchers_count:,}\n"
            output += f"🐛 Open Issues: {self.repo.open_issues_count:,}\n"
            
            # Popularity assessment
            stars = self.repo.stargazers_count
            if stars > 10000:
                output += f"\n🔥 Popularity: Highly popular project (top tier)\n"
            elif stars > 1000:
                output += f"\n✨ Popularity: Well-established project\n"
            elif stars > 100:
                output += f"\n📈 Popularity: Growing project\n"
            else:
                output += f"\n🌱 Popularity: Early stage or niche project\n"
            
            # Contributors
            output += f"\n{'='*70}\n"
            output += f"TOP CONTRIBUTORS\n"
            output += f"{'='*70}\n"
            try:
                contributors = list(self.repo.get_contributors()[:10])
                total_contributions = sum(c.contributions for c in contributors)
                
                for i, c in enumerate(contributors, 1):
                    percentage = (c.contributions / total_contributions * 100) if total_contributions > 0 else 0
                    output += f"{i:2d}. {c.login:20s} - {c.contributions:,} commits ({percentage:.1f}%)\n"
                
                if len(contributors) > 0:
                    top_contributor = contributors[0]
                    output += f"\n💡 Top contributor: {top_contributor.login} with {top_contributor.contributions:,} commits\n"
            except:
                output += "Could not fetch contributors\n"
            
            # Index statistics
            try:
                all_docs = self.collection.get()
                if all_docs and all_docs.get('metadatas'):
                    metadatas = all_docs['metadatas']
                    
                    commits = sum(1 for m in metadatas if m.get('type') == 'commit')
                    prs = sum(1 for m in metadatas if m.get('type') == 'pr')
                    code_chunks = sum(1 for m in metadatas if m.get('type') == 'code')
                    
                    output += f"\n{'='*70}\n"
                    output += f"INDEXED DATA STATISTICS\n"
                    output += f"{'='*70}\n"
                    output += f"💾 Indexed Commits: {commits:,}\n"
                    output += f"📋 Indexed Pull Requests: {prs:,}\n"
                    output += f"📄 Code Chunks: {code_chunks:,}\n"
                    output += f"📊 Total Documents: {len(metadatas):,}\n"
                    
                    output += f"\n💡 This index contains searchable history and code for semantic queries\n"
            except:
                pass
            
            output += f"\n🔗 GitHub URL: {self.repo.html_url}\n"
            
            return output
        except Exception as e:
            return f"Error getting stats: {str(e)}"
    
    def get_openai_tools(self):
        """Return tool definitions in OpenAI format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_commits",
                    "description": "Search commit history using semantic search with detailed analysis. Returns commits with pattern analysis, size context, and trend insights. Use for finding when changes were made, who made them, and understanding commit patterns.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language search query for commits (e.g., 'authentication changes', 'bug fixes by John', 'refactoring work')"
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
                    "description": "Get comprehensive PR analysis including impact assessment, file groupings, testing status, and review discussion. Provides context beyond raw PR data. Input should be ONLY the PR number.",
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
                    "description": "Search for code with semantic understanding and detailed context. Returns code snippets with file type analysis, size info, and location context. Can find implementations even if exact keywords don't match.",
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
                    "name": "get_timeline",
                    "description": "Get chronological timeline with event analysis and summaries. Shows commits, PRs, and code changes ordered by date with overview statistics. Use for understanding repository evolution.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Query to filter timeline (e.g., 'authentication module', 'API changes', 'recent refactoring')"
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
                    "description": "Get comprehensive repository analysis including popularity assessment, contributor breakdown with percentages, and indexed data statistics. Provides context beyond raw numbers.",
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
    
    def get_anthropic_tools(self):
        """Return tool definitions in Anthropic format"""
        return [
            {
                "name": "search_commits",
                "description": "Search commit history using semantic search with detailed analysis. Returns commits with pattern analysis, size context, and trend insights.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query for commits"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_pr_details",
                "description": "Get comprehensive PR analysis including impact assessment, file groupings, testing status, and review discussion.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pr_number": {
                            "type": "string",
                            "description": "The PR number (just the number, no symbols)"
                        }
                    },
                    "required": ["pr_number"]
                }
            },
            {
                "name": "search_code",
                "description": "Search for code with semantic understanding and detailed context.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query for code"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_timeline",
                "description": "Get chronological timeline with event analysis and summaries.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to filter timeline"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_repository_stats",
                "description": "Get comprehensive repository analysis including popularity assessment and contributor breakdown.",
                "input_schema": {
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
        ]