import chromadb
from typing import List, Dict, Tuple
import os
import shutil
import re
from sentence_transformers import SentenceTransformer

class RepoProcessor:
    def __init__(self):
        # Initialize Sentence Transformer model directly
        print("Loading Sentence Transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully")
        
        # Chunking parameters
        self.chunk_size = 1000  # characters
        self.chunk_overlap = 200  # characters overlap for context
        self.max_chunk_size = 1500  # hard limit
    
    def _get_embedding_function(self):
        """Return a callable embedding function for ChromaDB"""
        from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
        
        model = self.model
        
        class SentenceTransformerEF(EmbeddingFunction):
            def __call__(self, input: Documents) -> Embeddings:
                # Force convert_to_numpy=True, then convert to list
                embeddings = model.encode(
                    list(input), 
                    convert_to_numpy=True,  # This ensures numpy arrays, not tensors
                    show_progress_bar=False
                )
                # Convert numpy array to list
                return embeddings.tolist()
        
        return SentenceTransformerEF()
        
    
    def _smart_chunk_code(self, content: str, file_path: str) -> List[Tuple[str, Dict]]:
        """
        Smart chunking for code files that respects structure
        Returns list of (chunk_text, metadata) tuples
        """
        chunks = []
        
        # Detect if it's a Python file for smarter chunking
        if file_path.endswith('.py'):
            chunks = self._chunk_python_code(content)
        elif file_path.endswith(('.js', '.ts', '.jsx', '.tsx')):
            chunks = self._chunk_javascript_code(content)
        else:
            # Generic text chunking
            chunks = self._chunk_generic_text(content)
        
        # Add file context to each chunk
        result = []
        for i, chunk in enumerate(chunks):
            # Add file path context at the beginning of each chunk
            contextualized_chunk = f"File: {file_path}\n\n{chunk}"
            result.append((contextualized_chunk, {"chunk_index": i, "total_chunks": len(chunks)}))
        
        return result
    
    def _chunk_python_code(self, content: str) -> List[str]:
        """Chunk Python code by respecting class and function boundaries"""
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_size = 0
        in_function = False
        in_class = False
        indent_level = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            # Detect function/class definitions
            stripped = line.lstrip()
            if stripped.startswith('def ') or stripped.startswith('async def '):
                in_function = True
                indent_level = len(line) - len(stripped)
            elif stripped.startswith('class '):
                in_class = True
                indent_level = len(line) - len(stripped)
            
            # Check if we should start a new chunk
            if current_size + line_size > self.chunk_size:
                # If we're in a function/class, try to finish it first
                if in_function or in_class:
                    current_chunk.append(line)
                    current_size += line_size
                    
                    # Check if function/class ended (return to base indent or less)
                    if stripped and len(line) - len(stripped) <= indent_level and line != lines[0]:
                        chunks.append('\n'.join(current_chunk))
                        # Keep overlap
                        overlap_lines = current_chunk[-10:] if len(current_chunk) > 10 else current_chunk
                        current_chunk = overlap_lines
                        current_size = sum(len(l) + 1 for l in overlap_lines)
                        in_function = False
                        in_class = False
                else:
                    # Split here
                    if current_chunk:
                        chunks.append('\n'.join(current_chunk))
                        # Keep overlap
                        overlap_lines = current_chunk[-10:] if len(current_chunk) > 10 else current_chunk
                        current_chunk = overlap_lines
                        current_size = sum(len(l) + 1 for l in overlap_lines)
                    current_chunk.append(line)
                    current_size += line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _chunk_javascript_code(self, content: str) -> List[str]:
        """Chunk JavaScript/TypeScript code by respecting function boundaries"""
        chunks = []
        lines = content.split('\n')
        
        current_chunk = []
        current_size = 0
        brace_count = 0
        in_function = False
        
        for line in lines:
            line_size = len(line) + 1
            
            # Detect function declarations
            if re.search(r'(function\s+\w+|const\s+\w+\s*=\s*.*=>|\w+\s*\(.*\)\s*{)', line):
                in_function = True
            
            # Count braces to track scope
            brace_count += line.count('{') - line.count('}')
            
            # Check if we should start a new chunk
            if current_size + line_size > self.chunk_size and brace_count == 0 and not in_function:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    # Keep overlap
                    overlap_lines = current_chunk[-10:] if len(current_chunk) > 10 else current_chunk
                    current_chunk = overlap_lines
                    current_size = sum(len(l) + 1 for l in overlap_lines)
            
            current_chunk.append(line)
            current_size += line_size
            
            # Reset function flag when scope closes
            if brace_count == 0:
                in_function = False
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _chunk_generic_text(self, content: str) -> List[str]:
        """Generic text chunking with overlap for non-code files"""
        chunks = []
        
        # Try to split by paragraphs first
        paragraphs = content.split('\n\n')
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para) + 2  # +2 for \n\n
            
            if current_size + para_size > self.chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    # Keep last paragraph for overlap
                    current_chunk = [current_chunk[-1]] if current_chunk else []
                    current_size = len(current_chunk[0]) + 2 if current_chunk else 0
            
            current_chunk.append(para)
            current_size += para_size
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        # If no paragraphs, fall back to simple chunking
        if not chunks:
            for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
                chunk = content[i:i + self.chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
        
        return chunks
    
    def _create_commit_document(self, commit: Dict) -> str:
        """Create a rich commit document with context"""
        doc = f"Commit by {commit['author']} on {commit['date']}\n\n"
        doc += f"Message: {commit['message']}\n\n"
        
        if commit['files_changed']:
            doc += f"Files changed ({len(commit['files_changed'])}):\n"
            doc += '\n'.join(f"  - {f}" for f in commit['files_changed'][:20])
            if len(commit['files_changed']) > 20:
                doc += f"\n  ... and {len(commit['files_changed']) - 20} more files"
        
        doc += f"\n\nStats: +{commit['stats']['additions']} -{commit['stats']['deletions']}"
        return doc
    
    def _create_pr_document(self, pr: Dict) -> str:
        """Create a rich PR document with context"""
        doc = f"Pull Request #{pr['number']}: {pr['title']}\n\n"
        doc += f"Author: {pr['author']}\n"
        doc += f"State: {pr['state']}\n"
        doc += f"Created: {pr['created_at']}\n"
        
        if pr.get('merged_at'):
            doc += f"Merged: {pr['merged_at']}\n"
        
        doc += f"\nDescription:\n{pr['body']}\n\n"
        
        if pr['files']:
            doc += f"Files changed ({len(pr['files'])}):\n"
            doc += '\n'.join(f"  - {f}" for f in pr['files'][:30])
            if len(pr['files']) > 30:
                doc += f"\n  ... and {len(pr['files']) - 30} more files"
        
        if pr['comments']:
            doc += f"\n\nComments ({len(pr['comments'])}):\n"
            for comment in pr['comments'][:5]:
                doc += f"\n{comment[:300]}{'...' if len(comment) > 300 else ''}\n"
        
        return doc
    
    def process_and_store(self, repo_data: Dict, repo_name: str):
        """Process repo data with improved chunking and store in ChromaDB"""
        print(f"Processing {repo_name} with enhanced chunking...")
        
        documents = []
        metadatas = []
        ids = []
        id_counter = 0
        
        # Process commits with rich context
        print(f"Processing {len(repo_data.get('commits', []))} commits...")
        for commit in repo_data.get("commits", []):
            content = self._create_commit_document(commit)
            
            documents.append(content)
            metadatas.append({
                "type": "commit",
                "sha": commit["sha"],
                "date": commit["date"],
                "author": commit["author"],
                "additions": commit['stats']['additions'],
                "deletions": commit['stats']['deletions']
            })
            ids.append(f"commit_{id_counter}")
            id_counter += 1
        
        # Process PRs with rich context
        print(f"Processing {len(repo_data.get('pull_requests', []))} pull requests...")
        for pr in repo_data.get("pull_requests", []):
            content = self._create_pr_document(pr)
            
            documents.append(content)
            metadatas.append({
                "type": "pr",
                "number": pr["number"],
                "state": pr["state"],
                "date": pr["created_at"],
                "author": pr["author"],
                "title": pr["title"]
            })
            ids.append(f"pr_{id_counter}")
            id_counter += 1
        
        # Process files with smart chunking
        print(f"Processing {len(repo_data.get('files', []))} files with smart chunking...")
        for file in repo_data.get("files", []):
            chunks_with_meta = self._smart_chunk_code(file["content"], file["path"])
            
            for chunk, chunk_meta in chunks_with_meta:
                documents.append(chunk)
                metadatas.append({
                    "type": "code",
                    "file_path": file["path"],
                    "chunk_index": chunk_meta["chunk_index"],
                    "total_chunks": chunk_meta["total_chunks"],
                    "file_size": file["size"]
                })
                ids.append(f"code_{id_counter}_{chunk_meta['chunk_index']}")
            id_counter += 1
        
        print(f"Total documents to embed: {len(documents)}")
        
        # Set up persist directory
        persist_dir = f"./chroma_db_{repo_name.replace('/', '_')}"
        
        # Remove old database if it exists
        if os.path.exists(persist_dir):
            print(f"Removing old database at {persist_dir}...")
            shutil.rmtree(persist_dir)
        
        print(f"Creating new ChromaDB with Sentence Transformers at {persist_dir}...")
        
        # Create ChromaDB client (ChromaDB 0.4.x compatible)
        client = chromadb.PersistentClient(path=persist_dir)
        
        # Delete collection if it exists
        try:
            client.delete_collection(name="repo_data")
        except:
            pass
        
        # Create new collection with our embedding function
        collection = client.create_collection(
            name="repo_data",
            embedding_function=self._get_embedding_function(),
            metadata={"hnsw:space": "cosine"}
        )
        
        # Add documents in batches
        batch_size = 100  # Sentence Transformers can handle larger batches
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            batch_num = i // batch_size + 1
            print(f"Embedding batch {batch_num}/{total_batches}: documents {i} to {end_idx}")
            
            try:
                collection.add(
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )
            except Exception as e:
                print(f"⚠️  Error in batch {batch_num}: {str(e)}")
                # Try individual documents
                for j in range(i, end_idx):
                    try:
                        collection.add(
                            documents=[documents[j]],
                            metadatas=[metadatas[j]],
                            ids=[ids[j]]
                        )
                    except Exception as e2:
                        print(f"   Skipping document {j}: {str(e2)[:100]}")
                        continue
        
        print(f"✅ Successfully stored {len(documents)} documents with enhanced embeddings")
        return collection