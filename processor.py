from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Dict

class RepoProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def process_and_store(self, repo_data: Dict, repo_name: str):
        """Process repo data and store in vector DB"""
        documents = []
        
        # Process commits
        for commit in repo_data["commits"]:
            doc = {
                "content": f"Commit: {commit['message']}\nAuthor: {commit['author']}\nFiles: {', '.join(commit['files_changed'])}",
                "metadata": {
                    "type": "commit",
                    "sha": commit["sha"],
                    "date": commit["date"],
                    "author": commit["author"]
                }
            }
            documents.append(doc)
        
        # Process PRs
        for pr in repo_data["pull_requests"]:
            content = f"PR #{pr['number']}: {pr['title']}\n{pr['body']}\n"
            content += f"Files changed: {', '.join(pr['files'])}"
            
            doc = {
                "content": content,
                "metadata": {
                    "type": "pr",
                    "number": pr["number"],
                    "state": pr["state"],
                    "date": pr["created_at"],
                    "author": pr["author"]
                }
            }
            documents.append(doc)
        
        # Process files (code)
        for file in repo_data["files"]:
            chunks = self.text_splitter.split_text(file["content"])
            for i, chunk in enumerate(chunks):
                doc = {
                    "content": chunk,
                    "metadata": {
                        "type": "code",
                        "file_path": file["path"],
                        "chunk_index": i
                    }
                }
                documents.append(doc)
        
        # Store in ChromaDB
        texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=f"./chroma_db_{repo_name.replace('/', '_')}"
        )
        
        return vectorstore