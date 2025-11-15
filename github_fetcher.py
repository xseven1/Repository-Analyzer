from github import Github, GithubException
from datetime import datetime
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time

class RepoFetcher:
    def __init__(self, token: str):
        print(f"Initializing with token: {token[:10]}...")
        self.github = Github(token, per_page=100)
        # Check rate limit
        rate_limit = self.github.get_rate_limit()
        print(f"Rate limit: {rate_limit.core.remaining}/{rate_limit.core.limit}")
    
    def fetch_repo_data(self, repo_name: str) -> Dict:
        """Fetch all repo data in parallel: commits, PRs, files"""
        try:
            repo = self.github.get_repo(repo_name)
            print(f"Repository found: {repo.full_name}")
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                print("Starting parallel fetch...")
                futures = {
                    executor.submit(self._fetch_commits, repo): "commits",
                    executor.submit(self._fetch_prs, repo): "pull_requests",
                    executor.submit(self._fetch_files, repo): "files"
                }
                
                results = {}
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        results[key] = future.result()
                        print(f"✓ Completed: {key} - {len(results[key])} items")
                    except Exception as e:
                        print(f"✗ Error fetching {key}: {str(e)}")
                        results[key] = []
            
            print(f"Fetched {len(results.get('commits', []))} commits, "
                  f"{len(results.get('pull_requests', []))} PRs, "
                  f"{len(results.get('files', []))} files")
            return results
            
        except GithubException as e:
            print(f"GitHub API Error: {e.status} - {e.data}")
            raise
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise
    
    def _fetch_commits(self, repo) -> List[Dict]:
        commits = []
        try:
            # Reduce from 200 to 50 commits
            commit_count = 0
            for commit in repo.get_commits():
                if commit_count >= 50:
                    break
                
                try:
                    # Limit files per commit to 10
                    files_changed = []
                    if commit.files:
                        files_changed = [f.filename for f in commit.files[:10]]
                    
                    commits.append({
                        "sha": commit.sha,
                        "message": commit.commit.message,
                        "author": commit.commit.author.name if commit.commit.author else "Unknown",
                        "date": commit.commit.author.date.isoformat() if commit.commit.author else datetime.now().isoformat(),
                        "files_changed": files_changed,
                        "stats": {
                            "additions": commit.stats.additions,
                            "deletions": commit.stats.deletions
                        }
                    })
                    commit_count += 1
                    
                    if commit_count % 25 == 0:
                        print(f"  Fetched {commit_count} commits...")
                        
                except Exception as e:
                    print(f"  Skipping commit {commit.sha[:7]}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error in _fetch_commits: {str(e)}")
        
        return commits
    
    def _fetch_prs(self, repo) -> List[Dict]:
        prs = []
        try:
            pr_count = 0
            for pr in repo.get_pulls(state='all', sort='created', direction='desc'):
                if pr_count >= 100:
                    break
                
                try:
                    # Limit file fetching
                    files = []
                    try:
                        files = [f.filename for f in list(pr.get_files())[:30]]
                    except:
                        files = []
                    
                    # Limit comments
                    comments = []
                    try:
                        comments = [c.body for c in list(pr.get_comments())[:10]]
                    except:
                        comments = []
                    
                    prs.append({
                        "number": pr.number,
                        "title": pr.title,
                        "body": pr.body or "",
                        "state": pr.state,
                        "created_at": pr.created_at.isoformat(),
                        "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
                        "author": pr.user.login if pr.user else "Unknown",
                        "files": files,
                        "comments": comments
                    })
                    pr_count += 1
                    
                    if pr_count % 25 == 0:
                        print(f"  Fetched {pr_count} PRs...")
                        
                except Exception as e:
                    print(f"  Skipping PR #{pr.number}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error in _fetch_prs: {str(e)}")
        
        return prs
    
    def _fetch_files(self, repo) -> List[Dict]:
        """Fetch current file structure with limits"""
        files = []
        max_files = 500  # Limit total files
        max_file_size = 100000  # 100KB limit per file
        
        try:
            contents = list(repo.get_contents(""))
            file_count = 0
            
            while contents and file_count < max_files:
                file_content = contents.pop(0)
                
                if file_content.type == "dir":
                    try:
                        contents.extend(repo.get_contents(file_content.path))
                    except Exception as e:
                        print(f"  Skipping directory {file_content.path}: {str(e)}")
                        continue
                else:
                    try:
                        # Skip large files
                        if file_content.size > max_file_size:
                            print(f"  Skipping large file: {file_content.path} ({file_content.size} bytes)")
                            continue
                        
                        # Skip binary files
                        if file_content.path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.exe')):
                            continue
                        
                        files.append({
                            "path": file_content.path,
                            "content": file_content.decoded_content.decode('utf-8', errors='ignore'),
                            "size": file_content.size
                        })
                        file_count += 1
                        
                        if file_count % 100 == 0:
                            print(f"  Fetched {file_count} files...")
                            
                    except UnicodeDecodeError:
                        print(f"  Skipping binary file: {file_content.path}")
                    except Exception as e:
                        print(f"  Skipping file {file_content.path}: {str(e)}")
                        continue
                        
        except Exception as e:
            print(f"Error in _fetch_files: {str(e)}")
        
        return files