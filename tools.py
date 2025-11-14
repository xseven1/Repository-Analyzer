from langchain.tools import Tool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from github import Github

class RepoTools:
    def __init__(self, repo_name: str, github_token: str):
        self.repo_name = repo_name
        self.github = Github(github_token)
        self.repo = self.github.get_repo(repo_name)
        self.vectorstore = Chroma(
            persist_directory=f"./chroma_db_{repo_name.replace('/', '_')}",
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
        )
    
    def search_commits(self, query: str) -> str:
        """Search through commit history with detailed information"""
        results = self.vectorstore.similarity_search(query, k=10, filter={"type": "commit"})
        
        if not results:
            return "No commits found matching the query."
        
        output = f"=== COMMIT SEARCH RESULTS ===\n"
        output += f"Found {len(results)} relevant commits for query: '{query}'\n\n"
        
        for i, doc in enumerate(results, 1):
            output += f"--- Commit #{i} ---\n"
            output += f"Commit SHA: {doc.metadata['sha']}\n"
            output += f"Author: {doc.metadata['author']}\n"
            output += f"Date: {doc.metadata['date']}\n"
            output += f"Message:\n{doc.page_content}\n"
            output += f"GitHub URL: https://github.com/{self.repo_name}/commit/{doc.metadata['sha']}\n"
            output += "\n"
        
        return output
    
    def get_pr_details(self, pr_number: str) -> str:
        """Get comprehensive details about a specific PR"""
        try:
            pr = self.repo.get_pull(int(pr_number))
            
            output = f"=== PULL REQUEST #{pr.number} DETAILS ===\n\n"
            output += f"Title: {pr.title}\n"
            output += f"State: {pr.state.upper()}\n"
            output += f"Author: {pr.user.login}\n"
            output += f"Created: {pr.created_at}\n"
            
            if pr.merged:
                output += f"Merged: {pr.merged_at}\n"
                output += f"Merged by: {pr.merged_by.login if pr.merged_by else 'Unknown'}\n"
            
            output += f"GitHub URL: {pr.html_url}\n\n"
            
            output += f"--- Description ---\n"
            output += f"{pr.body if pr.body else 'No description provided'}\n\n"
            
            # Files changed
            files = list(pr.get_files())
            output += f"--- Files Changed ({len(files)}) ---\n"
            for f in files[:20]:  # Limit to first 20 files
                output += f"  • {f.filename} (+{f.additions} -{f.deletions})\n"
            if len(files) > 20:
                output += f"  ... and {len(files) - 20} more files\n"
            output += "\n"
            
            # Statistics
            output += f"--- Statistics ---\n"
            output += f"Total changes: +{pr.additions} -{pr.deletions}\n"
            output += f"Commits: {pr.commits}\n"
            output += f"Comments: {pr.comments}\n"
            output += f"Review comments: {pr.review_comments}\n\n"
            
            # Comments
            comments = list(pr.get_comments())
            if comments:
                output += f"--- Discussion ({len(comments)} comments) ---\n"
                for i, comment in enumerate(comments[:5], 1):
                    output += f"\nComment {i} by {comment.user.login} on {comment.created_at}:\n"
                    output += f"{comment.body[:200]}{'...' if len(comment.body) > 200 else ''}\n"
                if len(comments) > 5:
                    output += f"\n... and {len(comments) - 5} more comments\n"
            
            # Labels
            if pr.labels:
                output += f"\n--- Labels ---\n"
                for label in pr.labels:
                    output += f"  • {label.name}\n"
            
            return output
            
        except Exception as e:
            return f"Error fetching PR #{pr_number}: {str(e)}\nPlease verify the PR number exists in this repository."
    
    def search_code(self, query: str) -> str:
        """Search for code snippets with detailed context"""
        results = self.vectorstore.similarity_search(query, k=5, filter={"type": "code"})
        
        if not results:
            return "No code snippets found matching the query."
        
        output = f"=== CODE SEARCH RESULTS ===\n"
        output += f"Found {len(results)} relevant code snippets for: '{query}'\n\n"
        
        for i, doc in enumerate(results, 1):
            output += f"--- Result #{i} ---\n"
            output += f"File: {doc.metadata['file_path']}\n"
            output += f"Repository path: https://github.com/{self.repo_name}/blob/main/{doc.metadata['file_path']}\n"
            output += f"\nCode:\n```\n{doc.page_content}\n```\n\n"
        
        return output
    
    def get_timeline(self, query: str) -> str:
        """Get detailed timeline of changes"""
        results = self.vectorstore.similarity_search(query, k=15)
        
        if not results:
            return "No timeline information found for the query."
        
        # Sort by date
        sorted_results = sorted(
            results, 
            key=lambda x: x.metadata.get('date', ''), 
            reverse=True
        )
        
        output = f"=== TIMELINE ANALYSIS ===\n"
        output += f"Chronological history for: '{query}'\n\n"
        
        # Group by type
        commits = [r for r in sorted_results if r.metadata.get('type') == 'commit']
        prs = [r for r in sorted_results if r.metadata.get('type') == 'pr']
        
        if prs:
            output += f"--- Pull Requests ({len(prs)}) ---\n"
            for pr in prs:
                output += f"\n[{pr.metadata.get('date', 'Unknown date')}]\n"
                output += f"PR #{pr.metadata.get('number')} by {pr.metadata.get('author')}\n"
                output += f"Status: {pr.metadata.get('state')}\n"
                output += f"{pr.page_content[:150]}...\n"
        
        if commits:
            output += f"\n--- Commits ({len(commits)}) ---\n"
            for commit in commits[:10]:  # Show first 10 commits
                output += f"\n[{commit.metadata.get('date', 'Unknown date')}]\n"
                output += f"Commit {commit.metadata.get('sha', 'Unknown')[:7]} by {commit.metadata.get('author')}\n"
                output += f"{commit.page_content[:150]}...\n"
        
        output += f"\n--- Summary ---\n"
        output += f"Total events tracked: {len(sorted_results)}\n"
        output += f"Date range: {sorted_results[-1].metadata.get('date', 'Unknown')} to {sorted_results[0].metadata.get('date', 'Unknown')}\n"
        
        return output
    
    def get_repository_stats(self, query: str) -> str:
        """Get overall repository statistics"""
        try:
            output = f"=== REPOSITORY STATISTICS ===\n"
            output += f"Repository: {self.repo.full_name}\n\n"
            
            output += f"--- Overview ---\n"
            output += f"Description: {self.repo.description or 'No description'}\n"
            output += f"Language: {self.repo.language}\n"
            output += f"Created: {self.repo.created_at}\n"
            output += f"Last updated: {self.repo.updated_at}\n"
            output += f"Stars: {self.repo.stargazers_count:,}\n"
            output += f"Forks: {self.repo.forks_count:,}\n"
            output += f"Open issues: {self.repo.open_issues_count:,}\n"
            output += f"Watchers: {self.repo.watchers_count:,}\n"
            output += f"Size: {self.repo.size:,} KB\n\n"
            
            output += f"--- Activity ---\n"
            commits = list(self.repo.get_commits()[:10])
            output += f"Recent commits: {len(commits)}\n"
            
            contributors = list(self.repo.get_contributors()[:10])
            output += f"\nTop Contributors:\n"
            for i, contrib in enumerate(contributors, 1):
                output += f"  {i}. {contrib.login} - {contrib.contributions} contributions\n"
            
            output += f"\n--- Repository URLs ---\n"
            output += f"GitHub: {self.repo.html_url}\n"
            if self.repo.homepage:
                output += f"Homepage: {self.repo.homepage}\n"
            
            return output
            
        except Exception as e:
            return f"Error fetching repository statistics: {str(e)}"
    
    def get_tools(self):
        return [
            Tool(
                name="search_commits",
                func=self.search_commits,
                description="Search through commit history with detailed information about commits, authors, dates, and changes. Use this when you need to find when specific changes were made or who made them."
            ),
            Tool(
                name="get_pr_details",
                func=self.get_pr_details,
                description="Get comprehensive details about a specific pull request including description, files changed, comments, and statistics. Input should be just the PR number (e.g., '43'). Use this when asked about a specific PR."
            ),
            Tool(
                name="search_code",
                func=self.search_code,
                description="Search for code snippets and implementations in the repository with file paths and context. Use this to find where specific functionality is implemented or to locate code patterns."
            ),
            Tool(
                name="get_timeline",
                func=self.get_timeline,
                description="Get a detailed chronological timeline of changes related to a topic, including commits and PRs sorted by date. Use this for 'journey', 'history', or 'evolution' questions."
            ),
            Tool(
                name="get_repository_stats",
                func=self.get_repository_stats,
                description="Get overall repository statistics including contributors, stars, activity metrics, and general information. Use this for high-level questions about the repository."
            )
        ]