from openai import OpenAI
from tools import RepoTools
import os
from typing import List, Dict

class RepoAgent:
    def __init__(self, repo_tools):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=60.0,
            max_retries=2
        )
        self.repo_tools = repo_tools
        self.tools = repo_tools.get_openai_tools()
        self.max_tokens = 120000
        
        # System message for the agent
        self.system_message = {
            "role": "system",
            "content": """You are an expert software repository analyst with deep experience in software architecture, development patterns, and team dynamics.

When analyzing repositories, you should:
- Provide insights and interpretations, not just raw facts
- Identify patterns, trends, and their implications
- Make informed inferences about code quality, architecture decisions, and development practices
- Connect different pieces of information to tell a coherent story
- Highlight both strengths and potential areas of concern
- Consider the "why" behind changes, not just the "what"
- Use analogies and comparisons when they help understanding
- Point out interesting observations that might not be immediately obvious
- Assess team dynamics, collaboration patterns, and code ownership
- Comment on development practices (testing, documentation, commit discipline)
- Note architectural decisions and their potential impact

Think like a senior engineer reviewing a codebase for the first time - be analytical, insightful, and helpful. Your goal is to help users deeply understand the repository's history, architecture, and evolution.

When you have gathered information from tools, synthesize it into a comprehensive narrative rather than just listing facts.

IMPORTANT: You maintain conversation history. When users ask follow-up questions:
- Reference previous exchanges naturally
- Build upon earlier analysis
- Connect new findings to previous discussions
- Maintain context across the conversation"""
        }
    
    def _count_tokens(self, messages: list) -> int:
        """Rough token estimation - 1 token ≈ 4 characters"""
        total = 0
        for message in messages:
            total += 4
            
            if isinstance(message.get("content"), str):
                total += len(message["content"]) // 4
            elif isinstance(message.get("content"), list):
                for item in message["content"]:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            total += len(item.get("text", "")) // 4
                        elif item.get("type") == "tool_call":
                            total += len(str(item)) // 4
        
        total += 2000
        return total
    
    def _truncate_tool_result(self, result: str, max_tokens: int = 10000) -> str:
        """Truncate tool result if too long"""
        max_chars = max_tokens * 4
        if len(result) <= max_chars:
            return result
        
        truncated = result[:max_chars]
        return f"{truncated}\n\n[... Result truncated due to length ...]"
    
    def _trim_conversation_history(self, messages: list) -> list:
        """Remove old messages if conversation is too long"""
        current_tokens = self._count_tokens(messages)
        
        if current_tokens <= self.max_tokens:
            return messages
        
        print(f"⚠️  Conversation too long ({current_tokens} tokens), trimming history...")
        
        # Always keep system message
        system_msg = messages[0] if messages and messages[0]["role"] == "system" else None
        other_messages = messages[1:] if system_msg else messages
        
        # Keep last 10 messages (5 exchanges)
        keep_count = min(10, len(other_messages))
        trimmed = other_messages[-keep_count:]
        
        # Re-add system message
        if system_msg:
            trimmed = [system_msg] + trimmed
        
        new_token_count = self._count_tokens(trimmed)
        print(f"✂️  Trimmed from {current_tokens} to {new_token_count} tokens")
        
        return trimmed
    
    def _needs_enhancement(self, content: str) -> bool:
        """Check if response is too factual and needs more analysis"""
        if not content:
            return False
        
        analytical_phrases = [
            "suggests that", "indicates", "implies", "pattern", 
            "trend", "notably", "interesting", "significant",
            "architecture", "design decision", "approach",
            "insight", "observation", "implication"
        ]
        
        analysis_count = sum(1 for phrase in analytical_phrases if phrase.lower() in content.lower())
        
        return analysis_count < 3
    
    def query_with_history(self, question: str, conversation_history: List[Dict], max_iterations: int = 10) -> str:
        """Execute agentic loop with conversation history"""
        try:
            # Build messages with history
            messages = [self.system_message]
            
            # Add conversation history (user/assistant pairs)
            messages.extend(conversation_history)
            
            # Add current question
            messages.append({"role": "user", "content": question})
            
            iteration = 0
            tool_calls_made = 0
            
            while iteration < max_iterations:
                iteration += 1
                print(f"\n--- Iteration {iteration} ---")
                
                current_tokens = self._count_tokens(messages)
                print(f"📊 Current conversation: ~{current_tokens:,} tokens")
                
                if current_tokens > self.max_tokens:
                    messages = self._trim_conversation_history(messages)
                    current_tokens = self._count_tokens(messages)
                
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        tools=self.tools,
                        tool_choice="auto",
                        temperature=0.7
                    )
                except Exception as api_error:
                    error_msg = str(api_error)
                    if "context_length_exceeded" in error_msg.lower():
                        print(f"❌ Context too long even after trimming. Using minimal context...")
                        messages = [self.system_message, {"role": "user", "content": question}]
                        response = self.client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages,
                            tools=self.tools,
                            tool_choice="auto",
                            temperature=0.7
                        )
                    else:
                        raise
                
                message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason
                
                print(f"Finish reason: {finish_reason}")
                
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": message.tool_calls
                })
                
                if finish_reason == "stop":
                    if tool_calls_made >= 2:
                        print("🔄 Requesting comprehensive synthesis...")
                        messages.append({
                            "role": "user",
                            "content": """Now provide a comprehensive, insightful analysis based on everything you've learned.

Structure your response to:
1. Start with a clear summary of key findings
2. Provide detailed insights with supporting evidence from the data
3. Make connections between different aspects (code, PRs, commits, timeline)
4. Highlight interesting patterns, trends, or notable observations
5. Discuss architectural decisions or development practices you observe
6. Offer practical implications or recommendations where appropriate
7. Note any concerns, risks, or areas worth investigating further

Be thorough, analytical, and insightful. Help the reader truly understand this repository."""
                        })
                        
                        final_response = self.client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages,
                            temperature=0.7
                        )
                        return final_response.choices[0].message.content or "No response generated."
                    
                    elif self._needs_enhancement(message.content or ""):
                        print("🔄 Response needs more analysis, requesting enhancement...")
                        messages.append({
                            "role": "user",
                            "content": """Please enhance your analysis with:
1. Key insights and patterns you observe in the data
2. What these findings imply about the repository's development
3. Notable observations or potential concerns
4. Connections between different pieces of information
5. Your assessment of development practices, architecture, or team dynamics

Make your response more analytical, insightful, and less purely factual. Think like a senior engineer doing a code review."""
                        })
                        
                        enhanced_response = self.client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages,
                            temperature=0.7
                        )
                        return enhanced_response.choices[0].message.content or "No response generated."
                    
                    return message.content or "No response generated."
                
                if finish_reason == "tool_calls" and message.tool_calls:
                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = eval(tool_call.function.arguments)
                        
                        print(f"Tool: {function_name}, Input: {function_args}")
                        tool_calls_made += 1
                        
                        result = self._execute_tool(function_name, function_args)
                        result = self._truncate_tool_result(result, max_tokens=15000)
                        
                        print(f"Result preview: {result[:200]}...")
                        print(f"Result size: ~{len(result) // 4} tokens")
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": result
                        })
                else:
                    break
            
            return "Maximum iterations reached without final answer."
            
        except Exception as e:
            return f"Error processing query: {str(e)}\n\nPlease try rephrasing your question."
    
    def query(self, question: str, max_iterations: int = 10) -> str:
        """Execute agentic loop without history (backward compatibility)"""
        return self.query_with_history(question, [], max_iterations)
    
    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool and return result"""
        try:
            if tool_name == "search_commits":
                return self.repo_tools.search_commits(tool_input.get("query", ""))
            elif tool_name == "get_pr_details":
                return self.repo_tools.get_pr_details(tool_input.get("pr_number", ""))
            elif tool_name == "search_code":
                return self.repo_tools.search_code(tool_input.get("query", ""))
            elif tool_name == "get_timeline":
                return self.repo_tools.get_timeline(tool_input.get("query", ""))
            elif tool_name == "get_repository_stats":
                return self.repo_tools.get_repository_stats(tool_input.get("query", ""))
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"