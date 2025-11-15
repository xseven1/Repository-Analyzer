from openai import OpenAI
from tools import RepoTools
import os

class RepoAgent:
    def __init__(self, repo_tools):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=60.0,
            max_retries=2
        )
        self.repo_tools = repo_tools
        self.tools = repo_tools.get_openai_tools()  # Need to convert tool format
        self.max_tokens = 120000  # GPT-4 Mini has 128K context, leave buffer
    
    def _count_tokens(self, messages: list) -> int:
        """Rough token estimation - 1 token ≈ 4 characters"""
        total = 0
        for message in messages:
            # Count role
            total += 4
            
            # Count content
            if isinstance(message.get("content"), str):
                total += len(message["content"]) // 4
            elif isinstance(message.get("content"), list):
                for item in message["content"]:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            total += len(item.get("text", "")) // 4
                        elif item.get("type") == "tool_call":
                            total += len(str(item)) // 4
        
        # Add tokens for tools definition (~2000 tokens)
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
        
        # Keep last 8 messages (4 exchanges)
        keep_count = min(8, len(messages))
        trimmed = messages[-keep_count:]
        
        new_token_count = self._count_tokens(trimmed)
        print(f"✂️  Trimmed from {current_tokens} to {new_token_count} tokens")
        
        return trimmed
        
    def query(self, question: str, max_iterations: int = 10) -> str:
        """Execute agentic loop with GPT-4 Mini"""
        try:
            messages = [{"role": "user", "content": question}]
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                print(f"\n--- Iteration {iteration} ---")
                
                # Check token count before API call
                current_tokens = self._count_tokens(messages)
                print(f"📊 Current conversation: ~{current_tokens:,} tokens")
                
                if current_tokens > self.max_tokens:
                    messages = self._trim_conversation_history(messages)
                    current_tokens = self._count_tokens(messages)
                
                # Make API call
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",  # GPT-4 Mini
                        messages=messages,
                        tools=self.tools,
                        tool_choice="auto"
                    )
                except Exception as api_error:
                    error_msg = str(api_error)
                    if "context_length_exceeded" in error_msg.lower():
                        print(f"❌ Context too long even after trimming. Using minimal context...")
                        messages = [{"role": "user", "content": question}]
                        response = self.client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=messages,
                            tools=self.tools,
                            tool_choice="auto"
                        )
                    else:
                        raise
                
                message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason
                
                print(f"Finish reason: {finish_reason}")
                
                # Add assistant response to conversation
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": message.tool_calls
                })
                
                # Check if we're done
                if finish_reason == "stop":
                    return message.content or "No response generated."
                
                # Process tool calls
                if finish_reason == "tool_calls" and message.tool_calls:
                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = eval(tool_call.function.arguments)  # Parse JSON string
                        
                        print(f"Tool: {function_name}, Input: {function_args}")
                        
                        # Execute the tool
                        result = self._execute_tool(function_name, function_args)
                        
                        # Truncate result if too long
                        result = self._truncate_tool_result(result, max_tokens=15000)
                        
                        print(f"Result preview: {result[:200]}...")
                        print(f"Result size: ~{len(result) // 4} tokens")
                        
                        # Add tool result to conversation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": result
                        })
                else:
                    # Unexpected finish reason
                    break
            
            return "Maximum iterations reached without final answer."
            
        except Exception as e:
            return f"Error processing query: {str(e)}\n\nPlease try rephrasing your question."
    
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