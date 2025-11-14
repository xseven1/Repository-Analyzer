from tools import RepoTools
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
# from anthropic import Anthropic
# from langchain_anthropic import ChatAnthropic
import os

class RepoAgent:
    def __init__(self, repo_tools):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.tools = repo_tools.get_tools()
        # self.llm = ChatAnthropic(
        #     model="claude-sonnet-4-20250514",
        #     temperature=0,
        #     api_key=os.getenv("ANTHROPIC_API_KEY")
        # )
        
        # Custom verbose prompt
        template = """You are an expert software repository analyst. Your job is to provide comprehensive, detailed answers about GitHub repositories.

When answering questions:
- Provide extensive context and background information
- Include all relevant details (dates, authors, file names, commit SHAs)
- Explain the significance of changes, not just what changed
- Use multiple tools if needed to gather complete information
- Format responses with clear sections and structure
- Include code snippets when relevant
- Provide historical context and evolution of features
- Be thorough - aim for comprehensive answers rather than brief summaries

You have access to these tools:
{tools}

Tool names: {tool_names}

Format your responses with:
- Clear headings and sections
- Bullet points for lists
- Code blocks for code snippets
- Chronological timelines when relevant
- Attribution (authors, dates, commit SHAs)

Use this format:

Question: the input question you must answer
Thought: think about what information you need and which tools to use
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have enough information to provide a comprehensive answer
Final Answer: a detailed, well-structured answer to the original question

Question: {input}
Thought: {agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        
        agent = create_react_agent(self.llm, self.tools, prompt)
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,  # Allow more iterations for thorough research
            max_execution_time=60  # Give it time to be thorough
        )
    
    def query(self, question: str) -> str:
        result = self.agent_executor.invoke({"input": question})
        return result["output"]