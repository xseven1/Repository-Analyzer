import streamlit as st
import requests
import json

# Page config with dark theme
st.set_page_config(
    page_title="Repository Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme Claude-like interface
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #1e1e1e;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #171717;
        border-right: 1px solid #2d2d2d;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #e0e0e0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
    }
    
    /* Text inputs */
    .stTextInput input {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: 1px solid #3d3d3d !important;
        border-radius: 8px !important;
    }
    
    .stTextInput input:focus {
        border-color: #5d5d5d !important;
        box-shadow: 0 0 0 1px #5d5d5d !important;
    }
    
    /* Text area */
    .stTextArea textarea {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: 1px solid #3d3d3d !important;
        border-radius: 8px !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: 1px solid #3d3d3d !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        background-color: #3d3d3d !important;
        border-color: #5d5d5d !important;
    }
    
    .stButton button[kind="primary"] {
        background-color: #4a9eff !important;
        border: none !important;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #357abd !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #2d2d2d !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin: 8px 0 !important;
        border: 1px solid #3d3d3d !important;
    }
    
    /* User messages - slightly different color */
    [data-testid="stChatMessageContent"][data-testid*="user"] {
        background-color: #252525 !important;
    }
    
    /* Assistant messages */
    .stChatMessage[data-testid="stChatMessage"] p {
        color: #e0e0e0 !important;
        line-height: 1.6 !important;
    }
    
    /* Chat input container - remove black box */
    [data-testid="stChatInputContainer"] {
        background-color: transparent !important;
        border: none !important;
    }
    
    /* Chat input */
    .stChatInput {
        background-color: transparent !important;
    }
    
    .stChatInput textarea {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border: 1px solid #3d3d3d !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
    }
    
    .stChatInput textarea:focus {
        border-color: #4a9eff !important;
        box-shadow: 0 0 0 1px #4a9eff !important;
        outline: none !important;
    }
    
    /* Remove any background from the chat input wrapper */
    [data-testid="stBottom"] {
        background-color: transparent !important;
    }
    
    [data-testid="stBottom"] > div {
        background-color: transparent !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }
    
    /* Divider */
    hr {
        border-color: #3d3d3d !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0a0a0 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #4a9eff !important;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #1a4d2e !important;
        color: #4ade80 !important;
        border-radius: 8px !important;
    }
    
    .stError {
        background-color: #4d1a1a !important;
        color: #f87171 !important;
        border-radius: 8px !important;
    }
    
    .stInfo {
        background-color: #1a3a4d !important;
        color: #60a5fa !important;
        border-radius: 8px !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #4a9eff !important;
    }
    
    /* Caption text */
    .stCaptionContainer {
        color: #888888 !important;
    }
    
    /* Markdown in chat */
    .stChatMessage code {
        background-color: #1a1a1a !important;
        color: #4a9eff !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }
    
    .stChatMessage pre {
        background-color: #1a1a1a !important;
        border: 1px solid #3d3d3d !important;
        border-radius: 8px !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e1e1e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #3d3d3d;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #5d5d5d;
    }
    
    /* Remove default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Chat container styling */
    [data-testid="stChatMessageContainer"] {
        padding: 20px 0;
    }
    
    /* Make the chat feel more spacious */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_repo' not in st.session_state:
    st.session_state.current_repo = None

# Sidebar
with st.sidebar:
    st.markdown("# 🔍 Repository Analyzer")
    st.markdown("---")
    
    st.markdown("### 📚 Index Repository")
    repo_url = st.text_input(
        "GitHub URL",
        "https://github.com/SyracuseUniversity/preprint-bot",
        label_visibility="collapsed",
        placeholder="https://github.com/owner/repo"
    )
    
    if st.button("📥 Index Repository", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            with requests.post(
                "http://localhost:8000/index",
                json={"repo_url": repo_url},
                stream=True,
                timeout=300
            ) as response:
                if response.status_code != 200:
                    st.error(f"Error: {response.text}")
                else:
                    for line in response.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                data = json.loads(line[6:])
                                
                                if data['status'] == 'progress':
                                    progress_bar.progress(data['percent'] / 100)
                                    status_text.text(data['message'])
                                
                                elif data['status'] == 'complete':
                                    progress_bar.progress(1.0)
                                    status_text.empty()
                                    st.success(f"✅ {data['repo_name']}")
                                
                                elif data['status'] == 'error':
                                    progress_bar.empty()
                                    status_text.empty()
                                    st.error(f"❌ {data['message']}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    # List indexed repos
    try:
        response = requests.get("http://localhost:8000/indexed")
        if response.status_code == 200:
            data = response.json()
            if data['count'] > 0:
                st.markdown(f"### 📦 Indexed ({data['count']})")
                for repo in data['indexed_repos']:
                    st.markdown(f"✓ `{repo}`")
            else:
                st.info("No repositories indexed yet")
    except:
        st.warning("Cannot connect to backend")
    
    st.markdown("---")
    
    # Conversation management
    st.markdown("### 💬 Conversation")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 New", use_container_width=True):
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            if st.session_state.conversation_id:
                try:
                    requests.delete(f"http://localhost:8000/conversation/{st.session_state.conversation_id}")
                except:
                    pass
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()
    
    if st.session_state.conversation_id:
        st.caption(f"🆔 {st.session_state.conversation_id[:8]}...")
        st.caption(f"💬 {len(st.session_state.messages)} messages")
    
    st.markdown("---")
    
    with st.expander("💡 Example Questions", expanded=False):
        st.markdown("""
**Analysis:**
- What changed in the last month?
- Analyze PR #5 in detail
- Show me recent commits

**Code Search:**
- Where is authentication implemented?
- Find database connection logic

**Follow-ups:**
- Tell me more about that
- Who worked on this?
- What were the implications?

**Timeline:**
- Development history of auth module
- Recent activity overview
        """)
    
    st.markdown("---")
    st.caption("Repository Analyzer v1.0")
    st.caption("Powered by GPT-4 & ChromaDB")

# Main area
st.markdown("## 💬 Chat")

# Repository selector - cleaner version
query_url = st.text_input(
    "Repository",
    repo_url,
    key="query_url",
    label_visibility="collapsed",
    placeholder="Select repository to analyze..."
)

# Check if repo changed
if st.session_state.current_repo != query_url:
    st.session_state.current_repo = query_url
    if st.session_state.messages:  # Only reset if there were messages
        st.info("🔄 Switched repository - starting new conversation")
        st.session_state.conversation_id = None
        st.session_state.messages = []

st.markdown("---")

# Display conversation history
if not st.session_state.messages:
    # Welcome message when no conversation
    st.markdown("""
    <div style='text-align: center; padding: 60px 20px; color: #888888;'>
        <h3 style='color: #ffffff; margin-bottom: 16px;'>👋 Welcome to Repository Analyzer</h3>
        <p>Start by indexing a repository, then ask questions about it.</p>
        <p style='margin-top: 20px; font-size: 14px;'>Try asking about commits, PRs, code structure, or development timeline.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])

# Chat input - always at bottom
if prompt := st.chat_input("Ask a question about the repository...", key="chat_input"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    "http://localhost:8000/query",
                    json={
                        "repo_url": query_url,
                        "question": prompt,
                        "conversation_id": st.session_state.conversation_id
                    },
                    timeout=120
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    st.session_state.conversation_id = data["conversation_id"]
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Add to message history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                else:
                    error_msg = f"**Error:** {response.json().get('detail', 'Unknown error')}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except requests.exceptions.Timeout:
                error_msg = "**Error:** Request timed out. The query might be too complex."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                error_msg = f"**Error:** {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Show conversation stats at the very bottom
if st.session_state.messages:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💬 Messages", len(st.session_state.messages))
    with col2:
        st.metric("🔄 Exchanges", len(st.session_state.messages) // 2)
    with col3:
        if st.session_state.conversation_id:
            st.caption(f"ID: {st.session_state.conversation_id[:12]}")