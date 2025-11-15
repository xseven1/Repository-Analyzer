import streamlit as st
import requests
import json

st.set_page_config(page_title="Repository Analyzer", page_icon="🔍", layout="wide")

st.title("🔍 Repository Analyzer")
st.markdown("*Ask detailed questions about any GitHub repository*")

# Sidebar
with st.sidebar:
    st.header("📚 Index Repository")
    repo_url = st.text_input("GitHub URL", "https://github.com/fastapiutils/fastapi-utils")
    
    if st.button("Index Repository", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            with requests.post(
                "http://localhost:8000/index",
                json={"repo_url": repo_url},
                stream=True,
                timeout=300  # 5 minute timeout
            ) as response:
                if response.status_code != 200:
                    st.error(f"❌ Error: {response.text}")
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
            st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    # List indexed repos
    response = requests.get("http://localhost:8000/indexed")
    if response.status_code == 200:
        data = response.json()
        if data['count'] > 0:
            st.subheader(f"Indexed Repositories ({data['count']})")
            for repo in data['indexed_repos']:
                st.text(f"✓ {repo}")
    
    st.divider()
    
    with st.expander("💡 Example Questions"):
        st.markdown("""
        **Detailed Analysis:**
        - Tell me the complete history of this repository
        - What is in PR #43 with all details?
        - Give me a comprehensive overview of recent changes
        
        **Code Investigation:**
        - Where is [specific function] implemented? Show me the code
        - Find all authentication-related code
        - Show me the implementation of [feature]
        
        **Timeline & History:**
        - Tell me the journey of the authentication module
        - What changed in the last 6 months?
        - Who are the main contributors and what did they work on?
        
        **Statistics:**
        - Give me detailed repository statistics
        - What are the most active areas of the codebase?
        """)

# Main area
st.header("💬 Ask Questions")

col1, col2 = st.columns([3, 1])
with col1:
    query_url = st.text_input("Repository URL", repo_url, key="query_url")
with col2:
    st.write("")
    st.write("")

question = st.text_area(
    "Your Question",
    "Tell me everything about PR #1 in great detail",
    height=100,
    help="Ask detailed questions - the agent will provide comprehensive answers"
)

if st.button("🔍 Ask Question", type="primary"):
    with st.spinner("🤖 Analyzing repository and gathering detailed information..."):
        response = requests.post(
            "http://localhost:8000/query",
            json={"repo_url": query_url, "question": question}
        )
        
        if response.status_code == 200:
            data = response.json()
            st.success(f"**Repository:** {data['repo_name']}")
            
            # Display answer in an expandable container
            with st.container():
                st.markdown("### 📋 Detailed Answer")
                st.markdown(data["answer"])
                
                # Add copy button
                st.download_button(
                    label="📄 Download Answer",
                    data=data["answer"],
                    file_name=f"repo_analysis_{data['repo_name'].replace('/', '_')}.txt",
                    mime="text/plain"
                )
        else:
            st.error(f"❌ Error: {response.json()['detail']}")
