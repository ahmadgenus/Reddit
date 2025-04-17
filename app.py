

import streamlit as st
from chatbot import setup_db, fetch_reddit_data, get_chatbot_response, get_db_conn
import json
import re
from urllib.parse import urlparse

# Helper function to remove image URLs from text.
def remove_image_urls(text):
    # Regex pattern to remove URLs that end with typical image extensions.
    image_url_pattern = r'https?://\S+\.(?:png|jpg|jpeg|webp)\S*'
    return re.sub(image_url_pattern, '', text)

# Initialize DB and configure Streamlit
setup_db()
st.set_page_config(layout="wide")
st.title("🚀 Reddit Intelligent Chatbot")

# Use session state to store selected post ID for chat context
if 'selected_post_id' not in st.session_state:
    st.session_state['selected_post_id'] = None

# Sidebar: Enter keyword and fetch data
with st.sidebar:
    st.header("🔍 Fetch Reddit Data")
    keyword = st.text_input("Enter keyword/topic:")
    days = st.slider("Days range:", 1, 90, 7)
    if st.button("Fetch Data"):
        fetch_reddit_data(keyword, days=days)
        st.success("Data fetched!")
        st.session_state['selected_post_id'] = None  # Reset selected post on new fetch

# Create two columns: Chat area (left) and Posts display (right)
chat_col, posts_col = st.columns([3, 2])

# RIGHT SIDE: Display fetched posts with resource-gather style card layout.
with posts_col:
    st.subheader("📋 Reddit Posts")
    if keyword.strip():
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT reddit_id, title, post_text, comments, metadata, created_at
            FROM reddit_posts
            WHERE keyword = %s
            ORDER BY created_at DESC;
        """, (keyword,))
        posts = cur.fetchall()
        cur.close()
        conn.close()
        
        if posts:
            # Set up custom CSS for the card layout.
            st.markdown(
                """
                <style>
                .post-card {
                    border: 1px solid #ddd;
                    padding: 10px;
                    margin-bottom: 20px;
                    border-radius: 8px;
                    background-color: #f9f9f9;
                }
                .post-title {
                    font-size: 14px;
                    font-weight: bold;
                    margin-bottom: 5px;
                }
                .post-snippet {
                    font-size: 12px;
                    color: #444;
                    margin-bottom: 8px;
                }
                .post-meta {
                    font-size: 11px;
                    color: #777;
                    margin-bottom: 5px;
                }
                .reddit-logo {
                    width: 24px;
                    vertical-align: middle;
                    margin-right: 8px;
                }
                </style>
                """, unsafe_allow_html=True)
            
            for post in posts:
                reddit_id, title, post_text, comments, metadata, created_at = post
                try:
                    comments_list = json.loads(comments) if isinstance(comments, str) else comments
                except Exception:
                    comments_list = comments
                
                post_url = metadata.get('url', "#")
                subreddit = metadata.get('subreddit', 'N/A')
                created_str = created_at.strftime('%Y-%m-%d %H:%M:%S')
                
                # Remove image URLs from post_text so that they don't show up in the snippet.
                cleaned_text = remove_image_urls(post_text)
                snippet = cleaned_text[:200] + ("..." if len(cleaned_text) > 200 else "")
                
                # Build the card using HTML. The title itself is a clickable link.
                card_html = f"""
<div class="post-card">
    <div style="display: flex; align-items: center;">
<a href="{post_url}" target="_blank" style="font-size: 14px; font-weight: bold; color: blue;">{title}</a>
    </div>
    <div class="post-meta">
        Subreddit: {metadata.get('subreddit','N/A')} <br> Created: {created_str}
    </div>
    <div class="post-snippet">
        {snippet}
    </div>
</div>
"""
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Expander for comments preview (first 3 comments)
                with st.expander("Show Comments Preview"):
                    if comments_list:
                        for idx, comment in enumerate(comments_list[:3], start=1):
                            st.write(f"{idx}. {comment[:100]}{'...' if len(comment) > 100 else ''}")
                    else:
                        st.info("No comments available.")
                
                # Button to select this post for chat context
                if st.button("Chat with this Post", key=reddit_id):
                    st.session_state['selected_post_id'] = reddit_id
                    st.success("Post selected for chat!")
            st.markdown("---")
        else:
            st.info("No posts found. Please fetch data using the sidebar.")
    else:
        st.info("Please enter a keyword and fetch data from the sidebar.")

# LEFT SIDE: Chatbot Interaction Area
with chat_col:
    st.subheader("💬 Chat with AI")
    question = st.text_area("Enter your question:", height=150)
    context_choice = st.radio("Chat Context:", ["All Posts", "Selected Post"])
    reddit_id = st.session_state['selected_post_id'] if context_choice == "Selected Post" else None

    if st.button("Get Response"):
        if not keyword.strip():
            st.warning("Please enter a keyword/topic and fetch data first.")
        elif not question.strip():
            st.warning("Please enter your question.")
        else:
            with st.spinner("Generating response..."):
                response, _ = get_chatbot_response(question, keyword, reddit_id)
                st.markdown("**Chatbot Response:**")
                st.write(response)
