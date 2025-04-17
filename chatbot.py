

from langchain.chains import LLMChain
# chat_chain = LLMChain(
#     llm=llm,
#     prompt=chat_prompt,
#     memory=memory,
#     verbose=True  # Enable verbose logging for debugging
# )


import os
import psycopg2
import praw
import json
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

# Initialize the LLM via LangChain (using Groq)
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    # model_name=os.getenv("MODEL_NAME"),
    model_name= "meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0.2
)

# Embedding Model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Reddit API Setup
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

# Database connection function
import psycopg2
import os

def get_db_conn():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

# Set up the database schema: store raw post text, comments, computed embedding, and metadata.
def setup_db():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS reddit_posts (
            id SERIAL PRIMARY KEY,
            reddit_id VARCHAR(50) UNIQUE,
            keyword TEXT,
            title TEXT,
            post_text TEXT,
            comments JSONB,
            created_at TIMESTAMP,
            embedding VECTOR(384),
            metadata JSONB
        );
        CREATE INDEX IF NOT EXISTS idx_keyword_created_at ON reddit_posts(keyword, created_at DESC);
    """)
    conn.commit()
    cur.close()
    conn.close()

# Utility: Check if the keyword appears in the post title, selftext, or any comment.
def keyword_in_post_or_comments(post, keyword):
    keyword_lower = keyword.lower()
    combined_text = (post.title + " " + post.selftext).lower()
    if keyword_lower in combined_text:
        return True
    post.comments.replace_more(limit=None)
    for comment in post.comments.list():
        if keyword_lower in comment.body.lower():
            return True
    return False

# Fetch Reddit posts if the keyword is in the post or any comment.
# This version iterates over posts until reaching posts older than the specified day range.
def fetch_reddit_data(keyword, days=7, limit=None):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    subreddit = reddit.subreddit("all")
    posts_generator = subreddit.search(keyword, sort="new", time_filter="all", limit=limit)
    
    data = []
    for post in posts_generator:
        created = datetime.utcfromtimestamp(post.created_utc)
        if created < start_time:
            break  # Since sorted by new, we break once older posts are encountered.
        if not keyword_in_post_or_comments(post, keyword):
            continue

        post.comments.replace_more(limit=None)
        comments = [comment.body for comment in post.comments.list()]
        combined_text = f"{post.title}\n{post.selftext}\n{' '.join(comments)}"
        embedding = embedder.encode(combined_text).tolist()
        metadata = {
            "url": post.url,
            "subreddit": post.subreddit.display_name,
            "comments_count": len(comments)
        }
        data.append({
            "reddit_id": post.id,
            "keyword": keyword,
            "title": post.title,
            "post_text": post.selftext,
            "comments": comments,
            "created_at": created,
            "embedding": embedding,
            "metadata": metadata
        })
    save_to_db(data)

# Save posts data into PostgreSQL.
def save_to_db(posts):
    conn = get_db_conn()
    cur = conn.cursor()
    for post in posts:
        cur.execute("""
            INSERT INTO reddit_posts
            (reddit_id, keyword, title, post_text, comments, created_at, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
        """, (
            post["reddit_id"],
            post["keyword"],
            post["title"],
            post["post_text"],
            json.dumps(post["comments"]),
            post["created_at"],
            post["embedding"],
            json.dumps(post["metadata"])
        ))
    conn.commit()
    cur.close()
    conn.close()

# Retrieve context from the DB.
# Updated retrieval: if summarization intent is detected, retrieve more posts.
def retrieve_context(question, keyword, reddit_id=None, top_k=10):
    lower_q = question.lower()
    # Check for summarization intent.
    if any(word in lower_q for word in ["summarize", "overview", "all posts"]):
        requested_top_k = 50
    else:
        requested_top_k = top_k

    # Retrieve posts based on query embedding.
    query_embedding = embedder.encode(question).tolist()
    query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
    
    conn = get_db_conn()
    cur = conn.cursor()
    if reddit_id:
        cur.execute("""
            SELECT title, post_text, comments FROM reddit_posts
            WHERE reddit_id = %s;
        """, (reddit_id,))
    else:
        cur.execute("""
            SELECT title, post_text, comments FROM reddit_posts
            WHERE keyword = %s
            ORDER BY embedding <-> %s::vector LIMIT %s;
        """, (keyword, query_embedding_str, requested_top_k))
    results = cur.fetchall()
    conn.close()
    
    # If there are fewer posts than requested and none were retrieved by vector search,
    # fall back to retrieving all posts for that keyword.
    if not results:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT title, post_text, comments FROM reddit_posts
            WHERE keyword = %s ORDER BY created_at DESC;
        """, (keyword,))
        results = cur.fetchall()
        conn.close()
    return results

# --- New Summarization Step for Handling Long Context ---
# Create a summarization chain to compress the context if it exceeds a token/character threshold.
summarize_prompt = ChatPromptTemplate.from_template("""
You are a summarizer. Summarize the following context from Reddit posts into a concise summary that preserves the key insights. Do not add extra commentary.

Context:
{context}

Summary:
""")
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)


# Set up conversation memory and chain.
memory = ConversationBufferMemory(memory_key="chat_history")
# Updated prompt: we now expect a single input field "input"
chat_prompt = ChatPromptTemplate.from_template("""
Chat History:
{chat_history}

Context from Reddit and User Question:
{input}

Act as an Professional Assistant as incremental chat agent and also give reasioning and Answer clearly based on context and chat history, your response should be valid and concise, and relavant .

""")

chat_chain = LLMChain(
    llm=llm,
    prompt=chat_prompt,
    memory=memory,
    verbose=True  # Enable verbose logging for debugging
)

# Get chatbot response by merging context and question into a single input.
# Updated get_chatbot_response to handle summarization if context is too long.

def get_chatbot_response(question, keyword, reddit_id=None):
    context_posts = retrieve_context(question, keyword, reddit_id)
    context = "\n\n".join([f"{p[0]}:\n{p[1]}" for p in context_posts])
    
    # Set a threshold (e.g., 3000 characters); if context length exceeds it, compress the context.
    if len(context) > 3000:
        context = summarize_chain.run({"context": context})
    
    combined_input = f"Context:\n{context}\n\nUser Question: {question}"
    response = chat_chain.run({"input": combined_input})
    return response, context_posts