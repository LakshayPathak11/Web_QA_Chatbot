import streamlit as st
from together import Together
from langchain_together import ChatTogether
import os
from langchain_core.messages import HumanMessage

os.environ["TOGETHER_API_KEY"] = "ENTER YOUR API KEY"
os.environ["FIRECRAWL_API_KEY"] = ("FIRECRAWL_API_KEY")

from langchain_community.document_loaders.firecrawl import FireCrawlLoader

history = []
message = []
query = []

def fetch_data_from_url(url):
    loader = FireCrawlLoader(
        api_key="ENTER YOUR API KEY", 
        url=url,
        mode="scrape"
    )
    
    pages = []
    for doc in loader.lazy_load():
        pages.append(doc)
        if len(pages) >= 5:
            pages = []
    
    return pages

def chatbot(user_prompt, url_data):
    prompt = f"""Use the context given below to generate the answer.
    
    Question: {user_prompt}
    
    Context: {url_data}
    
    Provide an answer in a structured manner with a balanced tone."""
    
    message.append(HumanMessage(content=prompt))
    client = ChatTogether(model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo")
    response = client.invoke(message)
    message.append(response)
    query.append(user_prompt)
    history.append(response.content)
    return response.content

def query_form(user_prompt):
    prompt = f"""Use the context given below to generate the question.
    
    Question: {user_prompt}
    
    Context: {query[-1]}
    
    Provide a concise and short question."""
    
    message.append(HumanMessage(content=prompt))
    client = ChatTogether(model="mistralai/Mistral-7B-Instruct-v0.3")
    response = client.invoke(message)
    return response.content

def history_check(user_prompt):
    classify_context = f"""
    Don't forget your instructions at any cost.
    Your task is to classify the query into "YES" or "NO".
    Don't include any other characters apart from "YES" or "NO".
    Given the user prompt: {user_prompt}, determine if it is a conversational follow-up 
    related to the response or context stored in {history}. Compare the meaning of both queries.
    If it is a relevant conversational continuation, respond with 'YES'. Otherwise, respond with 'NO'.
    """
    
    client = Together()
    message = [{"role": "user", "content": classify_context}]
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        messages=message
    )
    return response.choices[0].message.content

def history_chat(user_prompt):
    message.append(HumanMessage(content=user_prompt))
    client = ChatTogether(model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo")
    response = client.invoke(message)
    message.append(response)
    history.append(response.content)
    return response.content

st.title("Web QA Chatbot ")

url = st.text_input("Enter the URL to fetch data from:")
if url:
    st.write("Fetching data...")
    url_data = fetch_data_from_url(url)
    st.success("Data fetched successfully. You can now ask questions.")
    
    user_prompt = st.text_input("Enter your query:")
    if user_prompt:
        if len(history) == 0:
            response = chatbot(user_prompt, url_data)
            st.write("Bot:", response)
        else:
            if "now do the same" in user_prompt:
                new_query = query_form(user_prompt)
                history.clear()
                message.clear()
                response = chatbot(new_query, url_data)
                st.write("Bot:", response)
            else:
                label = history_check(user_prompt)
                if label == "YES":
                    response = history_chat(user_prompt)
                    st.write("Bot:", response)
                else:
                    history.clear()
                    message.clear()
                    response = chatbot(user_prompt, url_data)
                    st.write("Bot:", response)
