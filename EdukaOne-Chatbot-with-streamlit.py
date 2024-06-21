import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="Welcome to EdukaOne",
    page_icon=":brain:",  # Favicon emoji
    layout="centered",  # Page layout option
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Embedding object
embeddings = GoogleGenerativeAIEmbeddings(api_key=GOOGLE_API_KEY, model='models/embedding-001')

# Pinecone object
pc = Pinecone(api_key=PINECONE_API_KEY)

# Vectorstore object
pinecone_index = 'edukaone-new'
vectorstore = PineconeVectorStore(index_name=pinecone_index, embedding=embeddings)

# Retriever object
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Output parser object
parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

message_template = """
You are One, a clever and friendly assistant who works for EdukaOne.
You will teach every student about science nature be it mathematics, physics, biology, and chemistry.
Your job is to teach, explain, and answer their questions about science.
You also will help students to tackle their problems,
help them to answer their curiosity towards science. 
Once you have the user's answer, you will explain further more so the student will become excited to scientific topics.
Answer based on the context provided. 
You are a physics expert. Please answer my question based on the context provided. 
If you cannot find the solution in the context and you don't know the answer, answer with your base knowledge about science
and make it fun and creative so the user will understand science better.

Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_messages([("human", message_template)])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display the chatbot's title on the page
st.title("🧑‍🚀One: your study companion")

# Display the chat history
for message in st.session_state["chat_history"]:
    role, text = message.split(": ", 1)
    with st.chat_message(role):
        st.markdown(text)

# Get user input
user_prompt = st.chat_input("Ask One...")
if user_prompt:
    # Add user's message to chat and display it
    st.session_state["chat_history"].append(f"user: {user_prompt}")
    st.chat_message("user").markdown(user_prompt)

    # Send user's message to Gemini-Pro and get the response
    context = {"context": "", "question": user_prompt}
    response = chain.invoke(user_prompt)

    # Append assistant's response to chat history and display it
    st.session_state["chat_history"].append(f"assistant: {response}")
    st.chat_message("assistant").markdown(response)
