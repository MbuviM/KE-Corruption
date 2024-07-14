from flask import Flask, request, jsonify, render_template
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize models
llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=google_api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=google_api_key)

# Load the PDF and create chunks
file_path = "EACC-NATIONAL-SURVEY-REPORT-2023.pdf"
pdf_loader = PyPDFLoader(file_path)
docs = pdf_loader.load()

# Chunking or Text Splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=0
)
chunks = text_splitter.split_documents(docs)

# Initialize VectorDB and Retriever
persist_directory = "./chroma"  # Specify a directory for Chroma to persist data
vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
vectordb.persist()

# Configure Chroma as a retriever with top_k=5
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# Define Retrieval Chain
template = """
You are a helpful AI assistant.
Answer based on the context provided. 
context: {context}
input: {input}
answer:
"""
prompt = PromptTemplate.from_template(template)

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_input = request.form['user_input']
    response = retrieval_chain.invoke({"input": user_input})
    return jsonify(response["answer"])

if __name__ == "__main__":
    app.run(debug=True)
