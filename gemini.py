### Install required modules and set the envvar for Gemini API Key
#pip install pypdf2
#pip install chromadb
#pip install google.generativeai
#pip install langchain-google-genai
#pip install langchain
#pip install langchain_community
#pip install jupyter

#export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

#Import Python modules
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma
import os
from dotenv import load_dotenv

#Load the models
# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=google_api_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#Load the PDF and create chunks
loader = PyPDFLoader("EACC-NATIONAL-SURVEY-REPORT-2023.pdf")
text_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=512,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)
pages = loader.load_and_split(text_splitter)

#Turn the chunks into embeddings and store them in Chroma
vectordb=Chroma.from_documents(pages,embeddings)

#Configure Chroma as a retriever with top_k=5
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

#Create the retrieval chain
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

#Invoke the retrieval chain
#response=retrieval_chain.invoke({"input":"What are the top 5 most corrupt ministries? Give me the percentage of corruption in each."})
response2 = retrieval_chain.invoke({"Give me a summary of the report."})
#Print the answer to the question
print(response2["answer"])