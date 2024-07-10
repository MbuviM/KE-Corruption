# %% [markdown]
# ## Install needed libraries

# %%
# %pip install --upgrade --quiet pypdf
# %pip install langchain
# %pip install 
# %pip install langchain-huggingface
# %pip uninstall transformers huggingface_hub sentence-transformers -y
# %pip install transformers==4.28.1 huggingface_hub==0.14.1 sentence-transformers==2.2.0
# %pip install -U sentence-transformers
# %pip install chromadb

# %% [markdown]
# ## Load PDF Document 

# %%
from langchain import *
from langchain_community.document_loaders import PyPDFLoader

# %%
file_path = "EACC-NATIONAL-SURVEY-REPORT-2023.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load()
print("Loaded pages:", len(pages))  # Debug statement
print("First page content:", pages[0].page_content[:500])  # Debug statement

# %% [markdown]
# ## Chunking

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter

# %%
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=0
)

chunks = text_splitter.split_documents(pages)
print("Number of chunks:", len(chunks))  # Debug statement
print("Sample chunk:", chunks[3].page_content[:500])  # Debug statement

# %% [markdown]
# ## Embedding

# %%
from langchain_huggingface import HuggingFaceEmbeddings

# %%
import sentence_transformers
embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

# %% [markdown]
# ## Vector Database

# %%
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# %%
# initialize the vector store (save to disk)
db = Chroma.from_documents(chunks, embeddings_model, persist_directory="./chroma_db")
print("Vector store initialized and saved.")  # Debug statement

# %%
query = "What is the most corrupt ministry?"

# %%
# retrieve from vector db (load from disk) with query
db2 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)
retrieved_docs = db2.similarity_search(query)
print("Retrieved document content:", retrieved_docs[0].page_content[:500])  # Debug statement

# %%
# initialize the retriever
retriever = db2.as_retriever(
    search_type="mmr",  # similarity
    search_kwargs={'k': 4}
)

# %%
# Create a tokenizer object by loading the pretrained "Intel/dynamic_tinybert" tokenizer.
tokenizer = AutoTokenizer.from_pretrained("Intel/dynamic_tinybert")

# Create a question-answering model object by loading the pretrained "Intel/dynamic_tinybert" model.
model = AutoModelForQuestionAnswering.from_pretrained("Intel/dynamic_tinybert")

# %%
# Define a question-answering pipeline using the model and tokenizer
question_answerer = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
)

# %%
question = "Which is the most corrupt ministry?"

def err_remove(er):
    lin = "------------"
    er = str(er)
    start_index = er.find(lin) + len(lin)
    end_index = er.rfind(lin)
    answer = er[start_index:end_index].strip()
    return answer

try:
    # Use the pipeline directly for question answering
    context = " ".join([doc.page_content for doc in retrieved_docs])
    result = question_answerer(question=question, context=context)
    answer = result['answer']
    print("Answer:", answer)  # Print the answer if successful
except Exception as error:
    answer = err_remove(error)
    print("Error Answer:", answer)  # Print the error answer if an exception occurs
