from src.helper import load_pdf,clean_text,remove_metadata_from_documents,text_split
import os
import re
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import Pinecone
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')


from pinecone import Pinecone, ServerlessSpec

# Create the Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# Optional: Check if your index exists
if "medical-chatbot" not in pc.list_indexes().names():
    pc.create_index(
        name="medical-chatbot",
        dimension=384,  # Use 384 if you're using all-MiniLM-L6-v2 embeddings
        metric="cosine",
        spec=ServerlessSpec(
             cloud="aws",
    region="us-east-1" 
        )
    )

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name="medical-chatbot"
)