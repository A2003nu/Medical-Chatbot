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


def load_pdf(data):
    all_docs = []
    for filename in os.listdir(data):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data, filename))
            pages = loader.load()
            all_docs.extend(pages[14:])  # Skip the first 14 pages (adjust this number if needed)
    return all_docs

def clean_text(text):
    text = re.sub(r"^GALE ENCYCLOPEDIA OF MEDICINE.*", "", text)  # Remove lines starting with GALE ENCYCLOPEDIA OF MEDICINE
    text = re.sub(r"^GEM - .+", "", text)  # Remove GEM metadata entirely
    text = re.sub(r"^Page \d{1,3}.*", "", text)  # Remove page number information
    text = re.sub(r"\d{1,3}.*", "", text)  # Remove any numeric references (like page numbers or codes)
    text = re.sub(r"Photograph.*", "", text)  # Remove photograph references
    text = re.sub(r"Reproduced by permission.*", "", text)  # Remove permission notice
    text = re.sub(r"\s{2,}", " ", text)  # Replace multiple spaces with a single space
    return text.strip()  # Strip trailing/leading spaces


def remove_metadata_from_documents(documents):
    cleaned_documents = []
    for doc in documents:
        cleaned_content = clean_text(doc.page_content)
        cleaned_documents.append(Document(page_content=cleaned_content))
    return cleaned_documents

def text_split(extracted_data):
    cleaned_docs = remove_metadata_from_documents(extracted_data)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(cleaned_docs)
    return text_chunks

