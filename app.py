from flask import Flask, render_template, jsonify, request
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
from src.prompt import *
from langchain.chains import create_retrieval_chain
from src.helper import load_pdf,clean_text,remove_metadata_from_documents,text_split
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnableLambda
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name="medical-chatbot"
)


retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",  # Use the appropriate model version
    temperature=0.4,
    max_output_tokens=500,
    google_api_key=GOOGLE_API_KEY,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

input_mapper = RunnableParallel({
    "context": retriever,  # Ensure retriever is a valid retriever object
    "input": RunnableLambda(lambda x: x)  # Pass the question as 'input'
})

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = RunnableSequence(input_mapper, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    user_input = msg
    print("User input:", user_input)
    
    result = rag_chain.invoke(user_input)
    
    print("Response:", result)
    
    # result could be a string or an object depending on your LLM output
    return str(result.content if hasattr(result, "content") else result)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)