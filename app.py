import os
from langchain.chains import RetrievalQAWithSourcesChain  
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
import pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Create OpenAIEmbeddings instance
embeddings = OpenAIEmbeddings()

# Initialize Pinecone with your existing index name
# Pass empty list as documents since the index already exists
doc_db = Pinecone.from_existing_index('ask-dr-bob', embeddings)

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',  # Adjust model_name as necessary
    temperature=0.2  # Adjust temperature as necessary
)

def retrieval_answer_with_sources(query):
    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, 
        chain_type='stuff',  # You might need to replace 'stuff' with the correct chain type
        retriever=doc_db.as_retriever(),
    )
    result = qa_with_sources.run(query)
    return result['answer'], result['sources']  # Return both 'answer' and 'sources'

def main():
    st.title("Ask Dr. Bob")

    text_input = st.text_input("Ask your query...") 
    if st.button("Ask Query"):
        if len(text_input)>0:
            st.info("Your Query: " + text_input)
            answer, sources = retrieval_answer_with_sources(text_input)  # Receive both 'answer' and 'sources'
            st.success("Answer: " + answer)
            st.info("Sources: " + ", ".join(sources))

if __name__ == "__main__":
    main()
