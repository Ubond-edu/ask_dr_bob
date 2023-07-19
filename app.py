import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

index_name = "ask dr bob"

# Initialize Pinecone with your existing index name
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Create OpenAIEmbeddings instance
embeddings = OpenAIEmbeddings()

# Initialize Pinecone vector store with the existing index
doc_db = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.2
)


def retrieval_answer_with_sources(query):
    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=doc_db.as_retriever(),
    )
    result = qa_with_sources.run(query)
    answer = result['answer']
    sources = result['sources']
    return answer, sources


def main():
    st.title("Ask Dr. Bob")

    query = st.text_input("Ask your query...")
    if st.button("Ask Query"):
        if len(text_input) > 0:
            st.info("Your Query: " + text_input)
            answer, sources = retrieval_answer_with_sources(text_input)
            if answer:
                st.success("Answer: " + answer)
            else:
                st.warning("No answer found.")
            if sources:
                st.info("Sources: " + ", ".join(sources))
            else:
                st.info("No sources found.")


if __name__ == "__main__":
    main()
