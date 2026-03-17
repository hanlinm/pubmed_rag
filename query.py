from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def load_retriever(persist_dir: str = "./chroma_db", k: int = 3):
    """Load the persisted vector store and return a retriever."""
    embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')
    vectorstore = Chroma(
        persist_directory = persist_dir,
        embedding_function = embeddings,
    )
    # k=3 means it will fetch the 3 most relevant chunks for each question
    return vectorstore.as_retriever(search_kwargs={"k": k})

def build_rag_chain(retriever):
    """Assemble the full RAG chain."""

    prompt = ChatPromptTemplate.from_template("""

                                              You are a biomedical research assistant with expertise in computational biology and drug discovery. Answer the question using ONLY the provided research abstracts. If the answer is not in the abstracts, say so clearly - do not hallucinate. Always cite the paper title and PMID when referencing a finding.

                                              Research abstracts:
                                              {context}

                                              Question: {question}

                                              Answer (with citations):
                                              """)
    
    # Using gpt-4o-mini during development to keep costs low
    llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0)

    def format_docs(docs):
        """Format retrieved chunks into a single context string."""
        formatted = []
        for i, doc in enumerate(docs):
            formatted.append(
                f"[{i+1}] PMID: {doc.metadata.get('pmid', 'N/A')}\n"
                f"Title: {doc.metadata.get('title','N/A')}\n"
                f"Content: {doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted)
    
    # The full pipeline: retrieve -> format -> prompt -> LLM -> parse
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


if __name__ == "__main__":
    retriever = load_retriever()
    chain = build_rag_chain(retriever)

    # Test with a question relevant to your indexed papers
    question = "What deep learning methods have been used for protein structure prediction?"

    print(f"Question: {question}\n")
    print("Answer:")
    print(chain.invoke(question))