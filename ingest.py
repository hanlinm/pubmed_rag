from Bio import Entrez
from dotenv import load_dotenv
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
Entrez.email = os.getenv("NCBI_EMAIL")

def fetch_pubmed_abstracts(query: str, max_results: int = 10) -> list[Document]:

    # Step 1: search for paper IDs matching your query
    handle = Entrez.esearch(db="pubmed", term = query, retmex=max_results)
    record = Entrez.read(handle)
    ids = record["IdList"]
    print(f"Found {len(ids)} papers for: '{query}'")

    # Step 2: fetch the actual abstracts using those IDs
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
    records = Entrez.read(handle)

    documents = []
    for paper in records["PubmedArticle"]:
         try:
             article = paper["MedlineCitation"]["Article"]
             title = str(article["ArticleTitle"])
             abstract = str(article.get("Abstract", {}).get("AbstractText",[""])[0])
             pmid = str(paper["MedlineCitation"]["PMID"])

             if abstract:
                  documents.append(Document(
                      page_content = f"Title: {title}\n\nAbstract: {abstract}",
                      metadata={
                          "pmid": pmid,
                          "title": title,
                          "source": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                       }
                   ))
         except (KeyError, IndexError):
             continue
    
    print(f"Successfully parsed {len(documents)} abstracts")
    return documents

def build_vector_store(documents: list[Document], persist_dir: str = "./chroma_db"):

    # Step 1: split documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")

    # Step 2: embed chunks and store in ChromaDB
    print("Embedding chunks... (this calls the OpenAI API, may take a moment)")
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    print(f"Vector store built and saved to {persist_dir}")
    return vectorstore

# Test it by running this file directly
if __name__ == "__main__":
    topics = [
        "protein structure prediction deep learning",
        "generative AI drug discovery small molecule",
        "GLP-1 receptor agonist obesity diabetes",
        "transformer models biomedical NLP",
        "cancer immunotherapy machine learning",
        "CRISPR gene editing machine learning",
        "single cell RNA sequencing deep learning",
        "clinical trial outcome prediction AI",
        "antibiotic resistance prediction neural network",
        "protein language models therapeutics"
    ]

    all_docs = []

    for topic in topics:
        all_docs.extend(fetch_pubmed_abstracts(topic, max_results = 10))

    print(f"\nTotal documents fetched: {len(all_docs)}")

    build_vector_store(all_docs)