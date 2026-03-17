import streamlit as st
from query import load_retriever, build_rag_chain

st.set_page_config(page_title='PubMed RAG', page_icon = "🧬", layout = 'wide')
st.title("🧬 Biomedical Literature Assistant")
st.caption("Ask questions grounded in PubMed research abstracts")

with st.sidebar:
    st.header("📚 Covered Topics")
    st.markdown("""
    This assistant has knowledge of the following research areas:
                - Protein structure prediction
                - Generative AI for drug discovery
                - GLP-1 receptor agonists
                - Biomedical NLP & transformers
                - Cancer immunotherapy
                - CRISPR & gene editing
                - Single cell RNA sequencing
                - Clinical trial outcome prediction
                - Antibiotic resistance prediction
                - Protein language models
    """)
    st.divider()
    st.caption("Powered by PubMed, LangChain, and OpenAI")

# Example questions on main page
st.markdown("**Try asking:**")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    - *How is deep learning used for protein structure prediction?*
    - *What are GLP-1 receptor agonists and how do they work?*
    - *How is CRISPR being used with machien learning?*
    - *What ML approaches predict clinical trial outcomes?*
    """)
with col2:
    st.markdown("""
    - *How is generative AI being applied to drug discovery?*
    - *What transformer models are used in biomedical NLP?*
    - *How is AI used in cancer immunotherapy research?*
    - *What are protein language models?*
    """)
st.divider()

# Cache the chain so it doesn't reload on every interaction
@st.cache_resource
def get_chain():
    retriever = load_retriever()
    chain = build_rag_chain(retriever)
    return chain, retriever

chain, retriever = get_chain()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if question := st.chat_input("Ask about the research literature..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Show retrieved sources in an expander above the answer
        with st.spinner("Searching literature..."):
            source_docs = retriever.invoke(question)
        
        with st.expander(f"📄 Retrieved {len(source_docs)} sources"):
            for doc in source_docs:
                st.markdown(f"**{doc.metadata.get('title', 'N/A')}**")
                pmid = doc.metadata.get('pmid', '')
                if pmid:
                    st.markdown(f"[PMID {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
                st.caption(doc.page_content[:300] + "...")
                st.divider()

        response = st.write_stream(chain.stream(question))

    st.session_state.messages.append({"role": "assistant", "content": response})