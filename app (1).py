import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langsmith import traceable

# ---- Load API keys FIRST (before any LangChain/LangSmith usage) ----
os.environ["GROQ_API_KEY"]         = st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_API_KEY"]    = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]    = "zyro-rag-challenge"

# ---- Page config ----
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="📋",
    layout="centered"
)

# ---- Build RAG pipeline (cached — runs only once per Streamlit session) ----
@st.cache_resource(show_spinner="Loading HR knowledge base, please wait...")
def build_pipeline():
    loader    = PyPDFDirectoryLoader("data/")
    documents = loader.load()
    splitter  = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks    = splitter.split_documents(documents)
    embeddings  = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=512)
    return retriever, llm

retriever, llm = build_pipeline()

RAG_SYSTEM = (
    "You are an expert HR assistant for Zyro Dynamics Pvt. Ltd.\n"
    "Answer employee questions based ONLY on the HR policy documents provided below.\n\n"
    "Rules:\n"
    "- Use ONLY the information in the context. Do not guess or use outside knowledge.\n"
    "- Be clear, concise, and helpful.\n"
    "- If the answer is not in the context, say: "
    "I do not have information about that in the current HR policy documents.\n"
    "- Use bullet points when listing multiple items.\n\n"
    "Context:\n{context}"
)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM),
    ("human",  "{question}")
])

OOS_SYSTEM = (
    "You are a topic classifier for an HR chatbot at Zyro Dynamics Pvt. Ltd.\n"
    "Reply with ONLY one word, no punctuation, no explanation:\n"
    "- HR  (if about company/HR topics: policies, leave, salary, benefits, conduct)\n"
    "- OUT (if completely unrelated: cooking, sports, weather, trivia, movies, etc.)"
)

OOS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", OOS_SYSTEM),
    ("human",  "{question}")
])

REFUSAL = (
    "I'm sorry, I can only assist with questions related to Zyro Dynamics HR policies, "
    "benefits, leave, workplace conduct, and other internal company matters. "
    "Please reach out to hr.helpdesk@zyrodynaamics.com for further assistance."
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@traceable(name="ask_bot", run_type="chain")
def ask_bot(question: str) -> str:
    classification = llm.invoke(
        OOS_PROMPT.format_messages(question=question)
    ).content.strip().upper()
    if "OUT" in classification:
        return REFUSAL
    docs    = retriever.invoke(question)
    context = format_docs(docs)
    msgs    = RAG_PROMPT.format_messages(context=context, question=question)
    return llm.invoke(msgs).content

# ---- UI ----
st.title("📋 Zyro Dynamics HR Help Desk")
st.caption("Your AI-powered assistant for HR policy queries.")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask your HR question here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            response = ask_bot(user_input)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
