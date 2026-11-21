# app.py - Chat with your PDFs (a small RAG chatbot)
#
# RAG = Retrieval-Augmented Generation. Instead of asking Gemini a question
# directly, we first find the parts of YOUR PDFs that are relevant to the
# question, paste those parts into the prompt, and then ask Gemini to answer
# using only that context. That way the chatbot can answer questions about
# files it has never seen during training.
#
# The pipeline in one line:
#   PDFs -> text -> chunks -> embedding vectors -> FAISS index
#   question -> embedding -> FAISS finds closest chunks -> Gemini answers from them

# ---------- 1. Imports ----------
import os                           # reads environment variables (where the API key lives)
import numpy as np                  # FAISS needs vectors as NumPy arrays (numpy comes with faiss-cpu)
import streamlit as st              # the web app framework: every st.* call draws something on the page
import faiss                        # vector search library: stores embeddings and finds the similar ones
from dotenv import load_dotenv      # loads the .env file so GOOGLE_API_KEY becomes an environment variable
from pypdf import PdfReader         # reads text out of PDF files

from langchain_text_splitters import RecursiveCharacterTextSplitter  # splits long text into chunks
# ^ if this import ever fails, run: pip install langchain-text-splitters
from langchain_google_genai import GoogleGenerativeAIEmbeddings       # turns text into embedding vectors
from langchain_google_genai import ChatGoogleGenerativeAI             # the Gemini chat model

# ---------- 2. Load the API key ----------
load_dotenv()  # reads the file named .env sitting next to app.py; after this, os.environ has GOOGLE_API_KEY

if not os.environ.get("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY is missing. Create a file named .env next to app.py "
             "with one line: GOOGLE_API_KEY=your-key-here")
    st.stop()  # stop the app here; nothing below can work without a key

# ---------- 3. Set up the two Gemini models ----------
# The embedding model turns a piece of text into a list of numbers (a "vector").
# Texts with similar meaning get similar numbers, which is what lets us search
# by meaning instead of by exact words.
# gemini-embedding-2 is Google's current embedding model and is on the free tier
# (ai.google.dev/gemini-api/docs/pricing, checked Nov 2026).
# If you ever get "model not found" on this line, use "gemini-embedding-001" instead.
embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

# The chat model is the LLM that writes the final answer. gemini-3.5-flash is
# the fast general model with a free tier (same pricing page, checked Nov 2026).
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)
# temperature=0.2 keeps answers factual instead of creative: what you want for Q&A over documents.

# ---------- 4. Helper: read the text out of the uploaded PDFs ----------
def read_pdfs(uploaded_files):
    """Return one big string with the text of every page of every uploaded PDF."""
    all_text = ""
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)  # Streamlit hands us a file-like object; pypdf reads it directly
        for page in reader.pages:
            page_text = page.extract_text() or ""  # image-only pages return None -> treat them as empty
            all_text += page_text + "\n"
    return all_text

# ---------- 5. Helper: split the text into chunks ----------
def split_into_chunks(text):
    """Split the text into chunks of about 1000 characters, with 200 characters of overlap.

    Why chunk at all? Two reasons:
    1. Embedding models have an input limit, so we cannot embed a whole document at once.
    2. Small chunks give precise search: we retrieve the one paragraph that matters,
       not a 50-page document.
    The overlap means a sentence that falls on a chunk boundary still appears whole
    in at least one chunk, so no idea gets cut in half.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # maximum characters per chunk
        chunk_overlap=200,  # how many characters repeat between neighbouring chunks
    )
    return splitter.split_text(text)

# ---------- 6. Helper: embed the chunks and build the FAISS index ----------
def build_vector_index(chunks):
    """Embed every chunk and store the vectors in a FAISS index.

    FAISS works like a small database for vectors: you hand it a query vector
    and it returns the stored vectors closest to it. "Closest" here means the
    smallest L2 (straight-line) distance, which for embeddings means "most
    similar in meaning".
    """
    # One API call embeds all the chunks; the result is a list of vectors (lists of floats).
    vectors = embeddings_model.embed_documents(chunks)

    vectors_np = np.array(vectors, dtype="float32")  # FAISS only accepts float32 NumPy arrays
    dimension = vectors_np.shape[1]                  # the length of one vector (e.g. 768)

    index = faiss.IndexFlatL2(dimension)  # the simplest FAISS index: compares the query against every stored vector
    index.add(vectors_np)                 # store all the chunk vectors in the index
    return index

# ---------- 7. Helper: turn Gemini's reply into plain text ----------
def extract_text(response_content):
    """Return the reply as one plain string.

    Newer versions of the Gemini API return the reply as a LIST of content
    blocks, for example [{"type": "text", "text": "..."}], instead of one
    plain string. Some blocks are not text at all (they carry metadata like a
    "signature"), so we keep only the "text" blocks and join them together.
    Without this step, st.write would show the raw block structure instead of
    a clean answer. If the reply is already a string (older behaviour), we
    return it unchanged.
    """
    if isinstance(response_content, str):
        return response_content
    parts = []
    for block in response_content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):  # some versions mix plain strings into the list
            parts.append(block)
    return "".join(parts)

# ---------- 8. Helper: answer one question with RAG ----------
def answer_question(question, index, chunks, k=4):
    """Retrieve the k most relevant chunks and ask Gemini to answer from them."""
    # Step A: embed the question the same way we embedded the chunks.
    query_vector = np.array([embeddings_model.embed_query(question)], dtype="float32")

    # Step B: ask FAISS for the positions of the k closest chunk vectors.
    distances, positions = index.search(query_vector, k)

    # Step C: collect those chunks and join them into one context block.
    relevant_chunks = [chunks[i] for i in positions[0]]
    context = "\n\n---\n\n".join(relevant_chunks)

    # Step D: build the prompt: instructions + retrieved context + the question.
    # Telling the model to use ONLY the context is what keeps it grounded in
    # your documents instead of answering from its general knowledge.
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the
context below, which comes from the user's PDF documents. If the answer is not
in the context, say that you could not find it in the documents.

Context:
{context}

Question: {question}

Answer:"""

    # Step E: call Gemini, then flatten the reply into a plain string with the
    # helper above before we store or show it.
    response = llm.invoke(prompt)
    return extract_text(response.content)

# ---------- 9. The Streamlit page ----------
st.set_page_config(page_title="Chat with your PDFs", page_icon=":page_facing_up:")
st.title("Chat with your PDFs")
st.write("Upload one or more PDFs, then ask questions about them. "
         "Every answer comes from your documents only.")

# st.session_state is Streamlit's memory between reruns. Every click reruns the
# whole script from top to bottom, so anything we want to keep (the index, the
# chat history) must live in session_state, not in normal variables.
if "index" not in st.session_state:
    st.session_state.index = None   # the FAISS index, once built
    st.session_state.chunks = []    # the text chunks, in the same order as the index
    st.session_state.messages = []  # the chat history shown on the page

# ---------- Sidebar: upload and process the PDFs ----------
with st.sidebar:
    st.header("Your documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )
    process_clicked = st.button("Process PDFs")

if process_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one PDF first.")
    else:
        with st.spinner("Reading and indexing your PDFs..."):
            text = read_pdfs(uploaded_files)
            chunks = split_into_chunks(text)
            index = build_vector_index(chunks)
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.messages = []  # new documents -> start a fresh conversation
        st.success(f"Done! Indexed {len(chunks)} chunks from {len(uploaded_files)} PDF(s). "
                   "Ask your first question below.")

# ---------- Show the conversation so far ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---------- The question box at the bottom of the page ----------
question = st.chat_input("Ask a question about your PDFs...")

if question:
    if st.session_state.index is None:
        st.warning("Please upload and process your PDFs first (see the sidebar).")
    else:
        # show and store the user's message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        # get the answer, then show and store it
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_question(question, st.session_state.index, st.session_state.chunks)
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
