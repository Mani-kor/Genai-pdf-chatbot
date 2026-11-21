# Chat with your PDFs (RAG Chatbot)

A web app that lets you upload PDF documents and chat with them. You ask questions in plain English and get answers grounded in the documents, using Retrieval-Augmented Generation (RAG).

This is my personal portfolio project, built to learn and demonstrate RAG, embeddings and vector search with the Google Gemini API.

## What it does

- Upload one or more PDF files in the browser
- Extracts the text and splits it into overlapping chunks
- Turns each chunk into an embedding vector with Google's embedding model
- Stores the vectors in a FAISS index for fast similarity search
- For each question, retrieves the most relevant chunks and asks Gemini to answer using only that context

## Features

- Multi-PDF upload
- Conversational chat interface built with Streamlit
- Answers grounded in your documents (the prompt tells the model to say when it cannot find the answer)
- Free to run on the Gemini API free tier

## Tech stack

- Python 3.11
- Streamlit: web interface
- LangChain (langchain-text-splitters, langchain-google-genai): chunking, embeddings and LLM calls
- FAISS (faiss-cpu): vector similarity search
- pypdf: PDF text extraction
- Google Gemini API: gemini-embedding-2 for embeddings, gemini-3.5-flash for answers

## How to run

1. Clone this repository and open the folder in VS Code.
2. Install the packages (a virtual environment is recommended):

   ```
   pip install streamlit langchain langchain-google-genai faiss-cpu pypdf python-dotenv
   ```

3. Get a free Gemini API key from Google AI Studio (aistudio.google.com).
4. Create a file named `.env` in the project folder with one line:

   ```
   GOOGLE_API_KEY=your-key-here
   ```

5. Start the app:

   ```
   streamlit run app.py
   ```

6. Your browser opens at http://localhost:8501. Upload your PDFs, click "Process PDFs", and start asking questions.
