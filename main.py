import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ------------------ LOAD ENV ------------------
load_dotenv()

# ------------------ APP ------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ PATHS ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend_dist")  # <-- Vite build folder
DATA_DIR = os.path.join(BASE_DIR, "data")

# ------------------ SERVE FRONTEND ------------------
# Serve all assets (JS/CSS/images) - include hashed names
app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="assets")

# Serve index.html for all SPA routes
@app.get("/{full_path:path}")
def serve_frontend(full_path: str):
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# ------------------ MENU ------------------
MENU_ITEMS = [
    {"name": "Turbo Latte", "category": "Drink", "notes": "Strong & energetic"},
    {"name": "Full Throttle Espresso", "category": "Drink", "notes": "High caffeine"},
    {"name": "Pit Stop Cappuccino", "category": "Drink", "notes": "Smooth & balanced"},
    {"name": "Garage Brownie", "category": "Bakery", "notes": "Rich chocolate"},
    {"name": "Trackside Croissant", "category": "Bakery", "notes": "Light & buttery"},
]

menu_text = ", ".join(
    [f"{i['name']} ({i['category']} – {i['notes']})" for i in MENU_ITEMS]
)

# ------------------ PDF LOADING ------------------
pdf_path = os.path.join(DATA_DIR, "cafe_knowledge.pdf")
loader = PyPDFLoader(pdf_path)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)

# ------------------ LLM ------------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ------------------ CUSTOM PROMPT ------------------
pitstop_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"""
You are the "Pit Stop Assistant" for a car-themed cafe called "The Pit Stop Café".
Owner: Rizwan

Your personality:
- Friendly
- Energetic
- Cafe-style
- Uses car metaphors like: Full Throttle, Turbo Charge, Pit Stop Relaxation

Cafe Menu:
{menu_text}

Cafe Knowledge:
{{context}}

User message:
{{question}}

Instructions:
- If the user talks about mood, tiredness, stress, or feelings:
  Recommend 1 drink + 1 bakery item
- Use car metaphors naturally
- Keep the answer short, fun, and helpful
"""
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": pitstop_prompt},
)

# ------------------ API ------------------
class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Query):
    answer = qa.run(q.question)
    return {"answer": answer}

# ------------------ RUN SERVER ------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
