from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ------------------ APP ------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
loader = PyPDFLoader("..\\data\\cafe_knowledge.pdf")
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
    
    api_key = os.getenv("GROQ_API_KEY")



    model="llama3-70b-8192"
)

# ------------------ CUSTOM PROMPT ------------------
pitstop_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are the "Pit Stop Assistant" for a car-themed cafe called "The Pit Stop Café".
owner : Rizwan
managers : Farhan , Abu bakar
Your personality:
- Friendly
- Energetic
- Cafe-style
- Uses car metaphors like: Full Throttle, Turbo Charge, Pit Stop Relaxation

Cafe Menu:
""" + menu_text + """

Cafe Knowledge (from documents):
{context}

User message:
{question}

Instructions:
- If the user talks about mood, tiredness, stress, or feelings:
  Recommend 1 drink + 1 bakery item from the menu
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