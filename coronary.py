import PyPDF2
import openai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import pandas as pd
import csv
from uuid import UUID

app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set your OpenAI API key
openai.api_key = "APi Key"  # Replace with your actual OpenAI API key

# Data structure to hold threads
threads: List["Thread"] = []
user_threads: Dict[str, List[Dict]] = {}

class Thread(BaseModel):
    id: UUID  # UUID will be provided in the request
    doctor_name: str
    user_id: str
    content: str

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    pdf_text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading the PDF file: {e}")
    return pdf_text

# Functions for other file types (TXT, CSV, Excel) remain the same...

# Function to split text into smaller chunks to fit token limits
def split_text_into_chunks(text, chunk_size=1500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to query OpenAI API with a single prompt
def query_pdf_content(chunk_text, query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"Analyze the following document: {chunk_text}. Based on this text, answer the question: {query}."
                }
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error querying OpenAI API: {e}")
        return f"Error querying OpenAI API: {e}"

# Function to handle PDF content queries
def query_pdf_content_in_chunks(combined_text, query):
    chunks = split_text_into_chunks(combined_text)
    responses = []

    for chunk in chunks:
        response = query_pdf_content(chunk, query)
        responses.append(response)

    combined_response = "\n".join(responses)

    final_response = query_pdf_content(combined_response, """generate a final report...""")  # Full instructions for coroner's report

    return final_response

@app.post("/upload_and_query/")
async def upload_and_query(
    files: List[UploadFile] = File(...), 
    query: str = Form(...),
    user_id: str = Form(...)
):
    # Check for specific queries that should not trigger PDF processing
    if "patient id" in query.lower() or "top" in query.lower():
        # Handle database queries here
        # Example response for patient ID query (this should connect to your actual database logic)
        # This is a placeholder for your database logic
        return JSONResponse(content={"query": query, "result": "Placeholder result based on database logic"}, status_code=200)

    combined_text = ""

    for file in files:
        filename = file.filename.lower()

        # Handle PDF files
        if filename.endswith(".pdf"):
            pdf_text = extract_text_from_pdf(file.file)
            combined_text += pdf_text + "\n"
        
        # Handle other file types (TXT, CSV, Excel)...
        
        else:
            return JSONResponse(content={"error": f"Unsupported file type: {file.filename}"}, status_code=400)

    if not combined_text:
        return JSONResponse(content={"error": "None of the provided files contain extractable text."}, status_code=400)

    answer = query_pdf_content_in_chunks(combined_text, query)
    
    return {"query": query, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
