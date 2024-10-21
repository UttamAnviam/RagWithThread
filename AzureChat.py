import os
import PyPDF2
import openai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import pandas as pd
import csv
from uuid import UUID, uuid4
import requests
import os
from dotenv import load_dotenv
# Function to query OpenAI API with a single chunk
load_dotenv()


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

# Data structure to hold user threads
user_threads: Dict[str, List[Dict]] = {}

# Define a directory to save uploaded files
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Create the directory if it doesn't exist

class Message(BaseModel):
    user_id: str
    content: str

class Thread(BaseModel):
    id: UUID  # UUID will be provided in the request
    doctor_name: str
    user_id: str
    content: str
    messages: List[Message] = []  # Add messages to the thread
    uploaded_files: List[str] = []  # Track uploaded file paths

# Utility function to extract text from PDF, TXT, CSV, and Excel
def extract_text(file: UploadFile):
    file_type = file.filename.lower()

    try:
        if file_type.endswith(".pdf"):
            return extract_text_from_pdf(file.file)
        elif file_type.endswith(".txt"):
            return extract_text_from_txt(file.file)
        elif file_type.endswith(".csv"):
            return extract_text_from_csv(file.file)
        elif file_type.endswith((".xls", ".xlsx")):
            return extract_text_from_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the file: {file.filename}. Details: {e}")

def extract_text_from_pdf(file):
    pdf_text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading the PDF file: {e}")
    return pdf_text

def extract_text_from_txt(file):
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        print(f"Error reading the TXT file: {e}")
        return ""

def extract_text_from_csv(file):
    csv_text = ""
    try:
        content = file.read().decode("utf-8").splitlines()
        reader = csv.reader(content)
        for row in reader:
            csv_text += " ".join(row) + "\n"
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
    return csv_text

def extract_text_from_excel(file):
    try:
        excel_data = pd.read_excel(file)
        return excel_data.to_string(index=False)
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return ""

# Function to split text into chunks to handle token limits
def split_text_into_chunks(text, chunk_size=1500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


# Get the endpoint and API key from environment variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")


# Function to query Azure OpenAI API with a single chunk
def query_pdf_content(chunk_text, query):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    
    data = {
        "messages": [
            {
                "role": "user",
                "content": f"Analyze the following document: {chunk_text}. Based on this text, answer the question: {query}."
            }
        ]
    }
    
    try:
        response = requests.post(AZURE_OPENAI_ENDPOINT, json=data, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error querying Azure OpenAI API: {e}")
        return f"Error querying Azure OpenAI API: {e}"
    

# Function to query OpenAI API with multiple chunks and get a combined response
def query_pdf_content_in_chunks(combined_text, query):
    chunks = split_text_into_chunks(combined_text)
    responses = []

    for chunk in chunks:
        response = query_pdf_content(chunk, query)
        responses.append(response)

    return "\n".join(responses)

# API to create a new thread
@app.post("/threads/", response_model=Thread)
async def create_thread(thread: Thread):
    if thread.user_id not in user_threads:
        user_threads[thread.user_id] = []

    if any(existing_thread['id'] == thread.id for existing_thread in user_threads[thread.user_id]):
        raise HTTPException(status_code=400, detail="Thread with this ID already exists for this user.")

    user_threads[thread.user_id].append(thread.dict())
    return thread

# API to read all threads
@app.get("/threads/", response_model=Dict[str, List[Thread]])
def read_threads():
    return {user_id: [Thread(**thread) for thread in threads] for user_id, threads in user_threads.items()}

# API to read threads by user ID
@app.get("/threads/{user_id}", response_model=List[Thread])
def read_user_threads(user_id: str):
    if user_id in user_threads:
        return [Thread(**thread) for thread in user_threads[user_id]]
    raise HTTPException(status_code=404, detail="User threads not found")

# API to read a specific thread
@app.get("/threads/{user_id}/{thread_id}", response_model=Thread)
def read_thread(user_id: str, thread_id: UUID):
    if user_id not in user_threads:
        raise HTTPException(status_code=404, detail="User threads not found")

    for thread in user_threads[user_id]:
        if thread['id'] == thread_id:
            return Thread(**thread)
    raise HTTPException(status_code=404, detail="Thread not found")

# API to update a thread
@app.put("/threads/{user_id}/{thread_id}", response_model=Thread)
def update_thread(user_id: str, thread_id: UUID, updated_thread: Thread):
    if user_id not in user_threads:
        raise HTTPException(status_code=404, detail="User threads not found")

    for index, thread in enumerate(user_threads[user_id]):
        if thread['id'] == thread_id:
            user_threads[user_id][index] = updated_thread.dict()
            return updated_thread
    raise HTTPException(status_code=404, detail="Thread not found")

# API to delete a thread
@app.delete("/threads/{user_id}/{thread_id}", response_model=Thread)
def delete_thread(user_id: str, thread_id: UUID):
    if user_id not in user_threads:
        raise HTTPException(status_code=404, detail="User threads not found")

    for index, thread in enumerate(user_threads[user_id]):
        if thread['id'] == thread_id:
            return Thread(**user_threads[user_id].pop(index))
    raise HTTPException(status_code=404, detail="Thread not found")

# API to upload files and ask a query
@app.post("/upload_and_query/")
async def upload_and_query(
    files: List[UploadFile] = File(...), 
    query: str = Form(...),
    user_id: str = Form(...)
):
    combined_text = ""
    uploaded_file_names = [] 

    for file in files:
        # Save the file to the specified directory
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())  # Save the file content

        uploaded_file_names.append(file_location)  # Store the saved file path

        # Extract text from the uploaded file
        extracted_text = extract_text(file)
        combined_text += extracted_text + "\n"

    # Create a new thread with uploaded files
    thread_id = uuid4()  # Generate a new UUID for the thread
    new_thread = Thread(id=thread_id, doctor_name="DocName", user_id=user_id, content=combined_text, uploaded_files=uploaded_file_names)
    
    # Create the thread
    await create_thread(new_thread)

    # Query the content
    answer = query_pdf_content_in_chunks(combined_text, query)
    
    return {"query": query, "answer": answer}
# API to upload files and continue chat on an existing thread
@app.post("/upload_and_continue_chat/")
async def upload_and_continue_chat(
    thread_id: UUID = Form(...),
    files: List[UploadFile] = File(...),
    query: str = Form(...),
    user_id: str = Form(...)
):
    combined_text = ""
    uploaded_file_paths = []  # To store file paths

    # Extract text from the uploaded files and save them to a directory
    for file in files:
        # Save the file to the specified directory
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())  # Save the file content

        uploaded_file_paths.append(file_location)  # Store the saved file path

        # Extract text from the uploaded file
        extracted_text = extract_text(file)
        combined_text += extracted_text + "\n"

    if not combined_text:
        return JSONResponse(content={"error": "None of the provided files contain extractable text."}, status_code=400)

    # Fetch the thread and append the new message
    if user_id in user_threads:
        for thread in user_threads[user_id]:
            if thread['id'] == thread_id:
                # Append the query and file paths as a message from the user
                thread['messages'].append({
                    "user_id": user_id,
                    "content": f"Query: {query}\nFiles: {uploaded_file_paths}"
                })
                break
        else:
            raise HTTPException(status_code=404, detail="Thread not found.")
    else:
        raise HTTPException(status_code=404, detail="User threads not found.")

    # Query the content
    answer = query_pdf_content_in_chunks(combined_text, query)

    # Append assistant's response
    thread['messages'].append({
        "user_id": "assistant", 
        "content": answer
    })
    
    # Return query, answer, uploaded files, thread_id, and user_id
    return {
        "query": query,
        "answer": answer,
        "uploaded_files": uploaded_file_paths,
        "thread_id": str(thread_id),  # Add thread_id to the response
        "user_id": user_id  # Add user_id to the response
    }

