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
openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your actual OpenAI API key

# Data structure to hold threads
threads: List["Thread"] = []

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

# Function to extract text from a TXT file
def extract_text_from_txt(file):
    try:
        content = file.read().decode("utf-8")
        return content
    except Exception as e:
        print(f"Error reading the TXT file: {e}")
        return ""

# Function to extract text from a CSV file
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

# Function to extract text from an Excel file
def extract_text_from_excel(file):
    try:
        excel_data = pd.read_excel(file)
        return excel_data.to_string(index=False)
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return ""

# Function to split text into smaller chunks to fit token limits
def split_text_into_chunks(text, chunk_size=1500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to query OpenAI API with a single prompt
def query_pdf_content(chunk_text, query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Ensure the correct model is used
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

# Function to query OpenAI API with each chunk and get a combined response
def query_pdf_content_in_chunks(combined_text, query):
    chunks = split_text_into_chunks(combined_text)
    responses = []

    for chunk in chunks:
        response = query_pdf_content(chunk, query)
        responses.append(response)

    # Combine responses for final output
    combined_response = "\n".join(responses)

    # Final query to OpenAI API to summarize combined responses
    final_response = query_pdf_content(combined_response, """generate a final report.if asked to generate a coronere report , generate it in a detailed coronere  format with proper explanation 
                                       General principles
The report should be a detailed factual account, based on the
medical records and your knowledge of the deceased.
• Include your full name and qualifications (Bachelor of Medicine
rather than MB).
• Describe your status at the time you saw the patient (eg, GP
registrar or consultant surgeon for 10 years).
• Type your report on headed paper where possible using full,
grammatically correct sentences.
• Divide your report up into clear paragraphs. Numbering paragraphs
may make it easier to refer to sections of your report in case you're
asked to give evidence.
What to include
Be specific about your contact with the patient. For example, did
you see the patient on the NHS or privately?
Where appropriate, say if you saw the patient alone or with
someone else during each consultation. Give the name and status
of the other person (eg, spouse, mother, social worker).
Style
If you're ever
asked to write a
coroner's report,
it's important to
know what to
include.
21 July 2022
The report should stand on its own
Don't assume the reader has any knowledge of the case. Several
people may have to read the report apart from the coroner and they
may not have access to or be able to interpret the medical records.
Write in the first person
The reader should have a good idea of who did what, why, when, to
whom, and how you know this occurred. Be precise and explicit.
• Example: instead of writing, 'The patient was examined again later
in the day' - it's more helpful to say, 'I remember asking my
registrar, Dr Jane Smith, to examine the patient again later on the
same day, which, according to the notes, she did at [time].'
Concentrate on your observations and understanding
Provide a detailed account of your interaction with the patient
including the history you were given, what examination took place
and what your clinical findings were. Include any relevant negative
findings. Give an account of your differential diagnosis,
management plan and any safety netting advice that was given.
Avoid jargon or medical abbreviations
Your report will be read by those with no medical knowledge. All
medical terms are best written in full, avoiding abbreviations and
technical language, if possible. If you have to use abbreviations or medical terms, explain these. If you mention a drug, give an idea of
what type of drug it is and why it was prescribed. Give the full
generic name, dosage and route of administration.
• Example: many lay people might be familiar with abbreviating blood
pressure to 'BP'. But 'SOB' for 'shortness of breath' is less common,
and could be misinterpreted.
Clinical notes
Give a factual chronology of events as you saw them, referring
to the clinical notes whenever you can. Describe each and every
relevant consultation or phone contact in turn and include your
working diagnosis or your differential diagnoses.
Outline any hospital referrals, identifying the name of the relevant
practitioner or consultant.
The coroner will often require disclosure of the whole medical
record. You should also ensure you have had access to the full
medical record when preparing your report.
The absence of an entry may be important. Just as negative
findings are often important in clinical reports, with a coroner's
report it's important to think about what's not included, as well as
what is.
• Example: you're reporting on a case of a child who has died. The
pathologist finds healed fractures at post-mortem, but the notes
don't indicate that the parents sought medical advice for these
injuries. This raises the question of non-accidental injury and could
have serious and immediate implications for surviving children in the family.
Say what you found, but also what you looked for and failed to
find. If you failed to put yourself in a position to make an adequate
assessment, your evidence could be challenged. If your report
clearly demonstrates that your history and examination were
thorough, you are less likely to be called to explain your evidence at
an inquest.
Specify what the different details of your account are based
on. This could be your memory, the contemporaneous notes you or
others wrote, or your usual or normal practice. A coroner won't
expect you to make notes of every last detail, or to remember every
aspect of a consultation that at the time appeared to be routine. It's
perfectly acceptable to quote from memory, making it clear that this
is what you're doing or explaining what your normal practice would be under those circumstances.
Identify any other clinician involved in the care of the
deceased by their full name and professional status. Describe your
understanding of what they did and the conclusions they reached
based on your own knowledge or the clinical notes. You should not,
however, comment on the adequacy or otherwise of their
performance.""")
    
    return final_response


user_threads: Dict[str, List[Dict]] = {}

class Thread(BaseModel):
    id: UUID  # UUID will be provided in the request
    doctor_name: str
    user_id: str
    content: str

@app.post("/threads/", response_model=Thread)
async def create_thread(thread: Thread):
    # Ensure the user has a list for their threads
    if thread.user_id not in user_threads:
        user_threads[thread.user_id] = []

    # Check for duplicate thread IDs for the same user
    if any(existing_thread['id'] == thread.id for existing_thread in user_threads[thread.user_id]):
        raise HTTPException(status_code=400, detail="Thread with this ID already exists for this user.")

    # Append the new thread to the user's thread list
    user_threads[thread.user_id].append(thread.dict())
    return thread

@app.get("/threads/", response_model=Dict[str, List[Thread]])
def read_threads():
    # Convert the stored threads to the response model
    return {user_id: [Thread(**thread) for thread in threads] for user_id, threads in user_threads.items()}

@app.get("/threads/{user_id}", response_model=List[Thread])
def read_user_threads(user_id: str):
    if user_id in user_threads:
        return [Thread(**thread) for thread in user_threads[user_id]]
    raise HTTPException(status_code=404, detail="User threads not found")

@app.get("/threads/{user_id}/{thread_id}", response_model=Thread)
def read_thread(user_id: str, thread_id: UUID):
    if user_id not in user_threads:
        raise HTTPException(status_code=404, detail="User threads not found")
        
    for thread in user_threads[user_id]:
        if thread['id'] == thread_id:
            return Thread(**thread)
    raise HTTPException(status_code=404, detail="Thread not found")

@app.put("/threads/{user_id}/{thread_id}", response_model=Thread)
def update_thread(user_id: str, thread_id: UUID, updated_thread: Thread):
    if user_id not in user_threads:
        raise HTTPException(status_code=404, detail="User threads not found")

    for index, thread in enumerate(user_threads[user_id]):
        if thread['id'] == thread_id:
            user_threads[user_id][index] = updated_thread.dict()
            user_threads[user_id][index]['id'] = thread_id  # Ensure ID remains the same
            return updated_thread
    raise HTTPException(status_code=404, detail="Thread not found")

@app.delete("/threads/{user_id}/{thread_id}", response_model=Thread)
def delete_thread(user_id: str, thread_id: UUID):
    if user_id not in user_threads:
        raise HTTPException(status_code=404, detail="User threads not found")

    for index, thread in enumerate(user_threads[user_id]):
        if thread['id'] == thread_id:
            return Thread(**user_threads[user_id].pop(index))
    raise HTTPException(status_code=404, detail="Thread not found")

# Endpoint to upload files and ask a question
@app.post("/upload_and_query/")
async def upload_and_query(
    files: List[UploadFile] = File(...), 
    query: str = Form(...),
    user_id: str = Form(...)
):
    combined_text = ""

    for file in files:
        filename = file.filename.lower()

        # Handle PDF files
        if filename.endswith(".pdf"):
            pdf_text = extract_text_from_pdf(file.file)
            combined_text += pdf_text + "\n"

        # Handle TXT files
        elif filename.endswith(".txt"):
            txt_text = extract_text_from_txt(file.file)
            combined_text += txt_text + "\n"

        # Handle CSV files
        elif filename.endswith(".csv"):
            csv_text = extract_text_from_csv(file.file)
            combined_text += csv_text + "\n"

        # Handle Excel files
        elif filename.endswith(".xls") or filename.endswith(".xlsx"):
            excel_text = extract_text_from_excel(file.file)
            combined_text += excel_text + "\n"

        else:
            return JSONResponse(content={"error": f"Unsupported file type: {file.filename}"}, status_code=400)

    if not combined_text:
        return JSONResponse(content={"error": "None of the provided files contain extractable text."}, status_code=400)

    answer = query_pdf_content_in_chunks(combined_text, query)
    
    return {"query": query, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
