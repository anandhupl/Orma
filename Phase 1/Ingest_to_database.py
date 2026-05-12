import os
import time
import re
import requests # <-- Ripping out the Google SDK for raw HTTP
from supabase import create_client, Client
from dotenv import load_dotenv

# --- INITIALIZATION ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Direct Google REST API endpoint for Batch Embeddings
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:batchEmbedContents?key={GEMINI_API_KEY}"

def smart_chunker(text, max_chars=1500, overlap=200):
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence + " "
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            words = current_chunk.split()
            overlap_text = " ".join(words[-int(overlap/5):]) if words else ""
            current_chunk = overlap_text + " " + sentence + " "
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks

def ingest_safely(subject_slug, language_slug, text_data):
    # 1. Clear existing data
    supabase.table("textbook_sections_v2").delete().eq("subject", subject_slug).eq("language", language_slug).execute()
    print(f"🧹 Cleared existing data for {subject_slug} ({language_slug}).")

    # 2. Page Tracking
    page_splits = re.split(r'(--- Page (?:\d+|Unknown) ---)', text_data)
    current_page = "Unknown"
    all_chunks = []

    for i in range(len(page_splits)):
        part = page_splits[i].strip()
        if not part: continue
        if re.match(r'--- Page (?:\d+|Unknown) ---', part):
            current_page = part.replace('-', '').strip()
            continue
        
        chunks = smart_chunker(part)
        for chunk_text in chunks:
            all_chunks.append({"content": chunk_text, "page": current_page})

    print(f"Generated {len(all_chunks)} chunks for {subject_slug}.")
    
    # 3. The REST API Bypass
    BATCH_SIZE = 20 # Safe size for 30k TPM limit
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        
        # Build the exact raw JSON payload Google expects
        api_requests = []
        for item in batch:
            api_requests.append({
                "model": "models/gemini-embedding-2-preview",
                "content": {
                    "parts": [{"text": item["content"]}]
                },
                "taskType": "RETRIEVAL_DOCUMENT",
                "outputDimensionality": 768
            })
            
        payload = {"requests": api_requests}
        
        try:
            # Send raw HTTP request
            response = requests.post(API_URL, json=payload)
            response_data = response.json()
            
            # Catch API-level errors
            if "embeddings" not in response_data:
                print(f"❌ Server Error on batch {i//BATCH_SIZE + 1}: {response_data}")
                time.sleep(30)
                continue
                
            embeddings = response_data["embeddings"]
            
            # Map the returned vectors to Supabase
            rows_to_insert = []
            for j, item in enumerate(batch):
                rows_to_insert.append({
                    "subject": subject_slug, 
                    "language": language_slug, 
                    "content": item["content"],
                    "metadata": {"page": item["page"]},
                    "embedding": embeddings[j]["values"] # Direct vector extraction
                })
            
            supabase.table("textbook_sections_v2").insert(rows_to_insert).execute()
            print(f"✅ Synced {min(i + BATCH_SIZE, len(all_chunks))} / {len(all_chunks)}")
            
            # Sleep 30s to respect the 30k TPM quota
            time.sleep(30) 
            
        except Exception as e:
            print(f"❌ HTTP/Database Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    subjects = [
        # ✅ ALREADY SYNCED: Commented out to save daily quota
        # {"slug": "Physics", "file": "Physics_eng_converted.txt", "lang": "english"}, 
        # {"slug": "Chemistry", "file": "Chemistry_eng_converted.txt", "lang": "english"}, 
        # {"slug": "Chemistry", "file": "Chemistry_mal_converted.txt", "lang": "malayalam"}, 
        # {"slug": "English", "file": "English_converted.txt", "lang": "english"}, 

        # 🕒 TO BE SYNCED: Start here
        {"slug": "Social Science", "file": "Social_Science_eng_converted.txt", "lang": "english"}, 
    ]
    
    for s in subjects:
        try:
            print(f"\n--- Starting Ingestion for {s['slug']} ({s['lang']}) ---")
            
            # Check if file exists before trying to open
            if not os.path.exists(s["file"]):
                print(f"⚠️ Warning: File {s['file']} not found. Skipping...")
                continue
                
            with open(s["file"], "r", encoding="utf-8") as f:
                content_text = f.read()
                ingest_safely(s["slug"], s["lang"], content_text)
                
        except Exception as e:
            print(f"❌ Critical Error during {s['slug']} processing: {e}")