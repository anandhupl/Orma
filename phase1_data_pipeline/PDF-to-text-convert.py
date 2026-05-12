import os
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv

# --- INITIALIZATION & SECURITY ---
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("SECURITY HALT: GEMINI_API_KEY not found in environment variables.")

ai_client = genai.Client(api_key=API_KEY)

# --- CONFIGURATION ---
PDF_FILE_PATH = "Social_Science_eng.pdf" 
OUTPUT_FILE_PATH = "Social_Science_eng_converted.txt"
EXTRACTION_MODEL = "gemini-2.5-pro"

EXTRACTION_PROMPT = """
<Instructions>
You are an expert academic data extraction system. Your singular task is to analyze 
the provided image of a textbook page and transcribe all textual, tabular, and 
mathematical content into strict Markdown formatting.

1. Locate the actual printed page number on the page (usually at the bottom or top corners).
2. You MUST start your response with a clear page delimiter in this exact format: `\n\n--- Page [Number] ---\n\n`. If there is absolutely no printed page number visible on the image, output `\n\n--- Page Unknown ---\n\n`.
3. Read the text naturally, correctly navigating multi-column visual layouts.
4. Convert all mathematical equations into valid LaTeX syntax. Mandatorily use 
   \\frac{} for fractions. Wrap block equations in $$ and inline math in $.
5. Extract all tabular data and format it strictly using Markdown table syntax.
6. Omit all other extraneous headers, footers, or watermarks, but extract the core textbook content perfectly.
7. Provide ONLY the raw Markdown. Do not include conversational filler, preambles, 
   or concluding remarks.
</Instructions>
"""

@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(6))
def extract_markdown_from_page(page_number: int, image_bytes: bytes) -> tuple[int, str]:
    try:
        response = ai_client.models.generate_content(
            model=EXTRACTION_MODEL,
            contents=[
                EXTRACTION_PROMPT,
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=8192
            )
        )
        return (page_number, response.text.strip() + "\n\n")
    except Exception as e:
        print(f"Network/Quota error processing page {page_number + 1}: {str(e)}")
        raise e

def process_pdf_to_local(pdf_path: str, output_path: str):
    print(f"Initializing extraction for: {pdf_path}")
    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)
    extracted_pages = {}
    
    # 144 DPI scaling for clear superscript vision
    zoom_matrix = fitz.Matrix(2, 2)
    
    print(f"Initiating concurrent Map-Reduce extraction for {total_pages} pages...")
    # Keep workers at 5 initially to avoid API quota exhaustion
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_page = {}
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=zoom_matrix)
            image_bytes = pix.tobytes("jpeg")
            
            future = executor.submit(extract_markdown_from_page, page_num, image_bytes)
            future_to_page[future] = page_num
            
        for future in as_completed(future_to_page):
            p_num = future_to_page[future]
            try:
                page_index, md_text = future.result()
                extracted_pages[page_index] = md_text
                print(f"Successfully processed Page {p_num + 1}/{total_pages}")
            except Exception as e:
                print(f"Catastrophic failure on page {p_num + 1}: {e}")
                extracted_pages[p_num] = f"\n\n[CRITICAL ERROR: PAGE EXTRACTION FAILED FOR PAGE {p_num + 1}]\n\n"
                
    pdf_document.close()

    print("Sequential concatenation executing...")
    full_markdown_corpus = ""
    for i in range(total_pages):
        full_markdown_corpus += extracted_pages.get(i, "")
        
    print("Writing artifact to local storage...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_markdown_corpus)
        
    print(f"Artifact secured at: {output_path}")

if __name__ == "__main__":
    process_pdf_to_local(PDF_FILE_PATH, OUTPUT_FILE_PATH)