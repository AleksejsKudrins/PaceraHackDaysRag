from pypdf import PdfReader
import sys

try:
    reader = PdfReader("RAG_Mastery_Building_Dynamic_AI.pdf")
    print(f"Num pages: {len(reader.pages)}")
    if len(reader.pages) > 0:
        page = reader.pages[0]
        text = page.extract_text()
        print(f"Text from page 0 (first 100 chars): '{text[:100]}'")
except Exception as e:
    print(f"Error: {e}")
