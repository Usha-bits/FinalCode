import json
import re
import os

# Base directories
base_path = os.path.dirname(os.path.abspath(__file__))
SCRAP_DIR = os.path.join(base_path, "output", "ScrapData")
CLEAN_DIR = os.path.join(base_path, "output", "CleanedData")

# Input files
TEXT_DATA_FILE = os.path.join(SCRAP_DIR, "text_data.json")
TABLE_DATA_FILE = os.path.join(SCRAP_DIR, "table_data.json")

# Output files
CLEANED_TEXT_FILE = os.path.join(CLEAN_DIR, "cleaned_text_data.json")
CLEANED_TABLE_FILE = os.path.join(CLEAN_DIR, "cleaned_table_data.json")

# Ensure output directory exists
os.makedirs(CLEAN_DIR, exist_ok=True)

def clean_text(text):
    """Removes excessive whitespace and fixes special characters."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # curly to straight quotes
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_text_data(file_path, output_path):
    """Process and clean text-based entries."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            content = content.strip()
        
            # Try parsing as full JSON array
            if content.startswith("["):
                data = json.loads(content)
            else:
                # Fallback to JSONL (each line is a JSON object)
                data = [json.loads(line) for line in content.splitlines() if line.strip()]
    except json.JSONDecodeError as e:
        print(f" JSON decode error in text file: {e}")
        return

    cleaned = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        content = entry.get("content", {})
        cleaned_content = {
            heading: clean_text(text)
            for heading, text in content.items()
            if clean_text(text)
        }
        if cleaned_content:
            cleaned.append({
                "url": entry.get("url", ""),
                "title": entry.get("title", ""),
                "content": cleaned_content
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=4)
    print(f"✅ Cleaned text data saved to {output_path}")

def clean_table_data(file_path, output_path):
    """Process and clean table-based entries."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("["):
                data = json.loads(content)
            else:
                data = [json.loads(line) for line in content.splitlines() if line.strip()]
    except json.JSONDecodeError as e:
        print(f" JSON decode error in table file: {e}")
        return

    cleaned = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        cleaned_tables = []
        content = entry.get("content", {})
        for heading, tables in content.items():
            for table in tables:
                cleaned_table = []
                for row in table:
                    cleaned_row = [clean_text(cell) for cell in row if clean_text(cell)]
                    if cleaned_row:
                        cleaned_table.append(cleaned_row)
                if cleaned_table:
                    cleaned_tables.append({
                        "section": heading,
                        "table": cleaned_table
                    })
        if cleaned_tables:
            cleaned.append({
                "url": entry.get("url", ""),
                "title": entry.get("title", ""),
                "tables": cleaned_tables
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=4)
    print(f" Cleaned table data saved to {output_path}")

# Run both processors
clean_text_data(TEXT_DATA_FILE, CLEANED_TEXT_FILE)
clean_table_data(TABLE_DATA_FILE, CLEANED_TABLE_FILE)
