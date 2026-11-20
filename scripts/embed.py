"""
Embedding ingestion script: chunks contexte.txt and upserts vectors into Pinecone.

Usage:
    python scripts/embed.py

Prerequisites:
    - .env file configured with PINECONE_API_KEY and PINECONE_INDEX_NAME
"""

import os
import re
import sys

# Add root directory to path for app imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "portfolio-chatbot")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "multilingual-e5-large")
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "contexte.txt")
if not os.path.exists(DATA_FILE):
    DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "contexte.txt")

MAX_CHUNK_CHARS = 5000
OVERLAP_CHARS = 200


def load_and_chunk(filepath: str) -> list[dict]:
    """Parse contexte.txt into semantic chunks."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Strip HTML comments and full-line hash comments
    content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
    content = re.sub(r"^# [^\n]*$", "", content, flags=re.MULTILINE)

    # Split by section titles (## TITLE)
    sections = re.split(r"^(## .+)$", content, flags=re.MULTILINE)

    chunks = []
    current_project = None
    current_section = "GENERAL"

    for part in sections:
        part = part.strip()
        if not part:
            continue

        if part.startswith("## "):
            section_title = part.replace("## ", "").strip()
            if section_title.startswith("PROJET :") or section_title.startswith("PROJECT :"):
                current_project = section_title.replace("PROJET :", "").replace("PROJECT :", "").strip()
                current_section = f"PROJECT : {current_project}"
            elif current_project:
                current_section = f"PROJECT : {current_project} > {section_title}"
            else:
                current_project = None
                current_section = section_title
            continue

        if current_section == "GENERAL":
            continue

        sub_chunks = []
        if len(part) <= MAX_CHUNK_CHARS:
            if part:
                sub_chunks.append(part)
        else:
            paragraphs = [p.strip() for p in part.split("\n\n") if p.strip()]
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) + 2 > MAX_CHUNK_CHARS:
                    if current_chunk:
                        sub_chunks.append(current_chunk)
                    current_chunk = para
                else:
                    current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
            if current_chunk:
                sub_chunks.append(current_chunk)

        for text in sub_chunks:
            if current_project:
                prefix = f"[{current_project}] "
                text_with_context = prefix + text
            else:
                text_with_context = f"[{current_section}] " + text

            chunks.append({
                "section": current_section,
                "text": text_with_context
            })

    return chunks


def main():
    print(f"Loading data from: {os.path.abspath(DATA_FILE)}")
    chunks = load_and_chunk(DATA_FILE)

    if not chunks:
        print("Error: No chunks found. Check data/contexte.txt content.")
        sys.exit(1)

    print(f"Extracted {len(chunks)} chunk(s):")
    for i, c in enumerate(chunks):
        print(f"   [{i}] ({c['section']}) {c['text'][:80]}...")

    print(f"\nConnecting to Pinecone index: {PINECONE_INDEX_NAME}")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    try:
        print("Clearing old vectors from Pinecone index before fresh ingestion...")
        index.delete(delete_all=True)
    except Exception as e:
        print(f"Note on delete_all: {e}")

    texts = [c["text"] for c in chunks]
    print(f"Generating embeddings via Pinecone Inference ({EMBEDDING_MODEL})...")
    embeddings = []
    batch_size = 96
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        response = pc.inference.embed(
            model=EMBEDDING_MODEL,
            inputs=batch_texts,
            parameters={
                "input_type": "passage",
                "truncate": "END",
            },
        )
        for data in response.data:
            embeddings.append(data.values)

    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding,
            "metadata": {
                "text": chunk["text"],
                "section": chunk["section"],
                "source": "contexte.txt",
            },
        })

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"   Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")

    print(f"\nCompleted! {len(vectors)} vectors inserted into '{PINECONE_INDEX_NAME}'.")
    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")


if __name__ == "__main__":
    main()
