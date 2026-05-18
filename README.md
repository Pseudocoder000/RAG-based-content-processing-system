
| # | Topic |
|---|-------|
| 1 | Why Vector RAG fails on professional documents |
| 2 | How PageIndex builds a tree index from a PDF |
| 3 | LLM Tree Search — reasoning over structure |
| 4 | Full end-to-end Vectorless RAG pipeline |
| 5 | Expert-guided retrieval (domain knowledge injection) |
| 6 | Chat API — zero LLM setup |
| 7 | Self-hosted open-source option |

---

## 🔑 Key Concept

> **Traditional RAG** → chunk → embed → cosine similarity → retrieve  
> **PageIndex RAG** → build tree → LLM reasons over tree → retrieve exact sections

**The problem with vector RAG:**  
`Similarity ≠ Relevance`  
A chunk about "market conditions" may score higher than the actual answer section just because it shares more words with your query.

---
## 📦 Section 1: Install & Setup

**What we do here:**
- Install PageIndex SDK + OpenAI
- Load API keys from `.env`
- Initialize both clients

> 🔑 Get your **PageIndex API key** from: https://dash.pageindex.ai/api-keys  
> 🔑 Get your **OpenAI API key** from: https://platform.openai.com

# Install required packages
!pip install -U pageindex openai python-dotenv
# ── Create a .env file (run this once) ──────────────────────────────────────
# Uncomment and fill in your keys, then run this cell ONCE

# env_content = """
# PAGEINDEX_API_KEY=your_pageindex_key_here
# OPENAI_API_KEY=your_openai_key_here
# """
# with open(".env", "w") as f:
#     f.write(env_content.strip())
# print("✅ .env file created")
import os, json, time
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = "6c2739b0ebba162168b47f0e2e"
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")

print("PageIndex key loaded:", "✅" if PAGEINDEX_API_KEY else "❌ Missing!")
print("OpenAI key loaded:   ", "✅" if OPENAI_API_KEY    else "❌ Missing!")
from pageindex import PageIndexClient
from openai import OpenAI

pi_client     = PageIndexClient(api_key=PAGEINDEX_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

print("✅ PageIndex client ready")
print("✅ OpenAI client ready")
---
## 🌲 Section 2: Upload & Index a PDF

**What happens here:**
1. Upload your PDF to the PageIndex cloud
2. PageIndex uses an LLM to read the document structure
3. Builds a hierarchical **tree index** (like a smart Table of Contents)
4. Returns a `doc_id` for all future operations

**Why NO chunking?**  
Instead of cutting the document into arbitrary 500-token pieces, PageIndex respects the document's natural section boundaries — chapters, sub-sections, paragraphs — as the author intended.

# ── Upload your PDF ─────────────────────────────────────────────────────────
# Replace with the path to your PDF file
# Great candidates: Annual reports, research papers, legal docs, textbooks

PDF_PATH = "./sample_document.pdf"   # ← change this

print(f"📤 Uploading: {PDF_PATH}")
result = pi_client.submit_document(PDF_PATH)
doc_id = result["doc_id"]

print(f"✅ Uploaded!")
print(f"📋 Document ID: {doc_id}")
print("   (Save this ID — you'll use it throughout the notebook)")
# ── Poll until processing is complete ───────────────────────────────────────
# PageIndex builds the tree asynchronously.
# For a 50-page PDF this typically takes 30–90 seconds.

print("⏳ Building tree index...")
print("   (This runs once per document — the index is cached for reuse)")

while True:
    status_result = pi_client.get_document(doc_id)
    status = status_result.get("status")
    print(f"   Status: {status}")
    
    if status == "completed":
        print("\n✅ Tree index ready!")
        break
    elif status == "failed":
        print("\n❌ Processing failed. Check your PDF format.")
        break
    
    time.sleep(5)
---
## 🔍 Section 3: Inspect the Tree Structure

**What the tree looks like:**

```
Document
├── Introduction (pages 1-3)
│   └── Background (pages 1-2)
├── Financial Stability (pages 21-31)
│   ├── Monitoring Vulnerabilities (pages 22-28)
│   └── International Cooperation (pages 28-31)
└── Conclusion (pages 45-47)
```

Each node has:
- `node_id` — unique ID used during retrieval
- `title` — section heading
- `page_index` — page number in original PDF
- `text` — section summary (when `node_summary=True`)
- `nodes` — child sections (nested)
