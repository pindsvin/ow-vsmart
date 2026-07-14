#!/usr/bin/env python3
"""
OW-vsmart: embed de catalogus via de Gemini API (gratis tier).

Eenmalig draaien (en opnieuw na elke datasetwijziging):
    python3 embed_gemini.py

Leest GEMINI_API_KEY uit .env, embedt titel+auteur+beschrijving van elke
entry met gemini-embedding-001 (768 dims, RETRIEVAL_DOCUMENT) en schrijft
genormaliseerde vectoren naar embeddings_gemini.npz. De cloud-server
(chat_server_cloud.py) gebruikt dat bestand voor semantisch zoeken; de
volgorde van de vectoren volgt dataset["results"].
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

os.chdir(str(Path(__file__).parent))
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("❌ GEMINI_API_KEY ontbreekt in .env")
    sys.exit(1)

MODEL = "gemini-embedding-001"
DIMS = 768
BATCH = 50          # ruim onder het API-maximum van 100
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:batchEmbedContents"

print("📚 Laden catalogus...")
with open("catalogus_klimaat.json") as f:
    dataset = json.load(f)
results = dataset["results"]
print(f"   {len(results)} records")

docs = []
for r in results:
    parts = [r['title']]
    if r['author']:
        parts.append(f"door {r['author']}")
    if r['description']:
        parts.append(r['description'])
    docs.append(" | ".join(parts))

print(f"🧠 Embedden via {MODEL} ({DIMS} dims), batches van {BATCH}...")
vectors = []
i = 0
while i < len(docs):
    batch = docs[i:i+BATCH]
    body = {"requests": [
        {"model": f"models/{MODEL}",
         "content": {"parts": [{"text": t}]},
         "taskType": "RETRIEVAL_DOCUMENT",
         "outputDimensionality": DIMS}
        for t in batch
    ]}
    resp = requests.post(URL, params={"key": API_KEY}, json=body, timeout=120)
    if resp.status_code == 429:
        print("   ⏳ rate limit — 30s wachten...")
        time.sleep(30)
        continue
    resp.raise_for_status()
    for emb in resp.json()["embeddings"]:
        vectors.append(emb["values"])
    i += BATCH
    print(f"   {min(i, len(docs))}/{len(docs)}")
    time.sleep(2)  # vriendelijk voor de gratis-tier limieten

matrix = np.array(vectors, dtype=np.float32)
# normaliseren zodat zoeken een simpele dot-product wordt (vereist bij ingekorte dims)
matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

np.savez_compressed("embeddings_gemini.npz", vectors=matrix,
                    ppns=np.array([r['ppn'] for r in results]))
print(f"✅ {matrix.shape[0]} vectoren ({matrix.shape[1]} dims) → embeddings_gemini.npz "
      f"({os.path.getsize('embeddings_gemini.npz')//1024} KB)")
