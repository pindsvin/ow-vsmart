#!/usr/bin/env python3
"""
OW-vsmart: Catalogus embedding & query pipeline
- Laadt catalogus_klimaat.json
- Embedt titel + beschrijving met sentence-transformers
- Slaat op in ChromaDB
- Biedt semantische zoekfunctie
"""
import json
import sys
import os

# Work in project dir
os.chdir("/Users/ron/Documents/ronOS/Projecten/OW-vsmart")

print("=" * 60)
print("OW-vsmart: Vector Store & Query Pipeline")
print("=" * 60)

# 1. Laad data
print("\n📚 Laden van catalogus...")
with open("catalogus_klimaat.json", "r") as f:
    dataset = json.load(f)
results = dataset["results"]
print(f"   {len(results)} records geladen")

# 2. Maak embedding model
print("\n🧠 Laden van embedding model (all-MiniLM-L6-v2)...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print("   Model geladen")

# 3. Bouw documenten voor embedding
print("\n📝 Voorbereiden documenten...")
documents = []
metadatas = []
ids = []

for i, r in enumerate(results):
    # Combineer titel, auteur, beschrijving voor betere embedding
    text_parts = [r['title']]
    if r['author']:
        text_parts.append(f"door {r['author']}")
    if r['description']:
        text_parts.append(r['description'])
    
    doc = " | ".join(text_parts)
    documents.append(doc)
    
    metadatas.append({
        "title": r['title'],
        "author": r['author'],
        "type": r['type'],
        "language": r['language'],
        "onLoan": r['onLoan'],
        "ppn": r['ppn']
    })
    
    ids.append(f"ppn_{r['ppn']}")

print(f"   {len(documents)} documenten klaar voor embedding")

# 4. Embed en sla op in ChromaDB
print("\n💾 Embedden en opslaan in ChromaDB...")
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path="./chroma_db")

# Verwijder bestaande collectie als die er is
try:
    client.delete_collection("bblthk_catalog")
    print("   Oude collectie verwijderd")
except:
    pass

collection = client.create_collection(
    name="bblthk_catalog",
    metadata={"description": "bblthk catalogus — zoekterm: klimaat"}
)

# Embed in batches
batch_size = 32
for i in range(0, len(documents), batch_size):
    batch_docs = documents[i:i+batch_size]
    batch_meta = metadatas[i:i+batch_size]
    batch_ids = ids[i:i+batch_size]
    
    collection.add(
        documents=batch_docs,
        metadatas=batch_meta,
        ids=batch_ids
    )
    print(f"   Batch {i//batch_size + 1}: {len(batch_docs)} docs geembed")

print(f"\n✅ {collection.count()} documenten in ChromaDB")

# 5. Query functie
def query_catalog(question, n_results=5):
    """Zoek de catalogus semantisch."""
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    
    print(f"\n🔍 Vraag: \"{question}\"")
    print(f"   Top {n_results} resultaten:\n")
    
    for i, (doc_id, doc_text, meta, distance) in enumerate(zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        status = "❌ Uitgeleend" if meta['onLoan'] else "✅ Beschikbaar"
        print(f"   {i+1}. [{status}] {meta['title']}")
        if meta['author']:
            print(f"      Auteur: {meta['author']}")
        # Toon snippet van de tekst
        snippet = doc_text.split(" | ")[-1][:150]
        if snippet:
            print(f"      {snippet}")
        print(f"      Relevantie: {1-distance:.2f}")
        print()

# 6. Test queries
print("\n" + "=" * 60)
print("TEST QUERIES")
print("=" * 60)

test_questions = [
    "Een spannend boek over de toekomst na klimaatverandering voor tieners",
    "Prentenboeken over ijsberen voor kleuters",
    "Boeken over Greta Thunberg en haar activisme",
    "Praktische tips om zelf duurzamer te leven",
    "Informatieve boeken over het weer voor kinderen van 8 tot 10 jaar",
]

for q in test_questions:
    query_catalog(q)
    print("-" * 40)

# 7. Interactieve modus (als er argumenten zijn)
if len(sys.argv) > 1:
    user_query = " ".join(sys.argv[1:])
    query_catalog(user_query, n_results=8)
else:
    print("\n💡 Gebruik: python3 embed_and_query.py 'jouw zoekvraag'")
    print("   Voorbeeld: python3 embed_and_query.py 'spannend boek over klimaat voor 12 jarige'")
