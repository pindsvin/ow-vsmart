# OW-vsmart

## Kern
Webapp die de bblthk-catalogus (V-smart via Sambis) inhoudelijk bevraagbaar maakt via een LLM. Gebruikers kunnen natuurlijke-taalvragen stellen over de collectie: titels, auteurs, onderwerpen, beschikbaarheid.

## Scope
- Alleen bblthk-collectie (bibliotheek Wageningen)
- Web-based chat-interface
- LLM-integratie voor semantisch zoeken + real-time catalogusbevraging

## Technisch
- **ILS**: V-smart (Axiell) via Sambis
- **Protocollen**: SRU, OAI-PMH, mogelijk IguanaXgateway REST API
- **Stack**: Next.js + Python (FastAPI) + Chroma (vector store)

## Nu bezig met
- Fase 1: Onderzoek & toegang — uitzoeken welke API's beschikbaar zijn

## Vastgelopen / wacht op
- Nog niet van toepassing

## Besluiten die nog genomen moeten worden
- Welke LLM-provider gebruiken we?
- IguanaXgateway API-key nodig? Zo ja, aanvragen bij Axiell/Sambis
- Hosting: lokaal of cloud?

## Contacten
- **Sambis**: Vereniging Samenwerkend Bibliotheek Informatiesysteem
- **Axiell**: leverancier V-smart (support@axiell.com / kantoor Maarssen)
- **bblthk**: Didi / andere collega's voor toegang

## Vaste context voor de agent
V-smart ondersteunt SRU, Z39.50, OAI-PMH, IguanaXgateway (REST/JSON), SPARQL. De OAI-PMH-endpoint is publiek: https://www.sambis.nl/webopac/oai2.CSP. SRU-endpoint mogelijk op https://www.sambis.nl/webopac/sru. Project volgt de fases uit het plan: onderzoek → architectuur → bouw → test/uitrol.

## Onderzoeksresultaten (30-06-2026)

### API's getest
- **OAI-PMH**: ✅ Beschikbaar — `https://www.sambis.nl/webopac/oai2.CSP`
  - Sets: ALLRECORDS, BOOKS, JOURNALS, COLLECTIEGELDERLAND
  - Geen bblthk-specifieke set — hele Sambis collectie
  - Metadata: oai_dc (Dublin Core), marc21, edm
- **SRU**: ❌ 404 op `/webopac/sru`
- **IguanaXgateway REST**: ❌ 404 op `/api`, `/xgateway`, `/iguana/xgateway`
- **Iguana web**: ✅ `https://www.sambis.nl/iguana/www.main.cls?surl=WGN_SearchHF`
  - bblthk-specifieke pagina (titel: "Bibliotheek Wageningen")
  - Zoekopdracht "klimaat": 108 resultaten in Wageningen
  - Dojo JavaScript frontend, backend via Vubis.csp
  - Resultaten tonen: titel, auteur, type, taal, beschrijving, beschikbaarheid
  - Filters: Wageningen (108), Sambis (454), E-books (529), Landelijk (5441)
- **Vubis.csp**: ✅ `https://www.sambis.nl/webopac/Vubis.csp?Profile=Sambis`

### Conclusie Fase 1
Voor ontsluiting via LLM zijn er twee primaire routes:
1. **OAI-PMH harvesting** → vector database voor semantisch zoeken (bulk, historisch)
   - Nadeel: geen bblthk-filter op set-niveau, moet bij harvesten per record gefilterd worden
2. **Iguana search scraping** → real-time zoeken via de webinterface
   - Nadeel: HTML-parsing, geen officiële API

Aanbevolen: OAI-PMH voor bulk metadata → Chroma vector store. Eventueel later Iguana scraping voor actuele beschikbaarheid.

## Scraper prototype (30-06-2026)

### Werkt
- ✅ Browser-based scraping van Iguana werkt
- ✅ 10 resultaten per pagina, via RowRepeat pagineerbaar
- ✅ Gestructureerde data: titel, auteur, type, taal, beschrijving, PPN, beschikbaarheid
- ✅ PPN (PICA Productie Nummer) als unieke identifier
- ✅ Cover URL beschikbaar via `KB.AZB.cls?ppn=...`

### Output
- `scraper_output_test.json` — 10 resultaten voor "klimaat" in Wageningen (108 totaal)
- `scraper.md` — documentatie van de scrapemethode
- `harvest_test.py` — mislukte OAI-PMH poging (bewaard als referentie)

### Beperkingen
- Via browser-DOM, niet via API → trager, afhankelijk van UI
- Resultaten per pagina gelimiteerd tot 50
- PPN werkt als unieke key, maar geen directe API-toegang tot volledig record
- Sommige titels (zonder auteur) parsen minder netjes

### Volgende stap
- Python/Node backend die de browser aanstuurt voor bulk-scraping
- Resultaten opslaan in lokale JSON/SQLite → vector store
- LLM-koppeling bouwen (chat-interface)

## Vector store prototype (30-06-2026)

### Werkt
- ✅ 90 records uit "klimaat"-search geembed in ChromaDB
- ✅ semantisch zoeken via `all-MiniLM-L6-v2` embeddings
- ✅ Natuurlijke-taalvragen → relevante catalogusresultaten
- ✅ Beschikbaarheid getoond (✅ beschikbaar / ❌ uitgeleend)

### Stack
- **Embedding model**: sentence-transformers `all-MiniLM-L6-v2` (lokaal, 80MB)
- **Vector store**: ChromaDB (persistent, `./chroma_db/`)
- **Query interface**: `embed_and_query.py` (CLI)

### Voorbeeldqueries
| Vraag | Beste match | Relevantie |
|-------|------------|------------|
| Spannend boek over klimaattoekomst voor tieners | "De belangrijkste vragen van je leven" — Harm Edens | ✅ |
| Prentenboeken over ijsberen voor kleuters | "Een ijsbeer in de tropen" — Hans de Beer (4+) | ✅ |
| Boeken over Greta Thunberg | "Ons huis staat in brand" — Jeanette Winter | ✅ |
| Praktische tips duurzamer leven | "Klimaat positief" — Melissa Oosterbroek | ✅ |

### Bestanden
- `catalogus_klimaat.json` — 90 records (volledige dataset)
- `embed_and_query.py` — embedding + query pipeline
- `chroma_db/` — ChromaDB persistente opslag

### Gebruik
```bash
cd Projecten/OW-vsmart
python3 embed_and_query.py "jouw zoekvraag hier"
```

## Chat-interface prototype (30-06-2026)

### Werkt
- ✅ Web-gebaseerde chat-UI op http://localhost:8765
- ✅ Vraag → ChromaDB vector search → Ollama LLM → natuurlijk antwoord
- ✅ LLM geeft contextueel advies (leeftijd, toon, beschikbaarheid)
- ✅ Bronnen met beschikbaarheidsindicatie

### Stack
- **Backend**: FastAPI (Python)
- **Vector search**: ChromaDB met all-MiniLM-L6-v2
- **LLM**: Ollama `gemma3:4b` (lokaal, 3.3GB)
- **Frontend**: HTML/CSS/JS (vanilla, geen framework)

### Starten
```bash
cd Projecten/OW-vsmart
python3 chat_server.py
# Open http://localhost:8765
```

### Voorbeeldinteractie
> **Gebruiker**: "Mijn dochter van 7 is bang voor het klimaat. Hebben jullie een hoopvol boek?"
>
> **Assistent**: "Wat goed dat je naar oplossingen zoekt! Ik raad 'Hitte en kou' van Anneriek van Heugten aan — het legt op begrijpelijke manier uit waarom het klimaat verandert. 'Klimaten' van Geert-Jan Roebers is ook mooi. Beide zijn beschikbaar. De andere boeken gaan meer in op de gevolgen, wat misschien overweldigend kan zijn voor een 7-jarige."

