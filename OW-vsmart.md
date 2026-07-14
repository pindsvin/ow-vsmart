# OW-vsmart

## Kern
Webapp die de bblthk-catalogus (V-smart via Sambis) inhoudelijk bevraagbaar maakt via een LLM. Gebruikers kunnen natuurlijke-taalvragen stellen over de collectie: titels, auteurs, onderwerpen, beschikbaarheid.

## Scope
- Alleen bblthk-collectie (bibliotheek Wageningen)
- Web-based chat-interface
- LLM-integratie voor semantisch zoeken + real-time catalogusbevraging

## Technisch
- **ILS**: V-smart (Axiell) via Sambis
- **Stack cloud (live demo)**: FastAPI + Gemini-embeddings (voorberekend) + DeepSeek — https://ow-vsmart.onrender.com
- **Stack lokaal**: FastAPI + ChromaDB (meertalige sentence-transformers) + DeepSeek
- **Data**: 721 records gescrapet uit Iguana (zie `scraper.md`), 20 thema's

## Nu bezig met
- Demo compleet en live; volgende stap is opschalen naar de volledige catalogus (35-40k titels)

## Vastgelopen / wacht op
- API-toegang tot de volledige catalogus (IguanaXgateway of OAI-PMH met bblthk-filter) — nodig voor opschaling en live beschikbaarheid

## Besluiten die nog genomen moeten worden
- IguanaXgateway API-key aanvragen bij Axiell/Sambis?
- Bij opschaling naar 40k: embeddings via Gemini-API (gratis/goedkoop, huidige route) of eigen hosting met 2GB+ RAM

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


## Demo v2 — volledige pijplijn live (14-07-2026)

### Wat er staat
Live demo op https://ow-vsmart.onrender.com (gratis Render-tier) met:
- **721 records, 20 thema's** — geoogst uit Iguana met 19 zoektermen (klimaat,
  duurzaamheid, natuur, biodiversiteit, koken, geschiedenis, thriller,
  kinderboeken, kunst, sport, roman, muziek, psychologie, filosofie, reizen,
  tuinieren, wetenschap, gezondheid, poëzie, opvoeding), gededupliceerd op PPN.
  Elk record heeft een `theme`-veld.
- **Semantisch zoeken op de gratis tier**: catalogus eenmalig geëmbed met
  `gemini-embedding-001` (768 dims) via `embed_gemini.py` → vectoren in
  `embeddings_gemini.npz` (~2MB, in de repo). Per vraag één Gemini
  embedding-call + numpy dot-product. Automatische TF-IDF-fallback
  (met Nederlandse stopwoordfilter) als key of vectorbestand ontbreekt.
- **Bronfiltering door de LLM**: DeepSeek krijgt de top-8 genummerd en sluit
  af met een `BRONNEN:`-regel; alleen die titels worden als bronkaart getoond.
- **UI (bblthk-huisstijl)**: statistiekstrook (aantal titels + themaverdeling
  via `/api/stats`), titel-autocomplete met Tab op titels uit eerdere
  resultaten, bronkaarten met beschikbaarheidsindicatie.

### Lokale variant
`chat_server.py` draait dezelfde ervaring op ChromaDB met het meertalige
model `paraphrase-multilingual-MiniLM-L12-v2` (herindexeren met
`embed_and_query.py`). Let op: eerdere versies gebruikten stilzwijgend
Chroma's Engelstalige default-embedding — de embedding-functie wordt nu
expliciet meegegeven.

### Belangrijkste lessen
- Handgeschreven TF-IDF kapte de vocabulaire alfabetisch af (`sorted()[:3000]`)
  — bij >90 records vielen alle woorden na ~"h" uit de index. Gefixt met
  selectie op documentfrequentie, daarna vervangen door embeddings.
- Sentence-transformers past niet in 512MB Render; voorberekende embeddings
  + query-embedding via API wél. Render Starter ($7) heeft óók maar 512MB;
  de eerste tier met 2GB is Standard ($25).
- Gemini API heeft een gratis tier (aistudio.google.com) — geen OpenAI-key
  nodig voor embeddings; Anthropic biedt geen embeddings-API.

### Opschalen naar 35-40k titels
Architectuur is er klaar voor: 40k titels ≈ 60MB vectoren (past in 512MB
met numpy), eenmalig embedden binnen de gratis Gemini-limieten (gespreid)
of voor enkele euro's betaald. Bottleneck is de data-aanvoer: API-toegang
(IguanaXgateway/OAI-PMH) nodig in plaats van browser-scraping. Live
beschikbaarheid blijft een aparte route (realtime Iguana-bevraging per
getoonde bron).

### Workflow bij datasetwijziging
```bash
python3 embed_gemini.py      # Gemini-vectoren (cloud)
python3 embed_and_query.py   # ChromaDB (lokaal, optioneel)
git add catalogus_klimaat.json embeddings_gemini.npz && git commit && git push
```

## Demo v2.1 — afronding demosessie (14-07-2026, avond)

- **Info-popup** ("i"-knop in de header): uitleg voor leken over wat het
  semantisch zoeken toevoegt, met twee vroeger/nu-voorbeelden.
- **Beschrijvingen in de LLM-context** (cloud): DeepSeek kreeg alleen
  titel/auteur en keurde daardoor te veel bronnen af; nu gaat per titel ook
  de beschrijving (200 tekens) mee — cloud en lokaal geven gelijkwaardige tips.
- **Meer-info per titel**: vraagt de gebruiker om meer informatie over één
  titel, dan geeft DeepSeek een INFO-regel terug en zoekt de server online
  via Gemini met Google Search-grounding (model gemini-flash-lite-latest).
  De UI toont een "Meer over dit boek"-blok met samenvatting en bronlink
  (directe vindplaats uit grounding-chunks, anders Google-zoeklink als
  fallback). Google Books API bleek onbruikbaar: keyloos deel je één
  anoniem dagquotum dat continu vol zit.
- **Ruimhartige bronselectie**: de INFO-promptuitbreiding maakte DeepSeek
  onbedoeld strenger (alleen nog letterlijke titelmatches als bron); prompt
  vraagt nu expliciet om ook deels passende titels mee te nemen en de
  temperature ging van 0.7 naar 0.4 voor consistentie.
- **Budget**: $5 gekoppeld aan de Gemini-key (grounding kost ~1-3 cent per
  meer-info-opzoeking; embeddings en chat verwaarloosbaar). Budgetalert in
  Google Cloud aangeraden. Vrije tier bleek te krap op dagen met veel
  testen/embedden: tekstmodellen en embeddings hebben aparte dagquota.
