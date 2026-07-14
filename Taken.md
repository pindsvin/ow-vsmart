# Taken — OW-vsmart

## Open
- [ ] Collega's laten testen (demo: https://ow-vsmart.onrender.com)
- [ ] API-toegang volledige catalogus regelen (IguanaXgateway-key aanvragen bij Axiell/Sambis, of OAI-PMH met bblthk-filter)
- [ ] Opschalen naar volledige catalogus (35-40k titels) zodra API-toegang er is
- [ ] Live beschikbaarheid (realtime Iguana-bevraging per getoonde bron)
- [ ] Meerdere databronnen toevoegen (Samenwerkende Bibliotheken, E-books)
- [ ] Markdown in LLM-antwoorden netjes renderen of strippen in de UI

## Gedaan
- [x] Project aangemaakt
- [x] API-onderzoek (OAI-PMH ❌, SRU ❌, Iguana REST ❌, browser scraping ✅)
- [x] Bulk-scraper: 90 resultaten voor "klimaat" opgehaald
- [x] ChromaDB vector store + semantische query's
- [x] Chat-interface (lokaal) met Ollama LLM
- [x] Chat-interface (cloud) met DeepSeek API
- [x] Deploy-documentatie + render.yaml
- [x] Deployen naar Render (auto-deploy vanaf GitHub master)
- [x] Meerdere zoektermen scrapen — 721 records over 19 zoektermen / 20 thema's (14-07-2026)
- [x] UI verfraaien (bblthk-huisstijl, design handoff) + statistiekstrook + titel-autocomplete
- [x] Bronkaarten gefilterd op relevantie (LLM-beoordeling via BRONNEN-regel)
- [x] Semantisch zoeken in de cloud (voorberekende Gemini-embeddings, gratis tier)
- [x] Semantisch zoeken lokaal verbeterd (meertalig model i.p.v. Engelse default)
- [x] Documentatie bijgewerkt naar demo v2 (OW-vsmart.md, DEPLOY.md)
