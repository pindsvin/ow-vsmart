# Vervang de bestaande routes in chat_server.py door dit blok.
# De rest van het bestand (imports, ChromaDB, Ollama, /api/chat) blijft ongewijzigd.

# Verwijder de CHAT_HTML variabele en de @app.get("/") route.
# Voeg onderaan de imports toe (staat al in het origineel, alleen nog mounten):

from fastapi.staticfiles import StaticFiles

# Mount VOOR de /api routes, vlak na `app = FastAPI(...)`:
app.mount("/static", StaticFiles(directory="static"), name="static")

# Nieuwe root route — stuurt door naar de HTML:
from fastapi.responses import RedirectResponse

@app.get("/")
async def index():
    return RedirectResponse(url="/static/index.html")
