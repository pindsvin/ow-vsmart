# Iguana Scraper — OW-vsmart

## Werkwijze
De Sambis Iguana-interface (bblthk) gebruikt Dojo JavaScript. Zoekresultaten worden client-side gerenderd in de DOM. De scraper gebruikt browser-automation om resultaten te extracten.

## Flow
1. Navigeer naar `https://www.sambis.nl/iguana/www.main.cls?surl=WGN_SearchHF`
2. Wacht tot pagina geladen is
3. Vul zoekveld: `document.getElementById('dojoUnique1_inputWords1')`
4. Klik Zoek: `document.getElementById('dijit_form_Button_0')`
5. Wacht op resultaten (poll op `.data` divs)
6. Extraheer resultaten via JavaScript
7. Dedupliceer op PPN (uit cover-URL)

## JS Extractiecode
```javascript
(function() {
  const results = [];
  const seen = new Set();
  
  document.querySelectorAll('div').forEach(div => {
    const hasImage = div.querySelector('.image, [id*="cover"]');
    const dataDiv = div.querySelector('.data');
    if (!hasImage || !dataDiv) return;
    
    const lines = dataDiv.innerText.trim().split('\n')
      .map(l => l.trim()).filter(l => l && l.length > 1);
    if (lines.length < 2) return;
    
    // Skip UI elements
    const first = lines[0];
    if (first.startsWith('Alle exemplaren') || 
        first.startsWith('Waardering') ||
        first.includes('selecteer alles')) return;
    
    // Deduplicate on PPN
    const img = div.querySelector('.image img');
    const ppn = (img?.src?.match(/ppn=(\d+)/) || [])[1] || first.substring(0, 100);
    if (seen.has(ppn)) return;
    seen.add(ppn);
    
    results.push({
      title: lines[0],
      author: lines[1] || '',
      type: lines[2] || '',
      language: lines[3] || '',
      description: lines.slice(4).join(' ') || '',
      onLoan: div.innerText.includes('uitgeleend'),
      ppn: ppn
    });
  });
  
  return results;
})()
```

## Resultaatstructuur
| Veld | Bron | Voorbeeld |
|------|------|-----------|
| title | `.data` regel 1 | "Eigen planeet eerst ..." |
| author | `.data` regel 2 | "Roxane van Iperen" |
| type | `.data` regel 3 | "Boek" |
| language | `.data` regel 4 | "Nederlands" |
| description | `.data` regel 5+ | "Essay over de beperkingen..." |
| onLoan | tekst-check | true/false |
| ppn | cover URL param | "444279873" |

## Kanttekeningen
- Boeken zonder auteur-lijn (bijv. atlassen, bloemlezingen) verschuiven de regel-index
- Paginering: URL bevat `RowRepeat` parameter (0, 10, 20, ...)
- Max 50 resultaten per pagina (`NumberToRetrieve=50` in form)
- Filter op Wageningen gebeurt via checkbox `db_2_WGN` in Dojo

## Paginering
Elke pagina toont 10 resultaten. Wijzig `RowRepeat`:
- Pagina 1: RowRepeat=0
- Pagina 2: RowRepeat=10
- Pagina n: RowRepeat=(n-1)*10
