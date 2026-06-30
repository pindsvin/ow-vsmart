#!/usr/bin/env python3
"""
Test harvest van 100 OAI-PMH records van Sambis.
Gebruikt oai_dc (Dublin Core) formaat.
Analyseert welke velden beschikbaar zijn en of er vestiging/locatie info in zit.
"""
import xml.etree.ElementTree as ET
import urllib.request
import time
import ssl

OAI_BASE = "https://www.sambis.nl/webopac/oai2.CSP"

# Ignore SSL cert errors for older servers
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

ns = {
    'oai': 'http://www.openarchives.org/OAI/2.0/',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'oai_dc': 'http://www.openarchives.org/OAI/2.0/oai_dc/',
}

def fetch_url(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'OW-vsmart/0.1'})
    resp = urllib.request.urlopen(req, context=ssl_ctx, timeout=30)
    return resp.read()

def harvest_records(count=100):
    """Haal <count> records op via OAI-PMH ListRecords."""
    records = []
    resumption_token = None
    
    while len(records) < count:
        if resumption_token:
            url = f"{OAI_BASE}?verb=ListRecords&resumptionToken={resumption_token}"
        else:
            url = f"{OAI_BASE}?verb=ListRecords&metadataPrefix=oai_dc"
        
        print(f"  Fetching: {url[:100]}...")
        data = fetch_url(url)
        root = ET.fromstring(data)
        
        # Check for error
        error = root.find('oai:error', ns)
        if error is not None:
            print(f"  OAI Error: {error.text}")
            break
        
        # Extract records
        for rec in root.findall('.//oai:record', ns):
            header = rec.find('oai:header', ns)
            metadata = rec.find('.//oai_dc:dc', ns)
            
            if header is not None and metadata is not None:
                identifier = header.find('oai:identifier', ns)
                datestamp = header.find('oai:datestamp', ns)
                
                rec_data = {
                    'id': identifier.text if identifier is not None else 'unknown',
                    'date': datestamp.text if datestamp is not None else 'unknown',
                }
                
                # Extract Dublin Core fields
                for child in metadata:
                    tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                    val = child.text.strip() if child.text else ''
                    if val:
                        if tag not in rec_data:
                            rec_data[tag] = val
                        elif isinstance(rec_data[tag], list):
                            rec_data[tag].append(val)
                        else:
                            rec_data[tag] = [rec_data[tag], val]
                
                records.append(rec_data)
                if len(records) >= count:
                    break
        
        # Check for resumptionToken
        rt = root.find('.//oai:resumptionToken', ns)
        if rt is not None and rt.text:
            resumption_token = rt.text
        else:
            break
        
        time.sleep(0.5)  # Be nice to the server
    
    return records[:count]

def analyze(records):
    """Analyseer de opgehaalde records."""
    print(f"\n{'='*60}")
    print(f"RESULTATEN: {len(records)} records opgehaald")
    print(f"{'='*60}")
    
    # Alle voorkomende velden tellen
    field_counts = {}
    field_examples = {}
    
    for rec in records:
        for field, value in rec.items():
            if field in ('id', 'date'):
                continue
            field_counts[field] = field_counts.get(field, 0) + 1
            if field not in field_examples:
                field_examples[field] = value
    
    print(f"\n--- VELDEN (hoe vaak in {len(records)} records) ---")
    for field, count in sorted(field_counts.items(), key=lambda x: -x[1]):
        pct = count / len(records) * 100
        example = field_examples[field]
        if isinstance(example, list):
            example = example[0]
        if len(str(example)) > 80:
            example = str(example)[:80] + '...'
        print(f"  {field:<20} {pct:5.0f}% ({count:3d})  voorbeeld: {example}")
    
    # Check voor vestiging/locatie metadata
    print(f"\n--- CHECK: vestiging/locatie velden ---")
    location_fields = ['publisher', 'coverage', 'source', 'type', 'format', 'relation', 'rights']
    for field in location_fields:
        if field in field_counts:
            print(f"  '{field}' aanwezig in {field_counts[field]} records")
        else:
            print(f"  '{field}' NIET aanwezig")
    
    # Toon 3 complete records
    print(f"\n--- 3 VOLLEDIGE RECORDS ---")
    for i, rec in enumerate(records[:3]):
        print(f"\n  Record #{i+1}: {rec.get('id', '?')}")
        for key, val in rec.items():
            if key in ('id', 'date'):
                continue
            if isinstance(val, list):
                for v in val:
                    print(f"    {key}: {v[:120]}")
            else:
                print(f"    {key}: {val[:120]}")
    
    # Samenvatting
    print(f"\n{'='*60}")
    print("CONCLUSIE")
    print(f"{'='*60}")
    print(f"Nuttige velden voor LLM embeddings:")
    wanted = ['title', 'creator', 'subject', 'description', 'publisher', 'date', 'type', 'language', 'identifier']
    for w in wanted:
        present = "✅" if w in field_counts else "❌"
        count = field_counts.get(w, 0)
        print(f"  {present} {w:<20} ({count} records)")

if __name__ == '__main__':
    print("OW-vsmart: OAI-PMH test harvest")
    print(f"Ophalen van 100 records...\n")
    records = harvest_records(100)
    analyze(records)
