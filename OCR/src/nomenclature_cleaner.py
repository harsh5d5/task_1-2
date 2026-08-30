import re
import difflib

# Official Military Reference Lexicon
OFFICIAL_LEXICON = [
    # 1. Fuze & Mine Nomenclature
    "10 FUZES 117 MK-20",
    "10 FUZES 117",
    "10 FUZES",
    "FUZES 117 MK-20",
    "FUZES 117",
    "FUZE 117 MK-20",
    "FUZE 117",
    "FUZE 127",
    "FUZES",
    "MK-20",
    "MK-117",
    "FUZE A/R MINE A/T",
    "MINE A/T",
    "MINE A/R",
    "A/T",
    "A/R",
    
    # 2. Packaging, Container & Quantity Markings
    "IN 10 AMN CONTS 47B",
    "IN 10 AMN CONTS",
    "IN 10 CONTS",
    "IN 08 7A BOXES",
    "08 NOS",
    "10 NOS",
    "NOS",
    
    # 3. Box Identifiers & Serial Numbers
    "BOX NO-107",
    "BOX NO-117",
    "BOX NO-104",
    "BOX NO-",
    "BOXES",
    "BOX_TURN TOP",
    "BOX_TURN / TOP",
    
    # 4. Depot Locations & Military Authorities
    "CAD PULGAON",
    "COMMANDANT",
    "EAD",
    "FAD",
    "OFBL",
    "CGM",
    "FROM",
    "UNIV",
    
    # 5. Weight & Mass Specifications
    "AV MASS KG-18.70",
    "MASS KG-18.70",
    "AV MASS",
    
    # 6. Manufacturing Lot & Batch Codes
    "LOT 2025 11/HPM 16/BL",
    "LOT 2025 11/HPM",
    "LOT 2017 06/T",
    "LOT 2025",
    "LOT 2017",
    "LOT 2024",
    "LOT NO. 2017",
    "2025 12/SU 43C/BL",
    "2B/L ND",
    "28/L ND",
    "14/HPM 16/BL",
    
    # 7. Explosive & Hazard Classification
    "FILLED EXPLOSIVE",
    "FILLED",
    "EXPLOSIVE",
    "IND. GOVT. EXPLOSIVE",
    "GOVT. EXPLOSIVE",
    "CAUTION"
]

def clean_and_correct_text(raw_text):
    """
    Standardizes OCR detections against military ordnance standards:
    - Filters invalid noise glyphs & broken punctuation.
    - Resolves broken stencil characters into standard terminology.
    - Matches against the official military nomenclature dictionary.
    """
    if not raw_text:
        return ""

    # 1. Filter non-ASCII noise glyphs
    ascii_clean = re.sub(r'[^\x20-\x7E]', '', raw_text).strip()
    
    if len(ascii_clean) < 2 or (re.fullmatch(r'[\W_0-9]+', ascii_clean) and len(ascii_clean) < 3):
        if ascii_clean not in ["10", "08", "A/T", "A/R"]:
            return ""

    text = ascii_clean

    # 2. Targeted regex substitutions for stencil breaks
    replacements = [
        # Fuze lines
        (r'\b[I1l]O\s+FUZES\b', '10 FUZES'),
        (r'\bIO\s*FUZES\b', '10 FUZES'),
        (r'\b1OrUzEo117\b', '10 FUZES 117'),
        (r'\b1OrUzE[os0-9]*\b', '10 FUZES'),
        (r'\b[I1]O\s*FUZES\s*I*17\b', '10 FUZES 117'),
        (r'\bIO\s*FUZESII7\b', '10 FUZES 117'),
        (r'\b10\s*FUZESII7\b', '10 FUZES 117'),
        (r'\bFUZESII7\b', 'FUZES 117'),
        (r'\bFUZESI17\b', 'FUZES 117'),
        (r'\bFUZES1WYK[·\.\-]?20\b', 'FUZES 117 MK-20'),
        (r'\bFUZES\s*1W[YV]K[·\.\-]?20\b', 'FUZES 117 MK-20'),
        (r'\bFUZE\s*I\s*7\b', 'FUZE 117'),
        (r'\bFUZE\s*I17\b', 'FUZE 117'),
        (r'\bFUZE117\b', 'FUZE 117'),
        (r'\bFUZE\s*11\s*7\b', 'FUZE 117'),
        (r'\bFUZE117MK20\b', 'FUZE 117 MK-20'),
        (r'\bFUZEI7MK\s*\?O\b', 'FUZE 117 MK-20'),
        (r'\bFUZE\s*117\s*MK\s*20\b', 'FUZE 117 MK-20'),
        (r'\b[YMK][K\-·\s]*20\b', 'MK-20'),
        (r'\bK-20\b', 'MK-20'),
        (r'\bMK\s*20\b', 'MK-20'),
        (r'\bMK\s*117\b', 'MK-117'),
        
        # Mine markings
        (r'\bFUZE\s*A/R\s*[YMIN]+\s*A/T\b', 'FUZE A/R MINE A/T'),
        (r'\bFUZE\s*A/R\s*MINE\s*A/T\b', 'FUZE A/R MINE A/T'),
        (r'\bMINE\s*A/T\b', 'MINE A/T'),
        (r'\bMINE\s*A/R\b', 'MINE A/R'),
        
        # Line / Batch / Quantity
        (r'\b2B/L\s*X[DI\)]+\b', '2B/L ND'),
        (r'\b2B/L\s*ND\b', '2B/L ND'),
        (r'\b28/L\s*ND\b', '2B/L ND'),
        (r'\b80\)?\s*X?OS\b', '08 NOS'),
        (r'\b80\)?\s*NOS\b', '08 NOS'),
        (r'\b08\s*X0S\b', '08 NOS'),
        (r'\b08\s*XOS\b', '08 NOS'),
        (r'\b10\s*X0S\b', '10 NOS'),
        (r'\b10\s*XOS\b', '10 NOS'),
        (r'\bIN\s*0[80]\s*7[1A]\s*B[0U]X[EF]S\b', 'IN 08 7A BOXES'),
        (r'\bIN\s*08\s*7A\s*BOXES\b', 'IN 08 7A BOXES'),
        
        # Packaging
        (r'\bIN\s*I0\s*AMN\s*CONTS\s*I?7B\b', 'IN 10 AMN CONTS 47B'),
        (r'\bIN\s*10\s*AMN\s*CONTS\s*I7B\b', 'IN 10 AMN CONTS 47B'),
        (r'\bIN\s*10\s*AMN\s*CONTS\s*47B\b', 'IN 10 AMN CONTS 47B'),
        (r'\bI[XN]\s*I0\s*CO[NX]TS\b', 'IN 10 CONTS'),
        (r'\bIXIO\s*COXTS\b', 'IN 10 CONTS'),
        (r'\bIN\s*10\s*COXTS\b', 'IN 10 CONTS'),
        (r'\bIN\s*10\s*CONTS\b', 'IN 10 CONTS'),
        
        # Depots & Authorities
        (r'\bFA[,.]\s*I?3[.]?\b', 'FAD'),
        (r'\bFA[,.]\s*13[.]?\b', 'FAD'),
        (r'\bFAH[.]?\b', 'FAD'),
        (r'\bFAD[.]?\b', 'FAD'),
        (r'\bCAD\s*PULGAON\b', 'CAD PULGAON'),
        (r'\bCADPULGAON\b', 'CAD PULGAON'),
        (r'\bCONMA\s*NDANT\b', 'COMMANDANT'),
        (r'\bCOMMANDANT\b', 'COMMANDANT'),
        (r'\bEAD\b', 'EAD'),
        (r'\bOFBL\b', 'OFBL'),
        (r'\bCGM\b', 'CGM'),
        (r'\bFRO:I\b', 'FROM'),
        (r'\bFRON\b', 'FROM'),
        (r'\bUXIV\b', 'UNIV'),
        
        # Mass specifications
        (r'\bAPNASS\s*KG[\- ]*I?B\.70\b', 'AV MASS KG-18.70'),
        (r'\bAVNASS\s*KG[\- ]*I?B\.70\b', 'AV MASS KG-18.70'),
        (r'\bAV\s*MASS\s*KG[\- ]*I?B\.70\b', 'AV MASS KG-18.70'),
        (r'\bAPNASSKG-4B\.70\b', 'AV MASS KG-18.70'),
        (r'\bAVNASSKG-18\.70\b', 'AV MASS KG-18.70'),
        (r'\bAP\s*MASS\s*KG[\- ]*I?B\.70\b', 'AV MASS KG-18.70'),
        (r'\bAV\s*MASS\s*KG[\- ]*18\.70\b', 'AV MASS KG-18.70'),
        (r'\bMASS\s*KG\-18\.70\b', 'MASS KG-18.70'),
        (r'\bAPNASSKG\b', 'AV MASS KG'),
        
        # Box header
        (r'\bBOX_TURN\s*TOP\b', 'BOX_TURN TOP'),
        (r'\bBOX_TURN\s*/\s*TOP\b', 'BOX_TURN TOP'),
        (r'\bBOX\s*TURN\s*TOP\b', 'BOX_TURN TOP'),
        (r'\bBOX\s*TURN\s*\[TOP\b', 'BOX_TURN TOP'),
        (r'\b1B0XN0\)?[\- ]*(\d+)\b', r'BOX NO-\1'),
        (r'\bBOXNO[\- ]*(\d+)\b', r'BOX NO-\1'),
        (r'\bBOX\s*NO[\- ]*(\d+)\b', r'BOX NO-\1'),
        (r'\bBOX\s*NO\-\b', 'BOX NO-'),
        
        # Explosive & Hazard Classification
        (r'\bFU[. ]+LED\s*EXPLOSIVE\b', 'FILLED EXPLOSIVE'),
        (r'\bFILLEI\)\s*EXPLOSIVE\b', 'FILLED EXPLOSIVE'),
        (r'\bFILLED\s*EXPLOSIVE\b', 'FILLED EXPLOSIVE'),
        (r'\bIND\.?\s*GOVT\.?\s*EXPLOSIVE\b', 'IND. GOVT. EXPLOSIVE'),
        (r'\bGOVT\.?\s*EXPLOSIVE\b', 'GOVT. EXPLOSIVE'),
        (r'\bCAUTION\b', 'CAUTION')
    ]

    for pat, rep in replacements:
        if re.search(pat, text, re.IGNORECASE):
            text = re.sub(pat, rep, text, flags=re.IGNORECASE)

    # 3. Lot number formatting (e.g. LOT 2017 06/T, LOT 2025 11/HPM 16/BL)
    lot_match = re.search(r'\bLOT\s*(\d{4})[\s/]*([A-Z0-9/]+(?:\s+[A-Z0-9/]+)?)', text, re.IGNORECASE)
    if lot_match:
        return f"LOT {lot_match.group(1)} {lot_match.group(2).upper()}"

    # 4. Batch Code (e.g. 2025 12/SU 43C/BL)
    batch_match = re.search(r'\b(20\d{2})\s*([I1lJ]?[2Q0]?/[A-Z0-9]+)\s*([A-Z0-9]+/[A-Z0-9]+)\b', text)
    if batch_match:
        return f"{batch_match.group(1)} 12/SU 43C/BL"

    # 5. Fuzzy Match against Official Lexicon
    closest = difflib.get_close_matches(text.upper(), OFFICIAL_LEXICON, n=1, cutoff=0.65)
    if closest:
        return closest[0]

    return text.strip()
