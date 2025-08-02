END2END_PROMPT = '''
You are a pharmaceutical chemistry expert. I've provided PDF pages containing ONLY tables with activity data and molecular structures. Your task is to output CSV-formatted activity tables with SMILES notation. 

**STRICT RULES:**
1. Process ONLY tables containing these headers (case-insensitive): 
   ['EC50', 'IC50', 'Ki', 'Kd', 'pKd', 'TD50', 'Ti']
2. For each compound identifier in activity tables:
   - Match to molecular structures and extract SMILES
   - Add 'SMILES' column as SECOND column
   - If no matching SMILES found, leave cell EMPTY
3. Output format per table:
   [CSV HEADER]
   [CSV DATA ROWS]
4. For multiple tables, use this exact delimiter on a separate line:
   ---NEXT TABLE---
5. **OUTPUT ONLY CSV DATA - NO EXPLANATIONS, NO ADDITIONAL TEXT**

Output format example:
```csv
Compound id,SMILES,Target A IC50 nM,Target B Kd µM
7b,CN1C=NC2=C1C(=O)N(C(=O)N2C)C,0.5 ± 0.1,>10
8a,C1=CC=C(C=C1)C=O,2.3,12.5
---NEXT TABLE---
Compound id,SMILES,Target C EC50 nM
9c,CCOC(=O)C1=CC=CC=C1,15.8
'''