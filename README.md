# Doc2SAR: Structure-Activity Relationship Extraction from Scientific Documents

This repository contains the implementation for Doc2SAR, a framework for extracting structure-activity relationships (SARs) from scientific documents. The codebase provides tools for processing PDF documents and evaluating extraction performance.

## 🚀 Features

- **PDF Processing**: Extract SAR data directly from PDF documents using multimodal language models
- **Comprehensive Evaluation**: Compare extraction results against ground truth annotations
- **Multi-table Support**: Handle documents with multiple activity tables
- **Flexible Output**: Generate structured CSV files for downstream analysis

## 📋 Requirements

```bash
pip install openai
pip install PyMuPDF
pip install pandas
pip install tqdm
```

## 🔧 Setup

1. **Configure Azure OpenAI API**:
   
   Edit the `API_CONFIG` in the extraction script:
   ```python
   API_CONFIG = {
        "api_base": "<your_azure_openai_endpoint>",
        "api_key": "<your_azure_openai_key>",
        "engine": "gpt-4o",
        "api_version": "2024-02-15-preview"
    }
   ```

2. **Prepare Directory Structure**:
   ```
   project/
   ├── end2end.py
   ├── eval.py
   ├── doi.txt                    # List of document identifiers
   ├── utils/
   │   ├── prompts.py
   │   ├── metrics.py
   │   └── reformat.py
   ├── data/
   │   ├── input_pdfs/           # Download PDFs here
   │   └── ground_truth/
   └── output/
       └── predictions/
   ```

## 📥 Data Acquisition

Due to file size constraints, PDF documents are not included in this repository. Instead, we provide `doi.txt` containing the document identifiers for all files in the DocSAR-200 benchmark.

### Download Documents

1. **Literature Articles**: Use the DOI identifiers to download from publishers
   ```
   # Example DOIs from doi.txt:
   10.1021/acsmedchemlett.4c00408
   10.1016/j.ejmech.2015.11.043
   10.1016/j.bmcl.2017.03.030
   ```

2. **Patent Documents**: Use patent numbers to download from patent offices
   ```
   # Example patent numbers from doi.txt:
   US20160355824A1
   CN102482273B
   EP3408265B1
   ```

### Suggested Download Sources
- **Literature**: Publisher websites, institutional access, or academic databases
- **Patents**: 
  - USPTO (patents.uspto.gov) for US patents
  - Google Patents (patents.google.com) for global access
  - EPO (worldwide.espacenet.com) for European patents
  - CNIPA for Chinese patents

Place downloaded PDF files in the `data/input_pdfs/` directory with filenames matching the identifiers in `doi.txt`.

## 🎯 Usage

### Extract SAR Data from PDFs

Process a directory of PDF files to extract structure-activity relationships:

```bash
python end2end.py --dirs ./data/input_pdfs --pred ./output/predictions
```

**Parameters:**
- `--dirs`: Input directory containing PDF files
- `--pred`: Output directory for extracted CSV files

### Evaluate Extraction Results

Compare predicted extractions against ground truth annotations:

```bash
python eval.py --pred ./output/predictions --test ./data/ground_truth
```

**Parameters:**
- `--pred`: Directory containing predicted CSV files
- `--test`: Directory containing ground truth CSV files

## 📊 Example Workflow

```bash
# Step 1: Download PDFs using identifiers from doi.txt
# (Manual step - download PDFs to ./data/input_pdfs/)

# Step 2: Extract SAR data from PDF documents
python end2end.py --dirs ./data/input_pdfs --pred ./results/extracted

# Step 3: Evaluate extraction performance
python eval.py --pred ./results/extracted --test ./data/ground_truth

# Output example:
# 0.8078  # Average Table Recall
# 200     # Number of processed files
```

## 📁 File Structure

### Input Format
- **PDF Files**: Scientific documents containing molecular structures and activity tables
- **Document List**: `doi.txt` contains all document identifiers for reproducibility
- **Ground Truth**: CSV files with annotated SAR data

### Output Format
The extraction process generates CSV files with the following structure:
```csv
SMILES,Compound_ID,IC50,Ki,Other_Activity_Metrics
CCO,1a,2.3 nM,1.5 nM,>90% inhibition
CCC,1b,15.2 nM,8.7 nM,85% inhibition
```
