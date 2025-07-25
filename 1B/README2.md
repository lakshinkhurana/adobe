# Persona-Driven Document Intelligence

## Overview
This tool extracts and prioritizes the most relevant sections from a collection of PDFs based on a given persona and job-to-be-done. It is designed to run fully offline, on CPU, and with a model size under 1GB.

## Features
- PDF text extraction
- Persona and job-to-be-done relevance analysis
- Outputs structured JSON with metadata and extracted sections
- Fast, lightweight, and offline

## Installation
1. Clone this repository or download the code.
2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the required sentence-transformers model in advance (see below).

## Usage
1. Place your PDF files in a folder (e.g., `./pdfs`).
2. Run the main script:
   ```bash
   python main.py --pdf_dir ./pdfs --persona "PhD Researcher in Computational Biology" --job "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
   ```
3. The output will be saved as `output.json` in the project directory.

## Offline Model Download
Before running offline, download the model with internet access:
```python
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2')
```
This will cache the model locally. Afterward, you can run the script without internet access.

## Constraints
- CPU only
- Model size < 1GB
- No internet access at runtime
- Processing time < 60s for 3-5 documents

## License
MIT 