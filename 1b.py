import os
import argparse
import json
import glob
from datetime import datetime
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import time
import logging
import nltk
from nltk.tokenize import sent_tokenize
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict

# Ensure nltk punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# --- Helper Functions ---
NOISE_WORDS = [
    'copyright', 'references', 'table of contents', 'index', 'acknowledgments',
    'disclaimer', 'appendix', 'about the author', 'preface', 'foreword', 'glossary'
]

def is_noise_section(section):
    text = section['text'].lower()
    return any(word in text for word in NOISE_WORDS)

def improved_section_heading(line):
    # Improved heading detection: numbered, all-caps, 'CHAPTER', 'Section', etc.
    if re.match(r'^(\d+\.?\d*)?\s*([A-Z][A-Za-z0-9 ,\-]+)$', line):
        return True
    if re.match(r'^(CHAPTER|SECTION)\s+\d+', line, re.IGNORECASE):
        return True
    if line.isupper() and len(line) > 3:
        return True
    return False

def extract_sections_from_pdf(pdf_path):
    sections = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ''
                lines = text.split('\n')
                current_section = {'page_number': i+1, 'section_title': 'Introduction', 'text': ''}
                for line in lines:
                    if improved_section_heading(line.strip()):
                        if current_section['text'].strip():
                            sections.append(current_section)
                        current_section = {
                            'page_number': i+1,
                            'section_title': line.strip(),
                            'text': ''
                        }
                    else:
                        current_section['text'] += line + '\n'
                if current_section['text'].strip():
                    sections.append(current_section)
        return sections, None
    except Exception as e:
        return [], str(e)

def get_pdf_title(pdf_path):
    return os.path.splitext(os.path.basename(pdf_path))[0]

def embed_texts(model, texts, batch_size=32):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)

def filter_sections(sections, min_length=30):
    filtered = []
    skipped = []
    for s in sections:
        if len(s['text']) > min_length and not is_noise_section(s):
            filtered.append(s)
        else:
            skipped.append(s)
    return filtered, skipped

def deduplicate_sections(sections):
    seen = set()
    deduped = []
    for s in sections:
        key = s['text'].strip()
        if key and key not in seen:
            deduped.append(s)
            seen.add(key)
    return deduped

def rank_sections(query_emb, section_embs):
    sims = cosine_similarity(query_emb, section_embs)[0]
    ranked_indices = np.argsort(sims)[::-1]
    return ranked_indices, sims

def build_output(doc_metadata, all_sections, ranked_indices, sims, top_n=10):
    extracted_sections = []
    subsection_analysis = []
    for rank, idx in enumerate(ranked_indices[:top_n]):
        sec = all_sections[idx]
        extracted_sections.append({
            'document': sec['document'],
            'page_number': sec['page_number'],
            'section_title': sec['section_title'],
            'importance_rank': rank+1
        })
        sentences = sent_tokenize(sec['text'])
        refined_text = ' '.join(sentences[:3]).strip()
        subsection_analysis.append({
            'document': sec['document'],
            'refined_text': refined_text,
            'page_number': sec['page_number']
        })
    return extracted_sections, subsection_analysis

def main():
    parser = argparse.ArgumentParser(description='Persona-Driven Document Intelligence')
    parser.add_argument('--pdf_dir', type=str, required=True, help='Directory containing PDF files')
    parser.add_argument('--persona', type=str, required=True, help='Persona description')
    parser.add_argument('--job', type=str, required=True, help='Job to be done')
    parser.add_argument('--output', type=str, default='output.json', help='Output JSON file')
    parser.add_argument('--min_section_length', type=int, default=30, help='Minimum section length to consider')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top sections to output')
    parser.add_argument('--skipped_sections_output', type=str, default=None, help='Optional: Write skipped sections to this file instead of including in main output')
    args = parser.parse_args()

    start_time = time.time()
    pdf_files = glob.glob(os.path.join(args.pdf_dir, '*.pdf'))
    if not pdf_files:
        logging.error('No PDF files found in directory.')
        return
    logging.info(f'Found {len(pdf_files)} PDF files.')

    # Load model (CPU only)
    logging.info('Loading embedding model...')
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    # Prepare query embedding
    query = args.persona + ' ' + args.job
    query_emb = model.encode([query])

    all_sections = []
    doc_metadata = []
    skipped_sections = []
    failed_files = defaultdict(str)

    # Parallel PDF section extraction
    with ThreadPoolExecutor() as executor:
        future_to_pdf = {executor.submit(extract_sections_from_pdf, pdf_path): pdf_path for pdf_path in pdf_files}
        for future in tqdm(as_completed(future_to_pdf), total=len(pdf_files), desc='Extracting PDFs'):
            pdf_path = future_to_pdf[future]
            title = get_pdf_title(pdf_path)
            doc_metadata.append({'filename': os.path.basename(pdf_path), 'title': title})
            try:
                sections, error = future.result()
                if error:
                    failed_files[os.path.basename(pdf_path)] = error
                    continue
                for section in sections:
                    section['document'] = os.path.basename(pdf_path)
                    section['title'] = title
                filtered, skipped = filter_sections(sections, min_length=args.min_section_length)
                all_sections.extend(filtered)
                skipped_sections.extend(skipped)
            except Exception as e:
                failed_files[os.path.basename(pdf_path)] = str(e)

    if not all_sections:
        logging.warning('No valid sections found after filtering. Outputting skipped sections for review.')
        output = {
            'metadata': {
                'input_documents': doc_metadata,
                'persona': args.persona,
                'job_to_be_done': args.job,
                'processing_timestamp': datetime.utcnow().isoformat() + 'Z',
                'flag': 'NO_VALID_SECTIONS_FOUND',
                'failed_files': dict(failed_files)
            },
            'extracted_sections': [],
            'subsection_analysis': [],
            'skipped_sections': skipped_sections
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        elapsed = time.time() - start_time
        logging.info(f'Output written to {args.output} in {elapsed:.2f} seconds.')
        return

    # Deduplicate sections
    before_dedup = len(all_sections)
    all_sections = deduplicate_sections(all_sections)
    after_dedup = len(all_sections)
    logging.info(f'{before_dedup} sections after filtering, {after_dedup} after deduplication. {len(skipped_sections)} sections skipped. {len(failed_files)} files failed.')

    # Concatenate section title and text for embedding, weight title more
    section_texts = [(s['section_title'] + ' ') * 2 + s['text'] for s in all_sections]  # Repeat title for weighting

    # For very large document sets, process in batches and keep only top-N
    batch_size = 256
    top_n = args.top_n
    section_embs = []
    batch_indices = []
    for i in tqdm(range(0, len(section_texts), batch_size), desc='Embedding sections'):
        batch = section_texts[i:i+batch_size]
        embs = embed_texts(model, batch, batch_size=32)
        section_embs.append(embs)
        batch_indices.extend(range(i, min(i+batch_size, len(section_texts))))
    section_embs = np.vstack(section_embs)

    # Compute relevance and rank
    ranked_indices, sims = rank_sections(query_emb, section_embs)
    # Keep only top-N in memory
    ranked_indices = ranked_indices[:top_n]

    # Build output
    extracted_sections, subsection_analysis = build_output(doc_metadata, all_sections, ranked_indices, sims, top_n=top_n)

    output = {
        'metadata': {
            'input_documents': doc_metadata,
            'persona': args.persona,
            'job_to_be_done': args.job,
            'processing_timestamp': datetime.utcnow().isoformat() + 'Z',
            'failed_files': dict(failed_files)
        },
        'extracted_sections': extracted_sections,
        'subsection_analysis': subsection_analysis
    }
    if args.skipped_sections_output:
        with open(args.skipped_sections_output, 'w', encoding='utf-8') as f:
            json.dump(skipped_sections, f, indent=2)
    else:
        output['skipped_sections'] = skipped_sections

    elapsed = time.time() - start_time
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    logging.info(f'Output written to {args.output} in {elapsed:.2f} seconds.')

if __name__ == '__main__':
    main() 