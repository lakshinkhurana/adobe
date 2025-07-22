import os
import json
import fitz  # PyMuPDF
import logging
from collections import Counter
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import traceback
import numpy as np
from sklearn.cluster import KMeans

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def extract_title(page: fitz.Page) -> str:
    """
    Extracts the document title from the first page using font size and position.
    """
    blocks = page.get_text("dict")['blocks']
    candidates = []
    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if len(text) > 5 and text.isprintable():
                    candidates.append((span["size"], span["bbox"][1], text))
    if not candidates:
        return "Untitled Document"
    return sorted(candidates, key=lambda x: (-x[0], x[1]))[0][2]

def cluster_font_sizes_kmeans(font_sizes: List[float], n_clusters: int = 3) -> Dict[float, str]:
    """
    Cluster unique font sizes using k-means and assign to heading levels.
    Only unique font sizes are clustered, then mapped back to all spans.
    """
    unique_font_sizes = sorted(set(font_sizes))
    if len(unique_font_sizes) <= n_clusters:
        # Fallback to most common sizes
        common_sizes = [fs[0] for fs in Counter(font_sizes).most_common()]
        levels = ['H1', 'H2', 'H3']
        mapping = {}
        for i, size in enumerate(common_sizes[:3]):
            mapping[size] = levels[i]
        return mapping
    arr = np.array(unique_font_sizes).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(arr)
    centers = sorted([c[0] for c in kmeans.cluster_centers_], reverse=True)
    levels = ['H1', 'H2', 'H3']
    mapping = {}
    for size, label in zip(unique_font_sizes, kmeans.labels_):
        mapping[size] = levels[centers.index(kmeans.cluster_centers_[label][0])]
    return mapping

def extract_headings(doc: fitz.Document, median_height: float, strict_bold: bool = False) -> List[Dict[str, Any]]:
    """
    Extracts headings (H1/H2/H3) with page numbers from all pages.
    Uses font size clustering, boldness (toggleable), and position. Falls back to PDF outline if available.
    """
    toc = doc.get_toc(simple=True)
    if toc:
        headings = []
        for level, title, page_num in toc:
            if level > 3:
                continue
            headings.append({
                "level": f"H{level}",
                "text": title.strip(),
                "page": page_num
            })
        if headings:
            return headings
    all_spans = []
    font_sizes = []
    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text or len(text) < 3:
                        continue
                    font_size = round(span["size"], 1)
                    font = span.get("font", "")
                    is_bold = "Bold" in font or "bold" in font
                    y0 = span["bbox"][1]
                    all_spans.append((text, font_size, is_bold, page_num, y0))
                    font_sizes.append(font_size)
    font_mapping = cluster_font_sizes_kmeans(font_sizes)
    headings = []
    top_margin = median_height * 0.05
    bottom_margin = median_height * 0.95
    for text, font_size, is_bold, page_num, y0 in all_spans:
        if font_size in font_mapping:
            if not (y0 < top_margin or y0 > bottom_margin):
                level = font_mapping[font_size]
                if level != 'H1' and strict_bold and not is_bold:
                    continue
                headings.append({
                    "level": level,
                    "text": text,
                    "page": page_num
                })
    return headings

def process_pdf(input_path: str, strict_bold: bool = False) -> Dict[str, Any]:
    """
    Processes a single PDF and returns the outline dict.
    Uses median page height for margin calculations.
    """
    doc = fitz.open(input_path)
    title = extract_title(doc[0])
    page_heights = [page.rect.height for page in doc]
    median_height = float(np.median(page_heights)) if page_heights else 842.0
    outline = extract_headings(doc, median_height, strict_bold)
    doc.close()
    return {"title": title, "outline": outline}

def process_pdf_safe(input_path: str, strict_bold: bool = False) -> Tuple[str, Dict[str, Any], str]:
    """
    Wrapper for process_pdf that catches and logs errors with tracebacks.
    Returns (filename, result_dict, error_message)
    """
    try:
        result = process_pdf(input_path, strict_bold)
        return (os.path.basename(input_path), result, "")
    except Exception as e:
        tb = traceback.format_exc()
        return (os.path.basename(input_path), {}, f"{e}\n{tb}")

def process_all_pdfs(input_dir: str, output_dir: str, combined: bool = False, max_workers: int = 4, pretty: bool = True, strict_bold: bool = False) -> None:
    """
    Processes all PDFs in the input directory in parallel.
    If combined is True, writes a single JSON file with all results.
    Otherwise, writes one JSON per PDF.
    Docker compatibility: Ensure entrypoint CMD matches CLI usage.
    """
    ensure_dir(output_dir)
    pdf_files = [
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.lower().endswith(".pdf")
    ]
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_pdf_safe, in_path, strict_bold) for in_path in pdf_files]
        for f in tqdm(as_completed(futures), total=len(futures), desc='Processing PDFs'):
            filename, result, error = f.result()
            if error:
                logging.error(f"Error processing {filename}: {error}")
            else:
                logging.info(f"Processed {filename}")
            results.append((filename, result))
    if combined:
        combined_dict = {filename: result for filename, result in results if result}
        out_path = os.path.join(output_dir, "combined_output.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(combined_dict, f, ensure_ascii=False, indent=2 if pretty else None)
        logging.info(f"Wrote combined output to {out_path}")
    else:
        for filename, result in results:
            if result:
                out_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2 if pretty else None)


def main():
    parser = argparse.ArgumentParser(description="Batch PDF Outline Extractor (Advanced Optimized)")
    parser.add_argument("--input", default="./input", help="Input directory")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--log", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--combined", action="store_true", help="Output a single combined JSON file")
    parser.add_argument("--compact", action="store_true", help="Compact JSON output (no pretty indent)")
    parser.add_argument("--strict-bold", action="store_true", help="Require bold for H2/H3 headings")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), None), format='%(levelname)s: %(message)s')
    if not os.path.exists(args.input):
        logging.error(f"Input directory does not exist: {args.input}")
        return
    process_all_pdfs(args.input, args.output, combined=args.combined, max_workers=args.workers, pretty=not args.compact, strict_bold=args.strict_bold)

if _name_ == "_main_":
    main()