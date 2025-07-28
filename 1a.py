import os
import json
import fitz  # PyMuPDF
import logging
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import traceback
import numpy as np
from sklearn.cluster import KMeans
from scipy import stats
import string

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def extract_title(page: fitz.Page) -> str:
    """
    Enhanced title extraction with better heuristics.
    """
    blocks = page.get_text("dict")['blocks']
    candidates = []
    
    page_height = page.rect.height
    top_quarter = page_height * 0.25
    
    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if len(text) > 5 and text.isprintable():
                    y_pos = span["bbox"][1]
                    # Prioritize text in the top quarter of the page
                    position_score = 1.0 if y_pos < top_quarter else 0.5
                    # Check if it looks like a title (not all caps, reasonable length)
                    title_score = 1.0
                    if text.isupper() and len(text) > 50:
                        title_score = 0.3
                    if len(text) > 200:
                        title_score = 0.2
                    
                    candidates.append((
                        span["size"], 
                        -y_pos,  # Negative for sorting (higher on page = better)
                        text,
                        position_score * title_score
                    ))
    
    if not candidates:
        return "Untitled Document"
    
    # Sort by combined score: font size, position, and title characteristics
    candidates.sort(key=lambda x: (x[0] * x[3], x[1]), reverse=True)
    return candidates[0][2]

def analyze_text_features(span: Dict) -> Dict[str, Any]:
    """
    Analyze comprehensive text features for better heading detection.
    """
    text = span["text"].strip()
    font = span.get("font", "")
    
    features = {
        'is_bold': any(keyword in font.lower() for keyword in ['bold', 'black', 'heavy']),
        'is_italic': any(keyword in font.lower() for keyword in ['italic', 'oblique']),
        'is_caps': text.isupper(),
        'is_title_case': text.istitle(),
        'has_numbers': bool(re.search(r'\d', text)),
        'starts_with_number': bool(re.match(r'^\d+\.?\s', text)),
        'word_count': len(text.split()),
        'char_count': len(text),
        'ends_with_punct': text.endswith(('.', ':', '?', '!')),
        'contains_common_heading_words': any(word in text.lower() for word in [
            'chapter', 'section', 'introduction', 'conclusion', 'summary', 
            'overview', 'background', 'methodology', 'results', 'discussion'
        ])
    }
    
    return features

def enhanced_font_clustering(font_data: List[Tuple[float, Dict]], n_clusters: int = 4) -> Dict[float, str]:
    """
    Enhanced font clustering using multiple features and statistical analysis.
    """
    if len(font_data) < 10:
        # Fallback for small datasets
        font_sizes = [fd[0] for fd in font_data]
        unique_sizes = sorted(set(font_sizes), reverse=True)
        levels = ['H1', 'H2', 'H3', 'body']
        mapping = {}
        for i, size in enumerate(unique_sizes[:4]):
            mapping[size] = levels[i] if i < 3 else 'body'
        return mapping
    
    # Extract features for clustering
    font_sizes = np.array([fd[0] for fd in font_data])
    bold_scores = np.array([1.0 if fd[1]['is_bold'] else 0.0 for fd in font_data])
    caps_scores = np.array([1.0 if fd[1]['is_caps'] else 0.0 for fd in font_data])
    
    # Statistical analysis to find natural breaks
    size_percentiles = np.percentile(font_sizes, [75, 90, 95])
    
    # Create feature matrix for clustering
    features = np.column_stack([
        font_sizes,
        bold_scores * np.max(font_sizes) * 0.1,  # Weight bold appropriately
        caps_scores * np.max(font_sizes) * 0.05   # Weight caps less
    ])
    
    # Determine optimal number of clusters
    unique_sizes = len(set(font_sizes))
    n_clusters = min(n_clusters, unique_sizes, 4)
    
    if n_clusters < 2:
        return {font_sizes[0]: 'H1'}
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(features)
    
    # Map clusters to heading levels based on average font size
    cluster_stats = defaultdict(list)
    for i, (size, feats) in enumerate(font_data):
        cluster_stats[kmeans.labels_[i]].append(size)
    
    # Sort clusters by average font size (descending)
    cluster_avg_sizes = [(cluster, np.mean(sizes)) for cluster, sizes in cluster_stats.items()]
    cluster_avg_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Assign levels
    levels = ['H1', 'H2', 'H3', 'body']
    mapping = {}
    
    for rank, (cluster_id, avg_size) in enumerate(cluster_avg_sizes):
        level = levels[min(rank, len(levels) - 1)]
        for i, (size, feats) in enumerate(font_data):
            if kmeans.labels_[i] == cluster_id:
                mapping[size] = level
    
    return mapping

def is_likely_heading(text: str, features: Dict, context: Dict) -> Tuple[bool, float]:
    """
    Determine if text is likely a heading based on multiple criteria.
    Returns (is_heading, confidence_score)
    """
    score = 0.0
    max_score = 10.0
    
    # Length criteria
    if 3 <= len(text) <= 100:
        score += 2.0
    elif len(text) > 200:
        score -= 3.0
    
    # Formatting criteria
    if features['is_bold']:
        score += 2.0
    if features['is_caps'] and len(text) < 50:
        score += 1.5
    if features['is_title_case']:
        score += 1.0
    
    # Content criteria
    if features['starts_with_number']:
        score += 1.5
    if features['contains_common_heading_words']:
        score += 2.0
    if not features['ends_with_punct'] or text.endswith(':'):
        score += 1.0
    
    # Context criteria
    if context.get('is_isolated', False):  # Standalone line
        score += 1.0
    if context.get('has_content_after', False):  # Has content following
        score += 1.0
    
    # Special handling for simple documents (flyers, invitations, etc.)
    if context.get('document_type') == 'simple':
        # For simple documents, be more lenient
        if features['is_caps'] or features['is_bold']:
            score += 1.0
        if len(text.split()) <= 3:  # Short phrases can be headings
            score += 1.0
        # Common flyer/invitation patterns
        if any(word in text.lower() for word in ['address:', 'rsvp:', 'contact:', 'phone:', 'email:', 'date:', 'time:', 'location:']):
            score += 2.0
    
    # Negative indicators
    if len(text.split()) > 15:  # Too many words
        score -= 2.0
    if features['char_count'] > 150:  # Too long
        score -= 1.5
    if text.count('.') > 2:  # Multiple sentences
        score -= 2.0
    
    confidence = max(0.0, min(1.0, score / max_score))
    is_heading = confidence > 0.5
    
    return is_heading, confidence

def detect_document_type(all_text_elements: List[Dict]) -> str:
    """
    Detect the type of document to adjust extraction strategy.
    """
    if not all_text_elements:
        return 'simple'
    
    total_elements = len(all_text_elements)
    
    # Count various indicators
    heading_indicators = 0
    caps_count = 0
    structured_indicators = 0
    
    for element in all_text_elements:
        text = element['text'].lower()
        features = element['features']
        
        # Count traditional heading indicators
        if features['starts_with_number'] or features['contains_common_heading_words']:
            heading_indicators += 1
        
        if features['is_caps']:
            caps_count += 1
        
        # Count structured document indicators
        if any(word in text for word in ['chapter', 'section', 'table of contents', 'abstract', 'introduction', 'conclusion']):
            structured_indicators += 1
    
    # Decision logic
    if total_elements < 20:  # Very short document
        return 'simple'
    elif structured_indicators > 2 or heading_indicators > total_elements * 0.1:
        return 'structured'
    elif caps_count > total_elements * 0.3:  # Lots of caps text
        return 'simple'
    else:
        return 'mixed'
def extract_headings_enhanced(doc: fitz.Document, median_height: float) -> List[Dict[str, Any]]:
    """
    Enhanced heading extraction with improved accuracy for different document types.
    """
    # First try TOC
    toc = doc.get_toc(simple=True)
    if toc:
        headings = []
        for level, title, page_num in toc:
            if level <= 3 and title.strip():
                headings.append({
                    "level": f"H{level}",
                    "text": title.strip(),
                    "page": page_num,
                    "confidence": 1.0,
                    "source": "toc"
                })
        if headings:
            return headings
    
    # Enhanced text analysis
    all_text_data = []
    page_structures = []
    all_text_elements = []
    
    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")["blocks"]
        page_text_elements = []
        
        for block_idx, block in enumerate(blocks):
            if block["type"] != 0:
                continue
                
            for line_idx, line in enumerate(block["lines"]):
                line_texts = []
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text and len(text) >= 2:
                        features = analyze_text_features(span)
                        font_size = round(span["size"], 1)
                        
                        element = {
                            'text': text,
                            'font_size': font_size,
                            'features': features,
                            'bbox': span["bbox"],
                            'page': page_num,
                            'block_idx': block_idx,
                            'line_idx': line_idx
                        }
                        
                        line_texts.append(element)
                        all_text_data.append((font_size, features))
                        all_text_elements.append(element)
                
                if line_texts:
                    page_text_elements.append(line_texts)
        
        page_structures.append(page_text_elements)
    
    if not all_text_data:
        return []
    
    # Detect document type
    document_type = detect_document_type(all_text_elements)
    
    # Enhanced font clustering
    font_mapping = enhanced_font_clustering(all_text_data)
    
    # Adjust confidence threshold based on document type
    confidence_threshold = {
        'structured': 0.6,
        'simple': 0.4,  # Lower threshold for simple documents
        'mixed': 0.5
    }.get(document_type, 0.5)
    
    # Extract headings with context analysis
    headings = []
    
    for page_idx, page_elements in enumerate(page_structures):
        for line_idx, line_elements in enumerate(page_elements):
            if len(line_elements) == 1:  # Single span per line (good for headings)
                element = line_elements[0]
                text = element['text']
                font_size = element['font_size']
                features = element['features']
                
                # For simple documents, be more flexible with font requirements
                if document_type == 'simple':
                    # Allow any text that looks like a heading, regardless of font mapping
                    font_level = font_mapping.get(font_size, 'H3')  # Default to H3 instead of body
                else:
                    # Skip if font size not in heading categories for structured docs
                    font_level = font_mapping.get(font_size, 'body')
                    if font_level == 'body' and document_type == 'structured':
                        continue
                
                # Context analysis
                context = {
                    'is_isolated': True,
                    'has_content_after': line_idx < len(page_elements) - 1,
                    'position_on_page': line_idx / max(1, len(page_elements)),
                    'document_type': document_type
                }
                
                # Check if it's likely a heading
                is_heading, confidence = is_likely_heading(text, features, context)
                
                if is_heading and confidence > confidence_threshold:
                    headings.append({
                        "level": font_level,
                        "text": text,
                        "page": element['page'],
                        "confidence": confidence,
                        "source": "analysis",
                        "document_type": document_type
                    })
    
    # Post-processing: remove duplicates and filter by confidence
    seen_texts = set()
    filtered_headings = []
    
    for heading in sorted(headings, key=lambda x: (-x['confidence'], x['page'])):
        text_lower = heading['text'].lower().strip()
        if text_lower not in seen_texts and heading['confidence'] > confidence_threshold:
            seen_texts.add(text_lower)
            filtered_headings.append(heading)
    
    # Sort by page and confidence
    filtered_headings.sort(key=lambda x: (x['page'], -x['confidence']))
    
    return filtered_headings

def validate_heading_hierarchy(headings: List[Dict]) -> List[Dict]:
    """
    Validate and fix heading hierarchy issues.
    """
    if not headings:
        return headings
    
    validated = []
    last_level = 0
    
    for heading in headings:
        level_num = int(heading['level'][1])  # Extract number from H1, H2, etc.
        
        # Ensure proper hierarchy (no skipping levels dramatically)
        if level_num > last_level + 2:
            # Adjust level to maintain hierarchy
            level_num = min(level_num, last_level + 1)
            heading['level'] = f'H{level_num}'
            heading['adjusted'] = True
        
        validated.append(heading)
        last_level = level_num
    
    return validated

def process_pdf(input_path: str) -> Dict[str, Any]:
    """
    Enhanced PDF processing with improved accuracy.
    """
    doc = fitz.open(input_path)
    
    # Extract title with enhanced method
    title = extract_title(doc[0])
    
    # Calculate page statistics
    page_heights = [page.rect.height for page in doc]
    median_height = float(np.median(page_heights)) if page_heights else 842.0
    
    # Extract headings with enhanced method
    outline = extract_headings_enhanced(doc, median_height)
    
    # Validate hierarchy
    outline = validate_heading_hierarchy(outline)
    
    doc.close()
    
    # Add metadata
    metadata = {
        'total_headings': len(outline),
        'avg_confidence': np.mean([h.get('confidence', 0) for h in outline]) if outline else 0,
        'page_count': len(page_heights),
        'extraction_method': 'enhanced_analysis'
    }
    
    return {
        "title": title,
        "outline": outline,
        "metadata": metadata
    }

def process_pdf_safe(input_path: str) -> Tuple[str, Dict[str, Any], str]:
    """
    Safe wrapper with enhanced error handling.
    """
    try:
        result = process_pdf(input_path)
        return (os.path.basename(input_path), result, "")
    except Exception as e:
        tb = traceback.format_exc()
        logging.error(f"Error processing {input_path}: {e}")
        return (os.path.basename(input_path), {}, f"{e}\n{tb}")

def process_all_pdfs(input_dir: str, output_dir: str, combined: bool = False, 
                    max_workers: int = 4, pretty: bool = True) -> None:
    """
    Enhanced batch processing with better progress tracking.
    """
    ensure_dir(output_dir)
    
    pdf_files = [
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.lower().endswith(".pdf")
    ]
    
    if not pdf_files:
        logging.warning(f"No PDF files found in {input_dir}")
        return
    
    logging.info(f"Found {len(pdf_files)} PDF files to process")
    
    results = []
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_pdf_safe, pdf_path) for pdf_path in pdf_files]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing PDFs'):
            filename, result, error = future.result()
            
            if error:
                logging.error(f"Failed to process {filename}: {error}")
                failed += 1
            else:
                logging.info(f"Successfully processed {filename} - "
                           f"Found {result.get('metadata', {}).get('total_headings', 0)} headings")
                successful += 1
            
            results.append((filename, result))
    
    logging.info(f"Processing complete: {successful} successful, {failed} failed")
    
    # Output results
    if combined:
        combined_dict = {filename: result for filename, result in results if result}
        out_path = os.path.join(output_dir, "combined_output.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(combined_dict, f, ensure_ascii=False, indent=2 if pretty else None)
        logging.info(f"Combined output written to {out_path}")
    else:
        for filename, result in results:
            if result:
                out_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2 if pretty else None)

def main():
    parser = argparse.ArgumentParser(description="Enhanced PDF Outline Extractor with Improved Accuracy")
    # Support both positional and named arguments for backward compatibility
    parser.add_argument("input_dir", nargs="?", default=None, help="Input directory (positional)")
    parser.add_argument("output_dir", nargs="?", default=None, help="Output directory (positional)")
    parser.add_argument("--input", default="./input", help="Input directory containing PDF files")
    parser.add_argument("--output", default="./output", help="Output directory for JSON files")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--log", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--combined", action="store_true", help="Output single combined JSON file")
    parser.add_argument("--compact", action="store_true", help="Compact JSON output")
    parser.add_argument("--strict-bold", action="store_true", help="Legacy option for compatibility")
    
    args = parser.parse_args()
    
    # Handle backward compatibility - use positional args if provided
    input_path = args.input_dir if args.input_dir else args.input
    output_path = args.output_dir if args.output_dir else args.output
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if not os.path.exists(input_path):
        logging.error(f"Input directory does not exist: {input_path}")
        return
    
    process_all_pdfs(
        input_path, 
        output_path, 
        combined=args.combined, 
        max_workers=args.workers, 
        pretty=not args.compact
    )

if __name__ == "__main__":
    main()