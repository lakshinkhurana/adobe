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
    Enhanced title extraction with better heuristics for simple documents.
    """
    blocks = page.get_text("dict")['blocks']
    candidates = []
    
    page_height = page.rect.height
    top_third = page_height * 0.33
    
    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if len(text) > 2 and text.isprintable():
                    y_pos = span["bbox"][1]
                    font_size = span["size"]
                    
                    # For simple documents, prioritize larger text in upper portion
                    position_score = 2.0 if y_pos < top_third else 1.0
                    
                    # Special handling for business names, event titles
                    title_indicators = ['jump', 'party', 'event', 'celebration', 'invitation']
                    title_score = 1.0
                    
                    # Boost score for potential business names or event titles
                    if any(indicator in text.lower() for indicator in title_indicators):
                        title_score = 2.0
                    
                    # Penalize very common address/contact patterns
                    if any(pattern in text.lower() for pattern in ['phone:', 'address:', 'email:', 'www.', 'http']):
                        title_score = 0.3
                    
                    # Penalize very long lines that look like addresses
                    if len(text) > 100 and any(word in text.upper() for word in ['STREET', 'AVENUE', 'ROAD', 'BLVD']):
                        title_score = 0.2
                    
                    candidates.append((
                        font_size, 
                        -y_pos,  # Negative for sorting (higher on page = better)
                        text,
                        position_score * title_score
                    ))
    
    if not candidates:
        return "Untitled Document"
    
    # Sort by combined score: font size, position, and title characteristics
    candidates.sort(key=lambda x: (x[0] * x[3], x[1]), reverse=True)
    
    # For simple documents, if the top candidate looks like an address/contact, try the next one
    top_candidate = candidates[0][2]
    if (len(candidates) > 1 and 
        any(pattern in top_candidate.upper() for pattern in ['ADDRESS:', 'PHONE:', 'EMAIL:', 'WWW.', 'RSVP:'])):
        return candidates[1][2]
    
    return top_candidate

def analyze_text_features(span: Dict) -> Dict[str, Any]:
    """
    Enhanced text feature analysis with better detection for simple documents.
    """
    text = span["text"].strip()
    font = span.get("font", "")
    
    try:
        # Enhanced pattern detection for simple documents
        address_patterns = [
            r'\d+\s+[A-Z\s]+(STREET|ST|AVENUE|AVE|ROAD|RD|BLVD|BOULEVARD|PARKWAY|PKWY|LANE|LN|DRIVE|DR|COURT|CT|CIRCLE|CIR)',
            r'[A-Z\s]+,\s*[A-Z]{2}\s+\d{5}',  # City, State ZIP
            r'\(\d{3}\)\s*\d{3}-\d{4}',  # Phone number
        ]
        
        contact_patterns = [
            r'(ADDRESS|PHONE|EMAIL|CONTACT|RSVP|LOCATION|TIME|DATE):\s*',
            r'WWW\.[A-Z0-9.-]+\.[A-Z]{2,}',
            r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}',
        ]
        
        business_patterns = [
            r'^[A-Z\s&]{3,30}$',  # All caps business names (not too long)
            r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # Title case business names
        ]
        
        instruction_patterns = [
            r'(REQUIRED|PLEASE|VISIT|FILL OUT|WAIVER|HOPE TO SEE|CLOSED TOE)',
            r'(PARENTS|GUARDIANS|CHILDREN|CHILD)',
        ]
        
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
            'ends_with_colon': text.endswith(':'),
            
            # Enhanced patterns for simple documents
            'is_address_line': any(re.search(pattern, text.upper()) for pattern in address_patterns),
            'is_contact_info': any(re.search(pattern, text.upper()) for pattern in contact_patterns),
            'is_business_name': any(re.search(pattern, text) for pattern in business_patterns),
            'is_instruction': any(re.search(pattern, text.upper()) for pattern in instruction_patterns),
            'is_website': bool(re.search(r'WWW\.|HTTP|\.COM|\.ORG|\.NET', text.upper())),
            'is_label': text.endswith(':') and len(text.split()) <= 3,
            
            # Content type detection
            'contains_common_heading_words': any(word in text.lower() for word in [
                'chapter', 'section', 'introduction', 'conclusion', 'summary', 
                'overview', 'background', 'methodology', 'results', 'discussion',
                # Add simple document headings
                'address', 'contact', 'information', 'details', 'instructions',
                'rsvp', 'location', 'time', 'date', 'event', 'party'
            ]),
            
            # Special handling for all-caps short phrases
            'is_short_caps': text.isupper() and 2 <= len(text.split()) <= 5,
            'is_long_caps': text.isupper() and len(text.split()) > 5,
        }
    except Exception as e:
        # Fallback with basic features if analysis fails
        logging.warning(f"Text feature analysis failed for '{text[:50]}...': {e}")
        features = {
            'is_bold': False, 'is_italic': False, 'is_caps': text.isupper() if text else False,
            'is_title_case': text.istitle() if text else False, 'has_numbers': False,
            'starts_with_number': False, 'word_count': len(text.split()) if text else 0,
            'char_count': len(text) if text else 0, 'ends_with_punct': False,
            'ends_with_colon': False, 'is_address_line': False, 'is_contact_info': False,
            'is_business_name': False, 'is_instruction': False, 'is_website': False,
            'is_label': False, 'contains_common_heading_words': False,
            'is_short_caps': False, 'is_long_caps': False
        }
    
    return features

def enhanced_font_clustering(font_data: List[Tuple[float, Dict]], n_clusters: int = 4) -> Dict[float, str]:
    """
    Enhanced font clustering optimized for simple documents.
    """
    if len(font_data) < 3:
        # For very simple documents, just use size-based classification
        font_sizes = [fd[0] for fd in font_data]
        unique_sizes = sorted(set(font_sizes), reverse=True)
        mapping = {}
        for i, size in enumerate(unique_sizes):
            if i == 0:
                mapping[size] = 'H1'  # Largest text
            elif i == 1:
                mapping[size] = 'H2'  # Second largest
            else:
                mapping[size] = 'H3'  # Everything else
        return mapping
    
    try:
        # Extract features for clustering
        font_sizes = np.array([fd[0] for fd in font_data])
        bold_scores = np.array([1.0 if fd[1]['is_bold'] else 0.0 for fd in font_data])
        caps_scores = np.array([1.0 if fd[1]['is_caps'] else 0.0 for fd in font_data])
        business_scores = np.array([1.0 if fd[1]['is_business_name'] else 0.0 for fd in font_data])
        
        # For simple documents, give more weight to formatting
        features = np.column_stack([
            font_sizes,
            bold_scores * np.max(font_sizes) * 0.15,  # Increased weight for bold
            caps_scores * np.max(font_sizes) * 0.1,   # Weight for caps
            business_scores * np.max(font_sizes) * 0.2  # Weight for business names
        ])
        
        # Determine optimal number of clusters (usually fewer for simple docs)
        unique_sizes = len(set(font_sizes))
        n_clusters = min(n_clusters, unique_sizes, 3)  # Max 3 levels for simple docs
        
        if n_clusters < 2:
            return {font_sizes[0]: 'H1'}
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(features)
        
        # Map clusters to heading levels based on characteristics
        cluster_stats = defaultdict(lambda: {'sizes': [], 'caps': [], 'bold': [], 'business': []})
        for i, (size, feats) in enumerate(font_data):
            cluster_id = kmeans.labels_[i]
            cluster_stats[cluster_id]['sizes'].append(size)
            cluster_stats[cluster_id]['caps'].append(feats['is_caps'])
            cluster_stats[cluster_id]['bold'].append(feats['is_bold'])
            cluster_stats[cluster_id]['business'].append(feats['is_business_name'])
        
        # Score clusters for heading likelihood
        cluster_scores = []
        for cluster_id, stats in cluster_stats.items():
            avg_size = np.mean(stats['sizes'])
            caps_ratio = np.mean(stats['caps'])
            bold_ratio = np.mean(stats['bold'])
            business_ratio = np.mean(stats['business'])
            
            # Combined score favoring larger, formatted text
            score = avg_size + (caps_ratio * 5) + (bold_ratio * 8) + (business_ratio * 10)
            cluster_scores.append((cluster_id, score, avg_size))
        
        # Sort by score (descending)
        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Assign levels
        levels = ['H1', 'H2', 'H3']
        mapping = {}
        
        for rank, (cluster_id, score, avg_size) in enumerate(cluster_scores):
            level = levels[min(rank, len(levels) - 1)]
            for i, (size, feats) in enumerate(font_data):
                if kmeans.labels_[i] == cluster_id:
                    mapping[size] = level
        
        return mapping
        
    except Exception as e:
        # Fallback to simple size-based mapping
        logging.warning(f"Font clustering failed, using fallback: {e}")
        font_sizes = [fd[0] for fd in font_data]
        unique_sizes = sorted(set(font_sizes), reverse=True)
        levels = ['H1', 'H2', 'H3']
        mapping = {}
        for i, size in enumerate(unique_sizes[:3]):
            mapping[size] = levels[i]
        return mapping

def is_likely_heading(text: str, features: Dict, context: Dict) -> Tuple[bool, float]:
    """
    Enhanced heading detection optimized for simple documents like flyers and invitations.
    """
    score = 0.0
    max_score = 15.0  # Increased max score for more granular scoring
    
    # Basic length criteria - more flexible for simple documents
    if 2 <= len(text) <= 150:  # Increased upper limit
        score += 2.0
    elif len(text) > 200:
        score -= 2.0
    
    # Formatting criteria - enhanced for simple documents
    if features['is_bold']:
        score += 3.0  # Increased weight for bold
    
    if features['is_short_caps']:  # Short all-caps text (likely headings)
        score += 3.5
    elif features['is_long_caps'] and features['is_address_line']:  # Long caps that's an address
        score -= 1.0  # Penalize address lines
    elif features['is_caps']:
        score += 1.5
    
    if features['is_title_case']:
        score += 1.5
    
    # Label detection (e.g., "ADDRESS:", "RSVP:")
    if features['is_label']:
        score += 4.0  # High score for labels
    
    # Business name detection
    if features['is_business_name'] and not features['is_address_line']:
        score += 3.0
    
    # Content type scoring
    if features['starts_with_number']:
        score += 1.0
    
    if features['contains_common_heading_words']:
        score += 2.0
    
    # Context criteria
    if context.get('is_isolated', False):  # Standalone line
        score += 2.0
    
    if context.get('has_content_after', False):  # Has content following
        score += 1.0
    
    if context.get('position_on_page', 1.0) < 0.3:  # Near top of page
        score += 1.5
    
    # Special patterns for simple documents
    document_type = context.get('document_type', 'mixed')
    
    if document_type == 'simple':
        # For simple documents, be more inclusive
        if features['is_caps'] or features['is_bold'] or features['is_label']:
            score += 2.0
        
        # Common simple document headings
        simple_headings = ['address', 'phone', 'email', 'contact', 'rsvp', 'location', 
                          'time', 'date', 'website', 'information', 'details', 'instructions']
        if any(heading in text.lower() for heading in simple_headings):
            score += 2.5
        
        # Short phrases can be headings in simple documents
        if 1 <= len(text.split()) <= 4 and not features['is_website']:
            score += 1.5
    
    # Negative indicators
    if features['is_address_line'] and not features['is_label']:
        score -= 2.0  # Address lines are usually not headings unless they're labels
    
    if features['is_website']:
        score -= 2.0  # Websites are usually not headings
    
    if features['is_instruction'] and len(text.split()) > 8:
        score -= 1.5  # Long instructions are usually body text
    
    if len(text.split()) > 20:  # Very long text
        score -= 3.0
    
    if text.count('.') > 2:  # Multiple sentences
        score -= 2.0
    
    # Special case: if it's a phone number or email, definitely not a heading
    if re.search(r'\(\d{3}\)\s*\d{3}-\d{4}|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text):
        score -= 5.0
    
    confidence = max(0.0, min(1.0, score / max_score))
    is_heading = confidence > 0.4  # Lower threshold for simple documents
    
    return is_heading, confidence

def detect_document_type(all_text_elements: List[Dict]) -> str:
    """
    Enhanced document type detection with better recognition of simple documents.
    """
    if not all_text_elements:
        return 'simple'
    
    total_elements = len(all_text_elements)
    
    # Count various indicators
    heading_indicators = 0
    caps_count = 0
    structured_indicators = 0
    simple_indicators = 0
    address_indicators = 0
    contact_indicators = 0
    
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
        
        # Count simple document indicators
        if features.get('is_label', False) or features.get('is_contact_info', False):
            simple_indicators += 1
        
        if features.get('is_address_line', False):
            address_indicators += 1
        
        if features.get('is_contact_info', False) or features.get('is_website', False):
            contact_indicators += 1
    
    # Enhanced decision logic
    simple_score = simple_indicators + address_indicators + contact_indicators
    structured_score = structured_indicators + (heading_indicators * 0.5)
    
    if total_elements < 15:  # Very short document
        return 'simple'
    elif simple_score > total_elements * 0.2:  # Lots of contact/address info
        return 'simple'
    elif structured_score > 3 or heading_indicators > total_elements * 0.15:
        return 'structured'
    elif caps_count > total_elements * 0.4:  # Lots of caps text (flyer style)
        return 'simple'
    else:
        return 'mixed'

def extract_headings_enhanced(doc: fitz.Document, median_height: float) -> List[Dict[str, Any]]:
    """
    Enhanced heading extraction optimized for simple documents.
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
                    if text and len(text) >= 1:  # Even single characters can be relevant
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
    logging.info(f"Detected document type: {document_type}")
    
    # Enhanced font clustering
    font_mapping = enhanced_font_clustering(all_text_data)
    
    # Adjust confidence threshold based on document type
    confidence_threshold = {
        'structured': 0.6,
        'simple': 0.35,  # Lower threshold for simple documents
        'mixed': 0.45
    }.get(document_type, 0.45)
    
    # Extract headings with enhanced context analysis
    headings = []
    
    for page_idx, page_elements in enumerate(page_structures):
        for line_idx, line_elements in enumerate(page_elements):
            # Handle both single and multi-span lines
            if len(line_elements) >= 1:
                # Combine text from all spans in the line
                combined_text = " ".join(element['text'] for element in line_elements).strip()
                if not combined_text:
                    continue
                
                # Use the largest font size and combine features
                primary_element = max(line_elements, key=lambda x: x['font_size'])
                font_size = primary_element['font_size']
                
                # Combine features from all spans
                combined_features = primary_element['features'].copy()
                # If any span is bold/caps, consider the whole line as such
                combined_features['is_bold'] = any(elem['features']['is_bold'] for elem in line_elements)
                combined_features['is_caps'] = any(elem['features']['is_caps'] for elem in line_elements)
                
                # Re-analyze combined text
                span_like = {'text': combined_text, 'font': '', 'size': font_size}
                combined_features.update(analyze_text_features(span_like))
                
                # Get font level
                font_level = font_mapping.get(font_size, 'H3')
                
                # For simple documents, allow more flexibility
                if document_type == 'simple':
                    # Labels and formatted text get priority
                    if combined_features.get('is_label', False) or combined_features.get('is_bold', False):
                        if font_level == 'body':  # Override body classification
                            font_level = 'H3'
                
                # Context analysis
                context = {
                    'is_isolated': len(line_elements) == 1,
                    'has_content_after': line_idx < len(page_elements) - 1,
                    'position_on_page': line_idx / max(1, len(page_elements)),
                    'document_type': document_type,
                    'total_lines_on_page': len(page_elements)
                }
                
                # Check if it's likely a heading
                is_heading, confidence = is_likely_heading(combined_text, combined_features, context)
                
                if is_heading and confidence > confidence_threshold:
                    headings.append({
                        "level": font_level,
                        "text": combined_text,
                        "page": page_idx + 1,
                        "confidence": confidence,
                        "source": "analysis",
                        "document_type": document_type,
                        "font_size": font_size
                    })
    
    # Enhanced post-processing for simple documents
    seen_texts = set()
    filtered_headings = []
    
    # Sort by confidence first, then by page and position
    headings.sort(key=lambda x: (-x['confidence'], x['page']))
    
    for heading in headings:
        text_lower = heading['text'].lower().strip()
        
        # For simple documents, be less strict about duplicates
        if document_type == 'simple':
            # Allow similar but not identical headings
            is_duplicate = text_lower in seen_texts
        else:
            # More strict duplicate detection for structured documents
            is_duplicate = any(text_lower in seen or seen in text_lower for seen in seen_texts)
        
        if not is_duplicate and heading['confidence'] > confidence_threshold:
            seen_texts.add(text_lower)
            filtered_headings.append(heading)
    
    # Sort final results by page and original order
    filtered_headings.sort(key=lambda x: (x['page'], -x['confidence']))
    
    # For simple documents, ensure we have some headings
    if document_type == 'simple' and len(filtered_headings) < 2:
        # Lower the threshold and try again
        backup_headings = [h for h in headings if h['confidence'] > 0.25]
        if backup_headings:
            filtered_headings = backup_headings[:5]  # Limit to top 5
    
    return filtered_headings

def validate_heading_hierarchy(headings: List[Dict]) -> List[Dict]:
    """
    Enhanced hierarchy validation for simple documents.
    """
    if not headings:
        return headings
    
    validated = []
    last_level = 0
    
    for heading in headings:
        try:
            # Safely extract level number
            level_str = heading['level']
            if level_str.startswith('H') and len(level_str) > 1:
                level_num = int(level_str[1:])
            else:
                level_num = 2
                heading['level'] = 'H2'
            
            # For simple documents, be more lenient with hierarchy
            document_type = heading.get('document_type', 'mixed')
            if document_type == 'simple':
                # Don't enforce strict hierarchy for simple documents
                # Just ensure we don't have H1 followed immediately by H3
                if last_level > 0 and level_num > last_level + 2:
                    level_num = last_level + 1
                    heading['level'] = f'H{level_num}'
                    heading['adjusted'] = True
            else:
                # Stricter hierarchy for structured documents
                if level_num > last_level + 2:
                    level_num = min(level_num, last_level + 1)
                    heading['level'] = f'H{level_num}'
                    heading['adjusted'] = True
            
            validated.append(heading)
            last_level = level_num
            
        except (ValueError, IndexError) as e:
            heading['level'] = 'H2'
            heading['adjusted'] = True
            validated.append(heading)
            last_level = 2
    
    return validated

def process_pdf(input_path: str) -> Dict[str, Any]:
    """
    Enhanced PDF processing optimized for simple documents.
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
    
    # Enhanced metadata
    metadata = {
        'total_headings': len(outline),
        'avg_confidence': np.mean([h.get('confidence', 0) for h in outline]) if outline else 0,
        'page_count': len(page_heights),
        'extraction_method': 'enhanced_analysis_v2',
        'document_type': outline[0].get('document_type', 'unknown') if outline else 'unknown',
        'heading_levels': list(set(h.get('level', 'unknown') for h in outline)) if outline else []
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
                metadata = result.get('metadata', {})
                doc_type = metadata.get('document_type', 'unknown')
                heading_count = metadata.get('total_headings', 0)
                avg_confidence = metadata.get('avg_confidence', 0)
                
                logging.info(f"Successfully processed {filename} - "
                           f"Type: {doc_type}, Found {heading_count} headings "
                           f"(avg confidence: {avg_confidence:.2f})")
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
    parser = argparse.ArgumentParser(description="Enhanced PDF Outline Extractor - Optimized for Simple Documents")
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