# 🧠 PDF Outline Extractor (Adobe India Hackathon - Round 1A Submission)

This project is an **automated, scalable PDF document parser** that extracts structured outlines (headings) from academic or business documents. It's optimized for batch processing and utilizes clustering, positional heuristics, and document styling to generate clean semantic outlines.

## ✨ Features

- 📄 Extracts **title** and **headings (H1, H2, H3)** from PDFs
- 🧠 Uses **K-Means clustering** on font sizes for dynamic heading detection
- 💪 Parallel processing using `ProcessPoolExecutor`
- 🔠 Optional enforcement of **bold fonts** for subheadings
- 📚 Fallback to **built-in PDF outlines** if available
- 🛠️ CLI-compatible and Docker-friendly
- 📦 Outputs:
  - Individual `.json` file per PDF or
  - Combined single `.json` file

## 📁 Directory Structure

```
.
├── input/               # Place your PDF files here
├── output/              # Extracted outlines will be saved here
├── main.py              # Main script (the one you're reading now)
├── README.md            # This file
```

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install pymupdf numpy scikit-learn tqdm
```

### 2. Place PDFs

Put all your `.pdf` files in the `input/` directory (or set a custom path using `--input`).

### 3. Run the script

```bash
python main.py
```

#### Available arguments:

| Flag               | Description                                                  | Default        |
|--------------------|--------------------------------------------------------------|----------------|
| `--input`          | Input folder containing PDF files                            | `./input`      |
| `--output`         | Output folder to save JSON files                             | `./output`     |
| `--workers`        | Number of parallel processes to use                          | `4`            |
| `--log`            | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)         | `INFO`         |
| `--combined`       | Generate a **single JSON file** for all PDFs                 | `False`        |
| `--compact`        | Output **compact JSON** without pretty indentation           | `False`        |
| `--strict-bold`    | Require bold text for subheadings (H2, H3)                   | `False`        |

#### Example:

```bash
python main.py --input ./my_pdfs --output ./results --combined --workers 8 --strict-bold
```

## 🧪 Output Format

Each JSON will contain:

```json
{
  "title": "Title of the Document",
  "outline": [
    {
      "level": "H1",
      "text": "Main Heading",
      "page": 1
    },
    {
      "level": "H2",
      "text": "Subheading",
      "page": 2
    }
  ]
}
```

## 🧠 Methodology

1. **Title Detection**: Based on the largest font on the first page.
2. **Heading Extraction**:
   - Font size clustering (KMeans) to classify heading levels
   - Filters using position (top-middle area) and optionally boldness
   - Fallback to PDF’s internal outline if available
3. **Parallel Processing**: Multiprocessing for faster batch handling
4. **Error Handling**: Graceful exception logging with traceback

## 🧱 Dependencies

- `PyMuPDF` (fitz)
- `scikit-learn`
- `numpy`
- `tqdm`
- `argparse`
- `logging`

## 🔒 License

This project is developed as part of **Adobe India Hackathon 2025 - Round 1A**. Usage outside the scope of this event must be authorized by the authors.

## 👤 Authors

**Lakshin Khurana and Yashvardhan Nayal**

Feel free to connect or reach out for discussions and collaboration opportunities!
