Of course. Here is the content of the `README.md` file. You can copy and paste this text into a file on your local machine.

-----

# 🧠 PDF Outline Extractor (Adobe India Hackathon - Round 1A Submission)

This project is a high-performance document parser built for the Adobe India Hackathon. It addresses the 'Connecting the Dots' challenge by transforming unstructured PDFs into clean, semantic outlines. The solution intelligently handles diverse document types—from academic papers to flyers—using a combination of machine learning and layout analysis to extract titles and hierarchical headings (H1, H2, H3).

### ✨ Features

  * 📄 **Versatile Outline Extraction:** Reliably extracts titles and hierarchical headings (H1-H3) from diverse and complex PDF layouts.
  * 🧠 **ML-Powered Detection:** Employs K-Means clustering to dynamically identify heading styles, eliminating the need for hard-coded rules and adapting to each document's unique structure.
  * 💪 **High-Speed Batch Processing:** Leverages a `ProcessPoolExecutor` for efficient, parallel processing of large document sets.
  * 📚 **Smart Fallback:** Intelligently uses a PDF's built-in Table of Contents for maximum accuracy when available.
  * 🛠️ **Reproducible Environment:** Fully containerized with Docker for consistent, one-command execution.
  * 📦 **Standardized Output:** Generates a clean, individual `.json` file for each PDF processed.

### 📁 Directory Structure

The project is organized as follows:

```
.
├── input/               # Place your PDF files here
├── output/              # Extracted outlines will be saved here
├── 1a.py                # The main Python script for the extractor
├── Dockerfile           # Instructions to build the Docker container
├── requirements.txt     # Python dependencies
├── .dockerignore        # Specifies files to exclude from the Docker build
└── README.md            # This file
```

### 🚀 How to Run

This solution is designed to be run inside a Docker container to ensure a consistent environment, as per the hackathon rules.

#### 1\. Build the Docker Image

Navigate to the project's root directory in your terminal and run the following command. Replace `mysolutionname:somerandomidentifier` with your team's name and a version tag (e.g., `team_alpha:v1`).

```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
```

#### 2\. Prepare Input Files

Place all your `.pdf` files into the `input/` directory.

#### 3\. Run the Container

Execute the following command to process the PDFs. The script will automatically find all PDFs in the `input` directory, process them, and place the resulting `.json` files in the `output` directory.

```bash
docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none mysolutionname:somerandomidentifier
```

### 🧪 Output Format

For each `filename.pdf` in the input directory, a corresponding `filename.json` will be created in the output directory with the following structure:

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

### 🧠 Methodology

  * **Title Detection:** Identifies the title by analyzing font size, position on the first page, and textual cues.
  * **Heading Extraction:**
      * First, attempts to use the PDF's built-in Table of Contents for maximum accuracy.
      * If no TOC is found, it performs a deep analysis of text blocks, using font size, weight (bold), and position to identify potential headings.
      * A K-Means clustering algorithm groups font styles into `H1`, `H2`, and `H3` levels.
  * **Parallel Processing:** Leverages multiprocessing to handle large batches of documents quickly and efficiently.
  * **Error Handling:** Includes robust error handling to gracefully manage corrupted or unparsable PDFs without crashing.

### 🧱 Dependencies

All dependencies are managed within the `Dockerfile`.

  * PyMuPDF (fitz)
  * scikit-learn
  * numpy
  * scipy
  * tqdm

### 🔒 License

This project is developed as part of **Adobe India Hackathon 2025 - Round 1A**. Usage outside the scope of this event must be authorized by the authors.

### 👤 Authors

  * Lakshin Khurana
  * Yashvardhan Nayal

Feel free to connect or reach out for discussions and collaboration opportunities
