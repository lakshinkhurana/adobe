
🧠 PDF Outline Extractor (Adobe India Hackathon - Round 1A Submission)
This project is an automated, scalable PDF document parser that extracts structured outlines (headings) from academic, business, or simple flyer-style documents. It's optimized for batch processing and utilizes clustering, positional heuristics, and document styling to generate clean semantic outlines.

✨ Features
📄 Extracts title and headings (H1, H2, H3) from various PDF types.

🧠 Uses K-Means clustering on font sizes and styles for dynamic heading detection.

💪 Parallel processing using ProcessPoolExecutor for high-speed batch jobs.

📚 Intelligently falls back to the built-in PDF Table of Contents if available.

🛠️ Fully containerized with Docker for consistent, reproducible execution.

📦 Outputs individual .json file per PDF.

📁 Directory Structure
The project is organized as follows:

.
├── input/               # Place your PDF files here
├── output/              # Extracted outlines will be saved here
├── 1a.py                # The main Python script for the extractor
├── Dockerfile           # Instructions to build the Docker container
├── requirements.txt     # Python dependencies
├── .dockerignore        # Specifies files to exclude from the Docker build
└── README.md            # This file

🚀 How to Run
This solution is designed to be run inside a Docker container to ensure a consistent environment, as per the hackathon rules.

1. Build the Docker Image
Navigate to the project's root directory in your terminal and run the following command. Replace mysolutionname:somerandomidentifier with your team's name and a version tag (e.g., team_alpha:v1).

docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .

2. Prepare Input Files
Place all your .pdf files into the input/ directory.

3. Run the Container
Execute the following command to process the PDFs. The script will automatically find all PDFs in the input directory, process them, and place the resulting .json files in the output directory.

docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none mysolutionname:somerandomidentifier

🧪 Output Format
For each filename.pdf in the input directory, a corresponding filename.json will be created in the output directory with the following structure:

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

🧠 Methodology
Title Detection: Identifies the title by analyzing font size, position on the first page, and textual cues.

Heading Extraction:

First, attempts to use the PDF's built-in Table of Contents for maximum accuracy.

If no TOC is found, it performs a deep analysis of text blocks, using font size, weight (bold), and position to identify potential headings.

A K-Means clustering algorithm groups font styles into H1, H2, and H3 levels.

Parallel Processing: Leverages multiprocessing to handle large batches of documents quickly and efficiently.

Error Handling: Includes robust error handling to gracefully manage corrupted or unparsable PDFs without crashing.

🧱 Dependencies
All dependencies are managed within the Dockerfile.

PyMuPDF (fitz)

scikit-learn

numpy

scipy

tqdm

🔒 License
This project is developed as part of Adobe India Hackathon 2025 - Round 1A. Usage outside the scope of this event must be authorized by the authors.

👤 Authors
Lakshin Khurana

Yashvardhan Nayal

Feel free to connect or reach out for discussions and collaboration opportunities!

Feel free to connect or reach out for discussions and collaboration opportunities!
