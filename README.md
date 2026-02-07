# Qualitative Analysis of Educational Video Lectures

## Description

This project presents a Natural Language Processing (NLP)-based framework for analyzing the qualitative aspects of educational video lectures. The system converts lecture audio into text, processes the transcription, extracts pedagogically relevant features, and generates structured qualitative ratings.

The objective is to provide an automated, reproducible, and scalable method for evaluating lecture quality using measurable linguistic and semantic attributes.

---

## Features

The system evaluates lectures based on the following qualitative dimensions:

- Depth of Knowledge
- Readability
- Fluency
- Speech Pace
- Engagement
- Emotional Tone (using DistilBERT)
- Vocabulary Richness

---

## Dataset Information

### Data Source

The dataset consists of publicly available educational lecture videos:

- NPTEL (National Programme on Technology Enhanced Learning)  
  https://nptel.ac.in
- Public NLP-related lecture videos

These videos are converted into `.wav` format before processing.

### Data Format

- Input: `.wav` audio files
- Intermediate Output: Chunk-wise transcription CSV
- Final Output: CSV files containing qualitative metrics and ratings

Each lecture audio is segmented into 1-minute chunks for fine-grained analysis.

---

## Code Information

### index.py
- Converts `.wav` audio files into text transcriptions.
- Saves output in CSV format.

### combine.py
- Combines chunk-wise transcriptions into a single dataset.

### analysis.py
- Performs NLP-based qualitative analysis.
- Extracts readability, vocabulary richness, speech pace, engagement indicators, and sentiment.

### ratings.py
- Aggregates computed metrics.
- Normalizes values.
- Generates final qualitative ratings.

### Power BI Dashboard
- Used for visualization of results.
- Displays metric trends and overall lecture ratings.

---

## Computing Infrastructure

- Operating System: Windows 10 / Windows 11
- Processor: Intel i5 / Ryzen 5 (or above)
- RAM: 8GB recommended
- Python Version: 3.8+
- PyTorch backend for transformer execution
- Power BI Desktop for visualization
- Internet connection required during first execution (for model download)

GPU is optional. CPU is sufficient for small to medium datasets.

---

## Requirements

Install required libraries:

```bash
pip install pandas numpy nltk textstat speechrecognition librosa torch transformers scikit-learn matplotlib
```

Download required NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## NLP Model Used

### DistilBERT (Transformer-Based Model)

- Model: distilbert-base-uncased
- Framework: HuggingFace Transformers
- Backend: PyTorch

DistilBERT consists of:
- 6 transformer encoder layers
- 12 attention heads
- 768 hidden dimension
- Approximately 66 million parameters

It retains about 95% of BERT’s performance while being faster and lighter.

### Example Initialization

```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased")
```

Model weights are downloaded automatically during first execution.

---

## Methodology

### 1. Audio Segmentation
Lecture audio is divided into 1-minute chunks to enable granular analysis.

### 2. Speech-to-Text Conversion
Audio is converted to text using speech recognition and stored in CSV format.

### 3. Text Preprocessing
- Lowercasing
- Tokenization
- Stopword removal
- Punctuation cleaning
- Sentence segmentation

### 4. Feature Extraction (Assessment Metrics)

**Readability**  
Measured using Flesch Reading Ease / Flesch-Kincaid scores.

**Speech Pace**  
Measured as words per minute (WPM).

**Vocabulary Richness**  
Measured using Type-Token Ratio (TTR).

**Emotional Tone**  
Measured using DistilBERT sentiment analysis.

**Fluency**  
Estimated using filler word frequency and sentence continuity.

**Engagement**  
Measured using frequency of questions and interactive phrases.

**Depth of Knowledge**  
Estimated using domain-specific technical term frequency and concept density.

### 5. Rating Aggregation
All metrics are normalized and combined to generate final qualitative ratings.

---

## Usage Instructions

### Step 1: Prepare Audio
Convert lecture videos to `.wav` and place them inside the `audio/` directory.

### Step 2: Generate Transcriptions

```bash
python index.py
```

### Step 3: Combine Transcriptions

```bash
python combine.py
```

### Step 4: Perform Analysis

```bash
python analysis.py
```

### Step 5: Generate Ratings

```bash
python ratings.py
```

### Step 6: Visualization
Import the final CSV file into Power BI to create dashboards.

---

## Output

- Transcription CSV file
- Combined transcription file
- Analysis results CSV
- Final ratings dataset
- Power BI visualizations

---

## Reproducibility

To reproduce results:

1. Use lecture videos from NPTEL or similar sources.
2. Maintain consistent 1-minute segmentation.
3. Use Python 3.8+.
4. Use DistilBERT model: distilbert-base-uncased.
5. Execute scripts in order:

```
index.py → combine.py → analysis.py → ratings.py
```

Model weights are cached locally after first download.

---

## Conclusions

The system demonstrates that transformer-based NLP models can provide structured and objective insights into lecture quality. The framework enables scalable evaluation of educational content and supports data-driven instructional improvement.

---

## Limitations

- Speech recognition errors may affect transcription accuracy.
- Emotional tone detection is limited to textual sentiment.
- Depth of knowledge estimation is heuristic-based.
- No multimodal analysis (e.g., facial expressions, gestures).
- Performance depends on audio clarity.

---

## License

This project is licensed under the MIT License.

---

## Sample PowerBI Visualization.

![PowerBI Screenshot](https://github.com/user-attachments/assets/4ade8b43-097b-4043-9223-ba6a85cc973b)
![PowerBI Screenshot](https://github.com/user-attachments/assets/45a102ea-4183-4392-a0f3-5f1da742bbe8)
![PowerBI Screenshot](https://github.com/user-attachments/assets/7e887970-1588-4054-8b03-7666096084a2)

