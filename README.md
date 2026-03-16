# Kavach AI - AI-Powered Scam Call Detection

> **Kavach** (कवच) means *shield* or *armor* in Sanskrit, your real-time defense against AI-driven voice fraud.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-kavach--ai--app--2026.azurewebsites.net-blue?style=for-the-badge)](https://kavach-ai-app-2026.azurewebsites.net/)
[![Built With](https://img.shields.io/badge/Built%20With-Python%20%7C%20PyTorch%20%7C%20HuBERT%20%7C%20Whisper-orange?style=for-the-badge)](https://github.com/the-calmcoder/kavach-ai)
[![Powered By](https://img.shields.io/badge/Powered%20By-Microsoft%20Azure-0078D4?style=for-the-badge&logo=microsoft-azure)](https://azure.microsoft.com/)

---

## Table of Contents

- [The Problem](#the-problem)
- [What is Kavach AI?](#what-is-kavach-ai)
- [How It Works](#how-it-works)
- [AI Models Used](#ai-models-used)
- [Project Structure](#project-structure)
- [Using the Live Website](#using-the-live-website)
- [Tech Stack](#tech-stack)
- [Roadmap](#roadmap)

---

## The Problem

AI-powered voice fraud is one of the fastest-growing threats in digital security:

- Scammers use **AI voice cloning** to impersonate family members, bank officials, and government authorities
- In India alone, victims have lost approximately **Rs. 475 Crores** to voice-based scams
- Global annual losses from AI voice fraud exceed **Rs. 22,845 Crores**
- Around **1 in 4 people** are actively being targeted, with the elderly and rural populations being the most vulnerable
- A single successful scam call can steal between **Rs. 20,000 and Rs. 1 Lakh** from a victim

---

## What is Kavach AI?

Kavach AI is a multi-engine AI system that analyzes audio recordings to determine:

1. **Is the voice real or AI-generated?** Deepfake voice detection using HuBERT
2. **Is there a scam attempt happening?** Behavioral and semantic fraud analysis
3. **What is being said?** Full multilingual transcription via Whisper
4. **Why was it flagged?** Plain-language explainability for every verdict

Results are delivered in under **10 seconds**, with a comprehensive risk report.

---

## How It Works

```
+--------------------------------------------------------------+
|                    Incoming Audio File                        |
+-------------------------+------------------------------------+
                          |
           +--------------v--------------+
           |     audio_processing.py      |
           |  (noise reduction, chunking) |
           +------+---------------+-------+
                  |               |
    +-------------v---+   +-------v----------+
    |  model_core.py  |   |  transcriber.py   |
    |  HuBERT model   |   |  Whisper STT      |
    | (voice features)|   | (speech to text)  |
    +--------+--------+   +--------+----------+
             |                     |
    +--------v--------+   +--------v----------+
    |  humanity.py    |   |  fraud_engine.py   |
    | Human vs AI     |   | Scam intent score  |
    | verdict         |   | (semantic analysis)|
    +--------+--------+   +--------+-----------+
             |                     |
             +----------+----------+
                        |
              +---------v----------+
              |    decision.py      |
              |  Final risk verdict |
              +---------+----------+
                        |
              +---------v----------+
              |  explainability.py  |
              |  Human-readable     |
              |  explanation        |
              +---------+----------+
                        |
              +---------v----------+
              |      api.py         |
              |  REST API layer     |
              +---------+----------+
                        |
              +---------v----------+
              |  Frontend Dashboard |
              |  Risk Report Card   |
              +--------------------+
```

**Three-step user flow:**

| Step | What Happens |
|------|-------------|
| 1. Upload Audio | Upload an MP3 or WAV recording (up to 10 MB) |
| 2. AI Analyzes | HuBERT scans for deepfake artifacts, Whisper transcribes, Fraud Engine checks intent |
| 3. Get Risk Report | A full report is returned with AI/Human verdict, fraud score, transcript, and explanation |

---

## AI Models Used

| Model | Purpose |
|-------|---------|
| **Facebook HuBERT** | Deep learning model for detecting synthetic/AI-generated voice artifacts |
| **OpenAI Whisper** | Multilingual speech-to-text transcription (supports Hindi, English, and more) |
| **Sentence Transformers** | Semantic analysis to detect scam patterns like false urgency, coercion, and payment demands |
| **Custom Classifier** (`classifier_weights.pth`) | Fine-tuned PyTorch model for the human vs. AI voice classification task |
| **Fraud Engine** (`fraud_weights.pkl/.json`) | Trained fraud detection model scoring intent and behavioral red flags |

---

## Project Structure

```
kavach-ai/
|
+-- api.py                   # FastAPI/Flask REST API, connects frontend to ML pipeline
+-- audio_processing.py      # Audio preprocessing, noise reduction, chunking, feature extraction
+-- transcriber.py           # Speech-to-text using OpenAI Whisper
+-- model_core.py            # Core ML model loader and inference logic
+-- humanity.py              # Human vs. AI voice classifier
+-- human_baseline.json      # Baseline acoustic profiles for human voices
+-- fraud_engine.py          # Scam intent detection engine
+-- fraud_weights.pkl        # Pickled fraud detection model weights
+-- fraud_weights.json       # JSON representation of fraud model configuration
+-- classifier_weights.pth   # PyTorch weights for the voice deepfake classifier
+-- decision.py              # Aggregates all signals into a final risk verdict
+-- explainability.py        # Generates human-readable explanations for verdicts
+-- utils.py                 # Shared utility functions
+-- requirements.txt         # Python dependencies
+-- startup.sh               # One-command startup script
|
+-- frontend/                # Web interface (HTML, CSS, JavaScript)
|   +-- ...                  # Dashboard, upload UI, risk report card
|
+-- .github/
    +-- workflows/           # CI/CD pipeline for Azure deployment
```

---

## Using the Live Website

The app is fully deployed and requires no installation.

**Live URL:** [https://kavach-ai-app-2026.azurewebsites.net/](https://kavach-ai-app-2026.azurewebsites.net/)

### Step-by-Step Guide

#### 1. Open the Website
Navigate to the link above. You'll land on the Kavach AI homepage with an overview of the problem and solution.

#### 2. Go to the Upload Section
Scroll down to the **"Powerful Features"** section or click **"Get Started"** in the hero banner to reach the audio upload panel.

#### 3. Upload Your Audio File
- Drag and drop an audio file onto the upload zone, or click the box to browse
- Accepted formats: **MP3** or **WAV**
- Maximum file size: **10 MB**

> **Tip:** Record a suspicious call, save it as MP3 or WAV, and upload it for analysis.

#### 4. Click "Analyze Audio"
Hit the **Analyze Audio** button. A loading animation will appear while the pipeline processes your file.

#### 5. Read Your Risk Report
Within about 10 seconds, your **Risk Report** will appear with the following sections:

| Section | What it shows |
|---------|--------------|
| Voice Analysis | Human or AI verdict with confidence percentage |
| Scam Detection | Fraud score (0 to 100%) indicating scam likelihood |
| Language | Detected language (e.g., Hindi, English) |
| Transcript | Full text of what was said in the call |
| Voice Explanation | Why the voice was classified as human or AI |
| Scam Explanation | Which phrases or patterns triggered the fraud alert |
| Processing Metrics | Breakdown of time spent in each analysis stage |

#### 6. Check Your History
Previous analyses are shown in the **"Risk Report History"** section, stored locally in your browser for privacy. Kavach AI never stores your audio or personal data on its servers.

#### 7. Clear History
Click **"Clear History"** to remove all locally stored analysis results from your browser.

---

### Understanding Your Results

**Voice Analysis Verdict:**
- `HUMAN VOICE` - The caller is likely a real person
- `AI GENERATED` - The voice shows signs of synthetic/deepfake generation

**Fraud Score:**

| Score Range | Interpretation |
|------------|----------------|
| 0% to 20% | Low risk, appears to be a normal call |
| 21% to 50% | Moderate risk, some suspicious patterns detected |
| 51% to 80% | High risk, strong scam indicators present |
| 81% to 100% | Critical risk, very likely a scam attempt |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python, FastAPI / Flask |
| **Deep Learning** | PyTorch, HuBERT (Facebook AI) |
| **Transcription** | OpenAI Whisper |
| **NLP / Fraud** | Sentence Transformers, scikit-learn |
| **Frontend** | HTML, CSS, JavaScript |
| **Cloud** | Microsoft Azure (App Service) |
| **CI/CD** | GitHub Actions |

---

## Roadmap

| Feature | Status |
|---------|--------|
| Upload and analyze audio files | Live |
| AI voice deepfake detection | Live |
| Scam intent analysis | Live |
| Multilingual transcription | Live |
| Explainability reports | Live |
| Browser-local history | Live |
| **Live call monitoring (Phase 2)** | Coming Soon |

---

## About

Kavach AI combines state-of-the-art deep learning with a practical, accessible interface to help everyday users protect themselves from the growing threat of AI-powered phone fraud.

*All audio processing is performed securely. Your audio files and personal data are never stored on Kavach AI's servers.*

---

*© 2026 Kavach AI*
