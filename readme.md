# Data Annotation QC Environment

## 💡 What This Is

A simulation of real-world annotation pipelines where AI must:
- Detect incorrect labels
- Correct them
- Reason under annotator noise

## 🚀 Unique Features

- Annotator reliability modeling
- Confidence calibration rewards
- Multi-difficulty tasks

## 🌍 Why It Matters

Used in:
- Meta AI pipelines
- HuggingFace datasets
- Scale AI workflows

## ⚙️ Run

pip install -r requirements.txt  
uvicorn app:app --reload  

## 🧪 API

GET /reset  
POST /step  
GET /state  

## 🏆 Goal

Train agents that can audit human-generated datasets.