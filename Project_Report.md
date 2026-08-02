# PPE Compliance Detection Agent — Capstone Project

## Objective
An AI-powered system for detecting personal protective equipment (PPE) compliance
violations in a chemical plant environment, combining computer vision detection with
a multi-agent LLM system for automated triage, alerting, and compliance reporting.

## Problem Statement
Manual PPE compliance monitoring in industrial settings is labor-intensive and
inconsistent. This project explores an automated pipeline that detects violations
from images and routes appropriately-severity alerts to the right personnel.

## Why It Matters
Chemical plant environments carry elevated safety risk; automated, consistent PPE
monitoring can reduce human oversight burden and improve response time to violations.

## Approach & Architecture
1. **Data & EDA**: Roboflow "Construction Site Safety" dataset (v27), 2,801 images,
   10 classes. EDA identified class imbalance (vehicle/Mask underrepresented).
2. **Modeling**: Classical ML baseline (Random Forest on crop features) established
   for comparison; YOLOv8n fine-tuned as the primary detector, tracked via W&B.
3. **Detection Pipeline**: Structured violation extraction from YOLOv8 output
   (hardhat/mask/vest compliance; gloves out of scope — no labeled data available).
4. **Agentic Alerting**: Multi-agent pattern (Triage → Routing → Reporting), each
   with distinct role/reasoning, implemented via direct LLM orchestration
   (Groq/Llama 3.3 70B) after a CrewAI framework compatibility issue.
5. **Deployment**: Streamlit app on Streamlit Community Cloud, model hosted on
   Hugging Face Hub, live and publicly accessible.

## Results
- YOLOv8 mAP50 = 0.77, mAP50-95 = 0.46
- Baseline classifier: [0.76]
- Per-class performance notes: vehicle and Mask classes show weaker detection
  due to smaller training representation; some vest detections show moderate
  (~50%) confidence, likely due to lighting/angle variation.

## Modules Applied
EDA · Classical ML · Deep Learning (YOLOv8/Ultralytics) · Computer Vision ·
NLP/LLM Agentic Systems · Hugging Face (Hub + ecosystem) · GitHub version control ·
Colab development · MLOps (W&B experiment tracking, reproducible requirements)

## Limitations & Future Work
- Gloves/goggles detection not covered (no labeled data in available dataset)
- Analytics are session-based, not persistently stored across visits
- Confidence calibration could improve with more training data/epochs

## Deployment
Live app: [your Streamlit Cloud URL]
Model repo: https://huggingface.co/innovativeskipper/ppe-compliance-yolov8
Code repo: https://github.com/innovativeskipper/ppe-compliance-agent