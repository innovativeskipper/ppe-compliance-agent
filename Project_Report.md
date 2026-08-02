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

**Classical ML Baseline (Random Forest, hardhat compliance classification):**
| Metric | Violation | Compliant |
|---|---|---|
| Precision | 0.72 | 0.78 |
| Recall | 0.64 | 0.84 |
| F1-score | 0.68 | 0.81 |

Overall accuracy: 76% (384 test samples, trained on cropped-region color/histogram
features). This baseline served as a reference point for evaluating YOLOv8's
performance gain — not intended as a production-grade classifier.

**YOLOv8n (primary detection model) — validation set (114 images, 697 instances):**
- **mAP50: 0.766**
- **mAP50-95: 0.456**
- Overall precision: 0.882, recall: 0.698

**Per-class performance (mAP50):**
| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Hardhat | 0.916 | 0.759 | 0.836 | 0.496 |
| Mask | 0.953 | 0.810 | 0.896 | 0.626 |
| NO-Hardhat | 0.876 | 0.551 | 0.634 | 0.336 |
| NO-Mask | 0.890 | 0.546 | 0.606 | 0.272 |
| NO-Safety Vest | 0.849 | 0.642 | 0.735 | 0.409 |
| Person | 0.896 | 0.729 | 0.788 | 0.460 |
| Safety Cone | 0.950 | 0.862 | 0.910 | 0.505 |
| Safety Vest | 0.869 | 0.659 | 0.789 | 0.469 |
| machinery | 0.869 | 0.873 | 0.919 | 0.608 |
| vehicle | 0.756 | 0.548 | 0.547 | 0.377 |

**Key observations:**
- Detection performance is strongest on well-represented, visually distinct classes
  (machinery, Safety Cone, Mask) and weakest on `vehicle` and violation classes
  like `NO-Mask` and `NO-Hardhat` — consistent with the class imbalance identified
  in EDA (vehicle and Mask were the smallest classes in the training set).
- Recall is consistently lower than precision across violation classes, meaning
  the model is more likely to miss a violation than to falsely flag a compliant
  case — worth noting as a safety-relevant limitation (false negatives are more
  concerning than false positives in a compliance-monitoring context).
- In live app testing, some Safety Vest detections showed confidence near 50%,
  likely due to lighting/angle variation underrepresented in training data.
- The classical baseline's 76% accuracy on a simplified binary hardhat task,
  versus YOLOv8's full multi-class detection performance (mAP50 of 0.766),
  illustrates the meaningful capability gap between traditional feature-based
  ML and deep learning-based object detection for this task.

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