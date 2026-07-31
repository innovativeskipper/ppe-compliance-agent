# PPE Compliance Agent

## Objective
An AI-powered system for detecting personal protective equipment (PPE) compliance violations in a chemical plant environment, using computer vision for detection and a multi-agent LLM system for automated triage, alerting, and compliance reporting.

## Status
Work in progress — bootcamp capstone project.

## Tech Stack
- YOLOv8 (Ultralytics) for PPE detection
- CrewAI + Anthropic Claude for agentic alerting
- Streamlit + Hugging Face Spaces for deployment
## Dataset & EDA

**Source:** Roboflow Universe — "Construction Site Safety" dataset (version 27)
**Size:** 2,801 images (train: 2,603 · valid: 114 · test: 82)
**Classes (10):** Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, vehicle

**Class distribution (instance counts):**

| Class | Count |
|---|---|
| Person | 10,031 |
| machinery | 5,337 |
| NO-Safety Vest | 4,153 |
| Hardhat | 3,551 |
| NO-Mask | 3,362 |
| Safety Cone | 3,306 |
| Safety Vest | 3,258 |
| NO-Hardhat | 2,428 |
| Mask | 1,792 |
| vehicle | 1,617 |

**Observations:**
- `Person` and `machinery` dominate as broad context classes, appearing frequently across nearly all images.
- Vest-related classes (Safety Vest vs. NO-Safety Vest) are reasonably balanced.
- Hardhat classes show mild imbalance, with compliant instances (Hardhat) outnumbering violations (NO-Hardhat) by roughly 1.5x.
- Mask classes are more skewed, with violations (NO-Mask) nearly double the compliant instances (Mask).
- `vehicle` and `Mask` are the smallest classes overall and are expected to show weaker detection performance due to fewer training examples.
- No `NO-Gloves` or `NO-Goggles` classes exist in this dataset; project scope is accordingly narrowed to **hardhat, mask, and safety vest** compliance.