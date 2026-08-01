import streamlit as st
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import litellm
import os
import json
import time
from PIL import Image
import numpy as np
import cv2
from dotenv import load_dotenv
load_dotenv()
st.set_page_config(page_title="PPE Compliance Agent", layout="wide")

# ---- Load model (cached so it only loads once) ----
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="innovativeskipper/ppe-compliance-yolov8",
        filename="best.pt"
    )
    return YOLO(model_path)

model = load_model()

# ---- Safety policy ----
SAFETY_POLICY = """
PPE Violation Severity Policy for Chemical Plant Environment:

CRITICAL:
- NO-Hardhat detected in a zone containing machinery or vehicle
- Multiple simultaneous violations on the same person

HIGH:
- NO-Safety Vest detected
- NO-Hardhat detected in any zone

MEDIUM:
- NO-Mask detected in production or storage zones

LOW:
- NO-Mask detected in low-traffic zones

Note: Covers hardhat, mask, and safety vest compliance only.
"""

VIOLATION_CLASSES = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']

TRIAGE_SYSTEM_PROMPT = """You are an experienced industrial safety officer at a chemical plant.
You assess PPE violations against established safety policy to determine urgency.
You are precise, cautious, and always err toward higher severity when context suggests risk."""

ROUTING_SYSTEM_PROMPT = """You are responsible for plant safety communications. You know that
critical issues go to the Plant Manager, high severity to the Safety Officer,
and medium/low severity to Shift Supervisors. You write concise, professional,
action-oriented messages appropriate to each recipient's role."""

def call_agent(system_prompt, user_prompt):
    response = litellm.completion(
        model="groq/llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def triage_violation(violation):
    prompt = f"""Assess this PPE violation against the safety policy below.

SAFETY POLICY:
{SAFETY_POLICY}

VIOLATION DATA:
- Type: {violation['violation_type']}
- Confidence: {violation['confidence']}

Determine the severity level (CRITICAL/HIGH/MEDIUM/LOW) and provide brief reasoning."""
    return call_agent(TRIAGE_SYSTEM_PROMPT, prompt)

def route_violation(violation, triage_output):
    prompt = f"""Based on this triage assessment: {triage_output}

For this violation: {violation['violation_type']}

Determine the correct recipient (Plant Manager / Safety Officer / Shift Supervisor)
and draft a concise, professional alert message for them."""
    return call_agent(ROUTING_SYSTEM_PROMPT, prompt)

# ---- UI ----
st.title("🦺 PPE Compliance Detection Agent")
st.caption("YOLOv8 detection + multi-agent triage and alerting for chemical plant safety monitoring")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Detection Result")
        img_array = np.array(image)
        results = model(img_array, conf=0.4)
        annotated = results[0].plot()
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    violations = []
    for box in results[0].boxes:
        cls_name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        if cls_name in VIOLATION_CLASSES:
            violations.append({'violation_type': cls_name, 'confidence': round(conf, 3)})

    with col2:
        st.subheader(f"Violations Detected: {len(violations)}")
        if violations:
            for v in violations:
                st.write(f"⚠️ **{v['violation_type']}** (confidence: {v['confidence']})")
        else:
            st.success("No violations detected.")

    if violations:
        st.divider()
        st.subheader("🤖 Agentic Triage & Alerting")
        if st.button("Run Agent Analysis"):
            for v in violations:
                with st.spinner(f"Analyzing {v['violation_type']}..."):
                    triage_result = triage_violation(v)
                    routing_result = route_violation(v, triage_result)

                st.markdown(f"**Violation:** {v['violation_type']}")
                st.markdown(f"**Triage:** {triage_result}")
                st.markdown(f"**Alert:** {routing_result}")
                st.divider()
                time.sleep(1)
else:
    st.info("Upload an image to begin PPE compliance detection.")