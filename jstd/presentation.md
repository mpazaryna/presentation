---
marp: true
theme: default
---

# Private Intelligence for Modern Practices

- Building solutions that don't depend on frontier models

---

# About Me

- Independent Developer 
- Self taught
- Spent 35 years in enterprise environments
- Focused on agentic development and building

---

# Why Apple Native

- Apple Foundation Models are available on 30+ Apple device models
- MLX framework allows developers to train models on Apple Silicon
- Privacy First Approach to data
- SwiftUI is a joy to work with
  
---

# How I Built the Beast

- Dev Time: 3 weeks to proof of concept
- Explore and Learn MLX framework
- Learn SwiftUI and xCode practices

---

# Building the MLX Models

- Collected actual notes from partner
- Built synthetic notes for comprehensive coverage 
- Python based workflow to process notes
- Initially models where deployed to HuggingFace for dynamic loading

---

# Data Processing: Notes → Models

1. **Extract & Annotate** - Parse raw notes for clinical findings, diagnoses, anatomical references, treatment codes

2. **Build Domain Vocabulary** - Create terminology mappings from actual practitioner language (not textbook terms)

3. **Convert to Training Tensors** - Format data for MLX: optimized vectors ready for neural network training

---

# Neural Network Training: MLX + Apple Silicon

1. **Input: Training Tensors** - From data processing step, ready for model training

2. **Metal-Accelerated Training** - Apple Silicon GPU optimization via MLX framework

3. **Output: 3 Specialized Models** - ICD-10, CPT, Vertebral Level ready for inference

---

# The Workflow: Dictation → Intelligence → Note

**Input:** Examination 

**Processing:** 3 MLX Models Run in Parallel (<50ms)
- ICD Classifier → M54.5, M99.03
- Vertebral Detector → L4-L5 lumbar segments
- CPT Classifier → 98940, 97110

**Output:** Complete SOAP Note with Codes
- Subjective/Objective: Clinical narrative
- Assessment: Diagnoses with ICD codes
- Plan: Procedures with CPT codes + treatment details

---

# Integration

![w:1000 h:480](image.png)

---

# What I Learned

- Generic models benefit from specialized training
- Modern hardware can handle complex NLP

---

# Contact

- Email: matthew@paz.land
- GitHub: https://github.com/mpazaryna


