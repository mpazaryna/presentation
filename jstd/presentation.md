---
marp: true
theme: default
---

# Private Intelligence for Modern Practices

- Building solutions that don't depend on frontier models

---

# About Me

- Independent Developer 
- I've worked over 35 years in enterprise development and start ups
- Currently focused on optimizing agentic development 

---

# What I Built

- Apple Native Application
- Available on iOS, ipadOS and MacOS
- Depends on Apple Foundation Models, on device AI 
- Domain Independent Approach

---

# Why Apple Native

- Apple Foundation Models are available on 30+ devices
- MLX Framework allows models to be trained on Apple Silicon
- HIPPA rules require a privacy first approach 
  
---

# How I Built the application

- Learn MLX 
- Learn SwiftUI
- Learn xCode
- Gather clinical notes and source materials

---

# Collecting Source Materials

- **Collect** - Practitioner notes 
- **Build** - Synthetic notes to extend clinical range

---

# Process Source Materials

- **Extract** - Parse notes for findings, anatomical references, treatment codes
- **Transform** - Create terminology mappings from practitioner language 
- **Output** - Tensors aka optimized vectors

---

# Building Models

- **Input** - Tensors 
- **Transform:** - Using the MLX framework
- **Output:** - Custom specialized models 

---

# Application Workflow: Exam

**Input:** Examination 
**Voice:** Versions of Whisper is HIPPA compliant
**Keyboard:** On device

---

# Application Workflow: Exam → Process

- Models Run in Parallel (<50ms)
- ICD Classifier detects codes M54.5, M99.03
- CPT Classifier detects code 98940, 97110
- Vertebral Detector detects L4-L5 lumbar segments

---

# Application Workflow: Exam → Process → Output

- Subjective: Clinical narrative
- Assessment: Diagnoses with ICD codes
- Plan: Procedures with CPT codes + treatment details

---

# Integration

![w:1000 h:480](image.png)

---

# What I Learned

- Generic models and specialized training are powerful
- Modern hardware can handle complex natural language processing
- No Swift packages but elegant architecture to maintain domain independence
- Refactor into official Swift packages at a latter time 

---

# Contact

- Email: matthew@paz.land
- GitHub: https://github.com/mpazaryna