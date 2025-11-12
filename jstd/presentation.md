---
marp: true
theme: default
---

# Private Intelligence for Modern Practices

- Building solutions that don't depend on frontier models

---

# About Me

- Independent Developer 
- 35 years experience in full stack development
- Focused on agentic software development - rebuilding my toolkit 

---

# What I Built

- Medical notation application for chiropractors
- Best Practices Apple Native Application
- Specs, Unit Tests, End to End Tests 
- Reliance on Apple Foundation Model and MLX Framework
- No Cloud

---

# Why Apple Native

- Apple Foundation Models are available on 30+ devices
- MLX Framework allows models to be trained on Apple Silicon 
- Partner asked 'can we make this more star trek like'
  
---

# How I Built It

- Learn MLX 
- Learn SwiftUI
- Learn xCode

---

# Collecting Materials and Build Workflows

- **Collect** - Practitioner notes 
- **Build** - Synthetic notes
- **Extract** - Parse notes for findings, anatomical references, treatment codes
- **Transform** - Create terminology mappings from practitioner language 
- **Output** - Tensors aka optimized vectors

---

# Extracting Data

```python
def extract_icd_codes(text):
    """
    Extract ICD-10 codes from ASSESSMENT section.

    Handles formats:
    - Neck pain — M54.2
    - ● Description | CODE
    - M54.2 Cervicalgia
    """
    codes = []

    # Pattern 1: Bullet format with — or | separator
    # ● Description — M54.2 or ● Description | M54.2
    bullet_pattern = r'[●•]\s*[^—|]+[—|]\s*([A-Z]\d{2}\.?\d+[A-Z]?)'
    matches = re.findall(bullet_pattern, text)
    for code in matches:
        # Filter out vertebral levels (C1-C7, T1-T12, L1-L5, S1-S5)
        if not re.match(r'^[CTLS]\d{1,2}$', code):
            codes.append({'code': code, 'description': ''})

    # Pattern 2: Traditional format (M54.2 Description)
    if not codes:  # Only use if bullet pattern didn't find anything
        traditional_pattern = r'\b([A-Z]\d{2}\.\d+[A-Z]?)\s+([^\n]+)'
        matches = re.findall(traditional_pattern, text)
        for code, description in matches:
            # Filter out vertebral levels
            if not re.match(r'^[CTLS]\d{1,2}', code):
                codes.append({'code': code, 'description': description.strip()})

    return codes
```

---

# Building Models

- **Input** - Tensors 
- **Transform:** - Using the MLX framework
- **Output:** - Custom specialized models 

---

# Process: 700-900ms End-to-End

- Phase 1: MLX Analysis (50-100ms) - Run all 3 processors in parallel, extract codes with confidence scores
- Phase 2: Apple Intelligence (600-900ms) - Enhanced system instructions with MLX results → structured JSON SOAP note

---

# Run MLX Processors in Parallel

```swift
private func runMLXAnalysis(on text: String) async -> (
    icd10: [ICD10Suggestion],
    vertebral: VertebralAnalysisResult,
    cpt: CPTAnalysisResult
) {
    async let icd10Task = icd10Processor.processWithMLX(text)
    async let vertebralTask = vertebralProcessor.processWithMLX(text)
    async let cptTask = cptProcessor.processWithMLX(text)

    let (icd10Results, vertebralResults, cptResults) = await (icd10Task, vertebralTask, cptTask)

    return (icd10: icd10Results, vertebral: vertebralResults, cpt: cptResults)
}
```

---

# Application Workflow: Exam

**Input:** Examination 
**Voice:** Wisper with HIPPA compliance
**Keyboard:** On device

---

# Application Workflow: Exam → Process

- Models Run in Parallel (<50ms)
- ICD Classifier detects codes M54.5, M99.03
- CPT Classifier detects code 98940, 97110
- Vertebral Detector detects L4-L5 lumbar segments

---

# Application Workflow: Exam → Process → Output

- Examination: type or voice input 
- Process: Models Run in Parallel (<50ms)
- Report: Apple Foundation Models build the final report
- Subjective: Clinical narrative
- Objective: measurable data and observations
- Assessment: clinician's professional judgment
- Plan: outlines the next steps for treatment. 

---

# Integration

![](image.png)

---

# What I Learned

- Generic models and specialized training are powerful
- Modern hardware can handle complex natural language processing
- Build an elegant architecture to maintain domain independence
- Refactor into Swift packages when I'm more confident 
- 150 hours of desk work flies by

---

# Contact

- Email: matthew@paz.land
- GitHub: https://github.com/mpazaryna
