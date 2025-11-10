# JSTD: Private Intelligence for Modern Practices

Presenter notes and study guide for the JSTD presentation.

---

## Understanding .safetensors

**What it is:**
- Binary file format for storing neural network model weights (parameters)
- Developed by Hugging Face as a safer alternative to Python's pickle format
- Industry standard for distributing trained models

**Why "safe":**
- Cannot execute arbitrary code (unlike pickle)
- Protects against malicious model files
- Safe for loading models from untrusted sources

**In JSTD's workflow:**
1. Train 3 MLX models on Apple Silicon � produces weights
2. Export weights to `.safetensors` format (safe, standardized)
3. Bundle `.safetensors` files with native macOS app
4. Load models at runtime for inference

**Dual format (.npz + .safetensors):**
- `.npz` = NumPy native format (development/training)
- `.safetensors` = Standardized format (deployment/distribution)
- Both formats allow flexibility depending on context

---

## Concept Map: JSTD Presentation Flow

### Theme: Private Intelligence for Modern Practices
**Core idea:** Building solutions that don't depend on frontier models

---

### 1. **Foundation: Why Apple Native?**
   - **30+ Apple device models** support Foundation Models
   - **MLX framework** enables training on Apple Silicon
   - **Privacy-first** approach (data stays on device)
   - **SwiftUI** developer experience is excellent

   **Key concept:** Apple Silicon is the ideal platform for on-device ML

---

### 2. **Development Context: Building in 3 Weeks**
   - Dev timeline: 3-week MVP sprint
   - Three learning curves: MLX, SwiftUI, Xcode
   - Starting point: Real clinical data from partner

   **Key concept:** Fast iteration possible with right tools + clear domain focus

---

### 3. **Data Pipeline: Raw Notes � Model-Ready Tensors**

   **Step 1: Extract & Annotate**
   - Parse clinical findings from raw notes
   - Identify diagnoses, anatomical references, treatment codes
   - Preserve practitioner language patterns

   **Step 2: Build Domain Vocabulary**
   - Map actual terms used (not textbook definitions)
   - Create vocabulary from 72+ real + synthetic notes
   - This vocabulary is domain-specific, not generic

   **Step 3: Convert to Training Tensors**
   - Format data for MLX consumption
   - Optimize vectors for neural network training
   - Output: Ready-to-train tensors

   **Key concept:** Quality training data comes from understanding real practitioner language

---

### 4. **Model Training: MLX + Apple Silicon**

   **Input � Process � Output**
   - **Input:** Training tensors from data processing
   - **Process:** Metal-accelerated training on Apple Silicon GPU
   - **Output:** 3 specialized models ready for inference

   **The 3 Models:**
   - ICD-10 Classifier: Diagnoses (M54.5, M99.03)
   - CPT Classifier: Billing codes (98940, 97110)
   - Vertebral Detector: Spinal level localization (L4-L5)

   **Key concept:** Models are trained from scratch (not fine-tuned), optimized for this specific practice

---

### 5. **The Workflow: Dictation � Intelligence � SOAP Note**

   **Real-time inference pipeline:**

   **Input:** Doctor dictates exam findings
   - Example: "Patient with acute lower back pain, positive SLR left, tenderness at L4-L5"

   **Processing:** 3 models run in parallel (<50ms)
   - ICD Model � M54.5, M99.03
   - Vertebral Model � L4-L5 segments
   - CPT Model � 98940, 97110

   **Output:** Complete SOAP Note
   - Subjective/Objective: Clinical narrative
   - Assessment: Diagnoses with ICD codes
   - Plan: Procedures with CPT codes + treatment

   **Key concept:** Parallel processing keeps latency under 50ms; Foundation Models enrich narrative quality

---

### 6. **Integration: Native macOS App**
   - SwiftUI interface
   - Bundled .safetensors models
   - Zero cloud dependency
   - HIPAA-compliant on-device inference

   **Key concept:** Technology is only valuable when it's usable in real workflows

---

### 7. **Real-World Performance: Proof of Concept**
   - **Model Loading:** 34ms (all 3 at startup)
   - **Fresh Models:** Training date 2025-11-10
   - **Inference Times:** 14ms (ICD), 3ms (CPT), 2ms (Vertebral)
   - **Predictions:** 8 ICD codes, 2 CPT codes, 8 vertebral levels
   - **Architecture:** Pure MLX, no fallbacks

   **Key concept:** Sub-100ms end-to-end is achievable with specialized models on Apple Silicon

---

### 8. **Key Learnings**
   - Generic models benefit from specialized domain training
   - Modern Apple Silicon hardware can handle real NLP complexity
   - Privacy-first architecture enables enterprise adoption

   **Key concept:** The future isn't "bigger models in the cloud"it's specialized intelligence on-device

---

## Presenter Notes by Slide

1. **Opening:** Anchor the "no frontier models" messagethis is about control and independence
2. **About Me:** Establish credibility (35 years) but frame as "learning new tools" (humble, relatable)
3. **Why Apple Native:** Plant the seedwhy did I choose this platform? Hint at advantages
4. **How I Built It:** Emphasize speed (3 weeks!) and learning curve (these tools are doable)
5. **Building Models:** Foreshadow the data pipelinereal notes are crucial
6. **Data Processing:** This is where magic happensexplain domain vocabulary matters
7. **Model Training:** Clear input � process � output; emphasize "from scratch"
8. **The Workflow:** This is the user experiencekeep it concrete with the example
9. **Integration:** Show the app exists and works (image)
10. **Real-World Performance:** Metrics prove it's not theoretical
11. **What I Learned:** Tie back to opening themespecialized < frontier, but sufficient

---

## Common Questions to Anticipate

**Q: Why not just use ChatGPT/Claude?**
A: HIPAA compliance prevents sending Protected Health Information (PHI) to third-party cloud services. Frontier models require cloud calls. These specialized models run 100% on-device, keeping patient data private and HIPAA-compliant.

**Q: How do you handle new codes/patterns?**
A: Models are trained on your specific practice data. They learn YOUR patterns, not generic ones.

**Q: Can this scale to other practices?**
A: Yesretraining pipeline is the same, just need domain data from that practice.

**Q: What about HIPAA?**
A: 100% on-device means zero data transmission. This is the HIPAA-optimal approach.

---

## Technical Depth Reference

If audience is technical, be ready to discuss:
- MLX framework architecture (lazy evaluation, Metal backend)
- Tensor shapes and training optimization
- Model inference latency breakdown
- .safetensors vs .npz trade-offs
- Foundation Model integration for narrative enrichment
