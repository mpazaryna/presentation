# **JST*D Lightning Talk: From Dictation to Documentation**  
## **Building On-Device AI for Healthcare with Apple's MLX Framework**  
  
**Date:** November 12, 2024  
**Duration:** 7-8 minutes  
**Theme:** "How I Use AI to Get Work Done"  
  
---
## **Slide 1: Title**  
### **From Dictation to Documentation: Building On-Device AI for Healthcare**  
**How I Use Apple's MLX Framework to Create Privacy-First Clinical Tools**  
  
*Speaker Notes:* Welcome to my lightning talk about building on-device AI for healthcare documentation using Apple's MLX framework. Today I'll show you how I created a privacy-first solution for chiropractic SOAP notes.  

---   
## **Slide 2: About Me**  
  
- **Freelance AI Developer & Consultant**  
- **Specializing in on-device AI with Apple Silicon & MLX**  
- **Building domain-specific AI tools for healthcare**  
- **Focus on privacy-first, HIPAA-compliant solutions**  
  
**Current Project:** Chiropractic SOAP notes automation system  
  
*Speaker Notes:* I'm a freelance AI developer specializing in on-device solutions. My current focus is on building domain-specific healthcare tools that prioritize patient privacy.  

--- 

## **Slide 3: The Documentation Burden**  
  
### **The Numbers**  
- **40%** of time spent on documentation  
- **15-20** minutes per patient on SOAP notes  
- **$200+** monthly for cloud-based solutions  
  
### **Current Solutions Fall Short**  
- **Cloud dependency** - Patient data leaves the device  
- **Privacy concerns** - HIPAA compliance complexity  
- **Generic outputs** - Not tailored for chiropractic  
- **Latency issues** - Internet connection required  
- **Ongoing costs** - Subscription fees add up  
  
**The Challenge:** How do we use AI to solve this while keeping patient data completely private?  
  
*Speaker Notes:* Healthcare providers spend an enormous amount of time on documentation. Current cloud-based solutions raise privacy concerns and come with ongoing costs. The challenge is creating an AI solution that keeps data completely private.  

---

## **Slide 4: Why MLX + Apple Silicon?**  
  
### **The On-Device Advantage**  
  
**Privacy First**  
- Patient data never leaves the device  
- Complete HIPAA compliance simplified  
  
**Zero Latency**  
- Instant processing with no internet required  
- Works everywhere, always  
  
**Cost Effective**  
- No API fees or subscriptions  
- One-time setup, unlimited use  
  
**Optimized Performance**  
- MLX leverages Apple's Neural Engine  
- Blazing-fast AI inference  
  
### **Performance Comparison**  
- **On-Device:** < 2 seconds  
- **Cloud:** 3-8 seconds + latency  
  
*Speaker Notes:* MLX and Apple Silicon provide the perfect combination for on-device AI. We get privacy by default, zero latency, no ongoing costs, and optimized performance using the Neural Engine.  
  
---

## **Slide 5: How I Built It**  
  
### **Development Workflow**  
  
1. **Data Preparation**  
    - Collected anonymized SOAP note examples from partner clinics  
2. **Model Selection**  
    - Tested various Apple Foundation Models for medical text understanding  
3. **Fine-tuning with MLX**  
    - Specialized training on chiropractic terminology and ICD-10 codes  
4. **Optimization**  
    - Quantized models for optimal M1/M2/M3 performance  
5. **Integration**  
    - Built a native macOS app  
  
### **Tech Stack**  
- **Framework:** MLX  
- **Models:** Apple Foundation Models  
- **Languages:** Python + Swift  
- **Voice:** Whisper MLX  
- **Platform:** macOS Native  
- **Hardware:** M1/M2/M3 chips  
- **Dev Time:** 3 weeks to MVP  
  
```
# Example: Loading model with MLX
model = mlx.load_model("chiro-soap-v2.mlx")
output = model.generate(voice_transcript)

```
  
  
*Speaker Notes:* The development process involved data preparation with anonymized SOAP notes, model selection from Apple Foundation Models, fine-tuning with MLX for chiropractic terminology, optimization for Apple Silicon, and integration into a native macOS app.  
  
---

## **Slide 6: Demo - Patient Encounter to SOAP Note**  
  
### **Workflow Demonstration (3-minute recorded demo)**  
  
1. **Voice Input (30 seconds)**  
    - "43-year-old male presents with acute onset lower back pain, started three days ago after lifting boxes. Pain is sharp, 7 out of 10, radiating down the left leg to the knee. Aggravated by forward flexion and prolonged sitting."  
  
2. **AI Processing (< 2 seconds)**  
    - Model identifies clinical entities  
    - Recognizes pain characteristics and radicular component  
    - Structures information into SOAP format  
  
3. **SOAP Note Generation**  
    - **S (Subjective):** Patient complaints and history formatted  
    - **O (Objective):** Examination findings structured  
    - **A (Assessment):** Diagnosis with ICD-10 codes (M54.5, M54.41)  
    - **P (Plan):** Treatment recommendations generated  
  
4. **Editing & Export**  
    - Inline editing capabilities  
    - Medical terminology validation  
    - Direct EHR integration  
  
*Speaker Notes:* Here's a recorded demo showing the complete workflow from voice input to finished SOAP note with ICD-10 codes, all processed locally in under 2 seconds.  
  
---

## **Slide 7: Real-World Impact**  
  
### **The Results**  
  
- **⏱️ 20 minutes** saved per patient  
- **🎯 95%** coding accuracy  
- **💵 $0** monthly fees  
- **🔐 100%** on-device privacy  
  
### **Partner Clinic Case Study**  
  
> "We've reduced documentation time by 30% across the practice. The ICD-10 coding accuracy has eliminated billing rejections. Most importantly, our patients' data never leaves our control."  
> 
> — Dr. John, Partner Chiropractor  
  

**ROI Achievement:** System paid for itself in **6 weeks** through time savings alone  
  
*Speaker Notes:* The real-world impact has been significant: 20 minutes saved per patient, 95% coding accuracy, zero monthly fees, and 100% on-device privacy. Our partner clinic saw a 30% reduction in documentation time.  
  
---

## **Slide 8: Key Takeaways**  
  
### **What I Learned Using AI This Way**  
  
1. **Domain Expertise Matters**  
    - Generic models need specialized training for healthcare applications  
  
2. **On-Device AI is Production-Ready**  
    - Modern hardware can handle complex NLP without relying on the cloud.  
  
3. **Privacy Enables Adoption**  
    - Healthcare providers prefer local processing for compliance and trust.  
  
4. **Real Users Drive Innovation**  
    - Iterative refinement with practitioner feedback is crucial for success.  
  
### **Get Started with On-Device AI**  
- **MLX Framework:** github.com/ml-explore/mlx  
- **Apple Foundation Models:** Available through MLX  
- **Development:** Python   
- **Timeline:** 2-3 weeks to MVP  
  
*Speaker Notes:* The key lessons: domain expertise matters, on-device AI is production-ready, privacy enables adoption in healthcare, and real user feedback drives innovation. You can build similar solutions in 2-3 weeks.  

---

## **Slide 9: Thank You & Questions**  
  
### **Thank You!**  
  
**Let's Discuss:**  
- MLX Optimization Techniques  
- Healthcare AI Applications  
- On-Device Deployment Strategies  
  
**Contact:**  
- Email: mpazaryna@gmail.com  
- GitHub: https://github.com/mpazaryna  
  
### **Key Takeaway**  
> "AI doesn't have to live in the cloud to be powerful"  
  

*Speaker Notes:* Thank you for your attention! I'm happy to discuss MLX optimization, healthcare AI applications, or on-device deployment strategies. Remember: AI doesn't have to live in the cloud to be powerful.  
