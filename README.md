# Radiology Submission

This repository contains the code for **CT segmentation, pMRI segmentation, and multimodal registration** with pulmonary disease analysis.

### Folder Structure:
- `CT_seg/` – Code for CT segmentation
- `MR_seg/` – Code for pMRI segmentation
- `Registration_and_Analysis/` – Code for multimodal registration and pulmonary disease analysis (our main contribution)

## Multimodal Registration Model
The multimodal registration model is implemented in **`LungReg.py`**, which is adapted from the following computer vision models:
- [Unsupervised Deformable Image Registration with U-Net](https://arxiv.org/abs/1807.03361)
- [Weakly-Supervised 3D Registration for Multi-modal Images](https://arxiv.org/abs/1907.12353)

## Inference Pipeline
1. **Run inference**  
   - `infer_ct2mr_all.py` (CT → pMRI)  
   - `infer_xe2mr_all.py` (XeMRI → pMRI)

2. **Apply dense displacement fields**  
   - `apply_ddf_ct_all.py` (Apply CT deformation)  
   - `apply_ddf_xe_all.py` (Apply XeMRI deformation)

## Analysis
- `lobe_analysis.py` – Lobe-wise analysis of registered images  
- `longituidinal_warped2Antspy.py` – Longitudinal registration processing  
- `rbctp_check_v5.py` – RBC:TP value metric evaluation  

---

This README now has clearer structure and improved readability while keeping the technical details intact. Let me know if you'd like any further refinements! 🚀
