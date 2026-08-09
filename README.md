# Camera-based Human Activity Recognition (HAR)
**Comparative analysis of Deep Learning approaches**

A research project from the Dept. of Computer Science and Media Technology at Malmö University evaluating the feasibility of deploying deep learning architectures for real-time human activity recognition on edge devices.

## 📌 Overview
The primary goal of this project is to classify atomic human activities (walking, running, jogging, boxing, handclapping, and hand waving) from RGB video inputs. Given that this model is intended for surveillance scenarios where computational power is scarce (e.g., Raspberry Pi, smartphones), the focus is heavily weighted toward **real-time feasibility, low latency, and computational efficiency (GFLOPS)**.
## Project Overview
The primary objective of this project is to classify atomic human activities (e.g., walking, standing, sitting, running, boxing, hand-clapping) from RGB video inputs using deep learning computer vision architectures. 

A specific focus of this research is **real-time feasibility** for deployment on edge devices (such as Raspberry Pi or smartphones) where computational resources are scarce. 

### Research Questions
1. Which deep learning architecture—3D CNN, RNN/LSTM, or Video Vision Transformer—achieves the highest accuracy on atomic activity classification from RGB video?
2. How do these architectures compare in terms of computational efficiency and inference time under real-time constraints?
3. What is the trade-off between recognition accuracy and computational cost across the evaluated models?

---

## Dataset
This project utilizes the **KTH Action Recognition Dataset** (2004), chosen for its minimal background noise and single-subject focus.
* **Classes (6):** Walking, running, jogging, boxing, handclapping, handwaving.
* **Format:** 600 videos (2391 sub-clips), 25 FPS, black and white.
* **Sampling:** 16 frames per sample were utilized for baseline evaluations to optimize the spatial-temporal trade-off.

---

## Model Architectures Evaluated
To understand the trade-off between speed and accuracy, three distinct architectural paradigms were trained and evaluated:

1. **2D CNN + LSTM (EfficientNet-B0 + LSTM)** 
   * *Concept:* Extracts spatial features per frame using a lightweight EfficientNet-B0 backbone, then models temporal dependencies using an LSTM.
   * *Advantage:* Highly lightweight, lowest computational footprint.
2. **Separable 3D CNN (S3D)** 
   * *Concept:* Replaces standard 3D convolutions with separable convolutions (spatial 2D followed by temporal 1D), pretrained on Kinetics-400.
   * *Advantage:* Excellent balance of spatio-temporal feature extraction with moderate parameter counts.
3. **Multiscale Vision Transformer (MViT v2 - Small)**
   * *Concept:* Uses patch embedding and self-attention mechanisms to dynamically capture global context and long-range dependencies across space and time.
   * *Advantage:* State-of-the-art capability, though extremely computationally expensive.

---

## Experimental Setup
* **Hardware:** NVIDIA A100 GPU
* **Environment:** Python v3.12.0, PyTorch v2.11.0, Torchvision v0.26.0
* **Training:** 30 epochs, 80/20 train-test split, 16-frame context length across all models.

---

## Results & Performance
The models were evaluated based on their accuracy, parameter count, GFLOPS, and inference latency.

| Model | Accuracy | Parameters | GFLOPS | Inference Time (ms)* |
| :--- | :--- | :--- | :--- | :--- |
| **EfficientNet-B0 + LSTM** | 86.0% | 5,584,002 | **6.41** | **5.72** |
| **S3D** | **96.5%** | 8,320,048 | 18.20 | 7.29 |
| **MViT v2 (Small)** | 92.7% | 34,537,744 | 64.46 | 23.75 |

*\*Inference time of one forward pass of a single sample using an Nvidia A100.*

### Takeaways
* **Best performance (S3D):** The S3D model represents the optimal sweet spot for real-time video surveillance. It achieved the highest accuracy (96.5%) with a highly manageable computational cost (18.2 GFLOPS / 7.29 ms inference).
* **Lightweight Option (EfficientNet + LSTM):** While it struggled slightly with fine-grained spatial-temporal dependencies (often confusing jogging with running), its incredibly low 6.41 GFLOPS makes it the only viable choice for extreme power-constrained edge deployments.
* **Heavyweight Option (MViT v2):** Despite strong accuracy, the Transformer-based model requires 64.46 GFLOPS (10x that of the CNN-LSTM). This massive architectural footprint makes it prone to thermal throttling and fundamentally incompatible with low-latency edge deployment.

---

## Future Work
* **Anomaly Detection:** Transitioning the classification task to anomaly detection, where KTH classes act as normal behavior and outliers trigger alerts.
* **Custom Dataset Collection:** Developing a modern dataset with higher resolutions, controlled pale backgrounds, and consistent camera angles to remove legacy dataset limitations.
* **Frame Shuffling:** Testing temporal reliance by shuffling input frames to measure accuracy decay, particularly on the Transformer models.
