# 🧵 Fabric Pattern Classification using CLIP-ViT

This project classifies segmented clothing images into fabric pattern categories using a fine-tuned Vision Transformer model (`CLIP-ViT`). The pipeline includes custom dataset creation, model training, and image segmentation for real-world testing.

---

## 📷 Pipeline Overview

1. **Data Collection:**  
   Images for each fabric class were scraped using [`icrawler`](https://github.com/hellock/icrawler), and organized into folders per class label.

2. **Dataset Structure:**  
   The dataset uses the PyTorch `ImageFolder` format:

3. **Model Training:**  
A CLIP Vision Transformer (`openai/clip-vit-base-patch32`) was fine-tuned directly for fabric pattern classification across multiple classes using cross-entropy loss.

4. **Segmentation:**  
During inference, the cloth region is segmented from the full image using SAM(Segment Anything Model) and GroundingDINO and then classified.

5. **Prediction:**  
The model outputs the **Top-3 predicted fabric classes** with confidence scores.

**Sample Outputs**

![image](https://github.com/user-attachments/assets/9836531f-78af-440e-b535-a717bbbbbea4)
![image](https://github.com/user-attachments/assets/16805d5a-2e5c-4701-a454-d254ae11b67f)

![image](https://github.com/user-attachments/assets/3accf506-0ba0-4f6f-93d9-c2b874dda6c3)
![image](https://github.com/user-attachments/assets/643c3136-1195-4d90-9cbc-dfd7c3989870)


## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install torch torchvision transformers numpy pillow scikit-learn

