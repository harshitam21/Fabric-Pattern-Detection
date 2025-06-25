# 🧵 Fabric Pattern Classification using CLIP-ViT

This project classifies segmented clothing images into fabric pattern categories using a fine-tuned Vision Transformer model (`CLIP-ViT`). The pipeline includes custom dataset creation, model training, and image segmentation for real-world testing.

---

## 📷 Pipeline Overview

1. **Data Collection:**  
   Images for each fabric class were scraped using [`icrawler`](https://github.com/hellock/icrawler), and organized into folders per class label.

2. **Dataset Structure:**  
   The dataset uses the PyTorch `ImageFolder` format:
