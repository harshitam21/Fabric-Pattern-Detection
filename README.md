# Fabric Pattern Classification using CLIP-ViT

This project classifies clothing fabric patterns from images using a fine-tuned Vision Transformer model (`CLIP-ViT`). It supports **multi-class classification** with real-world image input, enhanced by **image segmentation** to isolate clothing regions. The pipeline includes data collection, model training, segmentation with SAM & GroundingDINO, and top-3 fabric class predictions with confidence scores.

---

## Project Pipeline

### 1. Data Collection

Images were collected for each fabric pattern class (e.g., *floral*, *stripes*, *plaid*, etc.) using [`icrawler`](https://github.com/hellock/icrawler), a Python-based image crawler. Each class is stored in its respective folder.


### 2. Model Architecture

We fine-tuned `openai/clip-vit-base-patch32` from HuggingFace's Transformers library:

* Replaced CLIP’s zero-shot classification head with a custom classification head.
* Trained using **Cross Entropy Loss** across fabric pattern classes.
* Optimized with `AdamW` and a linear learning rate scheduler.

### 3. Segmentation for Inference

Before classification, the clothing region is segmented from full-body or complex images using:

* **[Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)**
* **[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)**

This ensures only the fabric region is passed to the classifier, improving real-world robustness.

### 4. Prediction Output

* Top-3 predicted fabric classes with confidence scores.
* Suitable for both **clean datasets** and **real-world fashion images**.


## Sample Outputs
  <img src="https://github.com/user-attachments/assets/9836531f-78af-440e-b535-a717bbbbbea4" >
  <img src="https://github.com/user-attachments/assets/16805d5a-2e5c-4701-a454-d254ae11b67f" >
  <img src="https://github.com/user-attachments/assets/6b08f245-3196-456c-8d32-b2762684616e" >
  <img src="https://github.com/user-attachments/assets/ed37ebdc-f0a2-4f9f-a505-2d17addbafdf" >

---

## Dependencies

* `torch`, `torchvision`
* `transformers` (HuggingFace)
* `datasets`, `evaluate`
* `icrawler`
* `Pillow`, `numpy`
* `Ultralytics`
* `GroundingDINO` (see the official repo for installation)


## Evaluation Metrics

* **Accuracy**
* **Precision, Recall, F1-Score**
* Optionally: per-class breakdown using `sklearn.metrics.classification_report`



## Use Cases

* Fashion e-commerce filtering
* Visual search by fabric type
* Apparel metadata tagging
* Fashion trend forecasting inputs



## Future Work

* Add support for **multi-label prediction** (e.g., floral + lace)
* Extend to detect **fabric texture types** (e.g., chiffon, denim)
* Improve segmentation pipeline with human pose estimation
* Addition of more diverse classification labels

## Acknowledgements

* [OpenAI CLIP](https://github.com/openai/CLIP)
* [HuggingFace Transformers](https://huggingface.co)
* [Segment Anything (Meta AI)](https://segment-anything.com/)
* [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

---

## Author

**Harshita Manocha**

For questions or collaborations: `harshitamano2005@gmail.com`
