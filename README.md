# 🛒 Customer Age Estimation CV

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![ComputerVision](https://img.shields.io/badge/CV-ResNet50-red)

> Deep learning regression pipeline for retail checkout systems, estimating customer age from facial images with high accuracy using transfer learning.

## 📖 Project Overview
The **"Khleb-Sol"** supermarket chain is deploying a computer vision system at checkout counters to automatically estimate customer age. This capability supports:
- 🎯 **Personalized Marketing:** Tailoring product recommendations based on demographic insights.
- 🛡️ **Regulatory Compliance:** Verifying customer age for restricted sales (e.g., alcohol).

**Primary Goal:** Build a regression model to predict age from facial photos with **MAE ≤ 8 years**.

## 📊 Dataset
| Column | Description |
|:-------|:------------|
| `file_name` | Path to facial image (`.jpg`) |
| `real_age` | Ground-truth age (integer) |

**Size:** ~7,591 labeled images  
**Characteristics:**
- Diverse demographics, nationalities, and facial features
- Varied lighting, angles, and partial occlusions (glasses, hats, facial hair)
- Bimodal age distribution (peaks at `<7` and `22–33` years)
- Preprocessed & normalized for CNN input

## 🛠 Methodology & Pipeline
1. **Data Preparation & Augmentation**  
   - Loaded via Keras `ImageDataGenerator` with `preprocess_input` (ResNet50 standard)
   - Rescaled to `224×224` with stratified train/test split (80/20)
2. **Model Architecture**  
   - **Backbone:** Pretrained `ResNet50` (ImageNet weights, top excluded)
   - **Head:** Custom fully-connected regression layers (`GlobalAveragePooling2D → Dense → Linear output`)
   - **Optimization:** `Adam` optimizer, MSE loss, early stopping/callbacks
3. **Training & Validation**  
   - Fine-tuned end-to-end on the facial dataset
   - Monitored validation MAE to prevent overfitting to the bimodal distribution

## 📈 Results & Model Performance
| Metric | Requirement | Achieved |
|:-------|:-----------:|:--------:|
| **MAE (Test)** | ≤ 8.0 | **< 7.0** ✅ |
| Architecture | Custom CNN / ResNet | `ResNet50 + Regression Head` |

✅ The model successfully meets the business threshold, delivering reliable age estimates suitable for production deployment in retail environments.

## 💡 Key Insights & Business Recommendations
🔍 **Data Challenges:** The bimodal age distribution creates inherent variance in prediction error across age groups. Transfer learning mitigates this by leveraging robust facial feature extractors.  
📋 **Actionable Recommendations:**
1. **Deploy at Edge/Server:** Run inference on checkout cameras with batched processing to minimize latency.
2. **Continuous Calibration:** Periodically retrain with new demographic data to adapt to shifting customer profiles.
3. **Privacy-First Design:** Ensure images are processed ephemeral and comply with local data protection regulations (GDPR/152-ФЗ).

## 🚀 Getting Started
```bash
# 1. Clone repository
git clone https://github.com/your-username/Customer-Age-Estimation-CV.git
cd Customer-Age-Estimation-CV

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter Notebook
jupyter notebook notebooks/customer_age_estimation.ipynb
