🕵️‍♂️ Deepfake & AI-Generated Image Detector
=============================================

> **A robust computer vision pipeline capable of distinguishing between authentic photographs and AI-generated synthetic imagery (Stable Diffusion) with high confidence.**

📖 Project Overview
-------------------

The rise of generative adversarial networks (GANs) and diffusion models has made it increasingly difficult to distinguish real images from deepfakes. This project addresses the **digital authenticity crisis** by treating deepfake detection as a supervised binary classification problem.

By leveraging **Transfer Learning (ResNet18)** and **Explainable AI (Grad-CAM)**, this tool not only predicts if an image is Real or AI-generated but also visualizes _why_ the decision was made by highlighting the specific artifacts in the image.

### ✨ Key Features

*   **High-Accuracy Detection:** Utilization of a pre-trained ResNet18 backbone fine-tuned on 100,000+ images.
    
*   **Visual Explainability:** Integrated **Grad-CAM (Class Activation Mapping)** to generate heatmaps showing which pixels influenced the model's decision.
    
*   **Modern User Interface:** A clean, responsive web dashboard built with **Streamlit** for easy testing.
    
*   **Confidence Scoring:** Provides probability metrics for every prediction.
    

🏗️ Architecture & Methodology
------------------------------

This project moves beyond simple classification by implementing a rigorous pipeline:

1.  **Data Ingestion:** Usage of the **CIFAKE dataset** (60k Real / 60k Fake).
    
2.  **Preprocessing:** Normalization using ImageNet standards (mean=\[0.485, 0.456, 0.406\], std=\[0.229, 0.224, 0.225\]).
    
3.  **Model Training:**
    
    *   **Backbone:** ResNet18 (Frozen early layers for feature extraction).
        
    *   **Head:** Custom Fully Connected layers with Dropout for binary classification.
        
    *   **Optimizer:** Adam.
        
    *   **Loss Function:** CrossEntropyLoss.
        
4.  **Inference:** Deployment via Streamlit for real-time analysis.
    

🚀 Installation & Setup
-----------------------

### Prerequisites

*   Python 3.9+
    
*   CUDA capable GPU (Optional, but recommended for training)
    

### 1\. Clone the Repository

Bash

`git clone https://github.com/utkarshmishra2507/deepfake-detector.git  cd deepfake-detector   `

### 2\. Install Dependencies

Bash
`   pip install -r requirements.txt   `

### 3\. Download the Model

_If you haven't trained the model yet, run the training notebook notebooks/02\_Training.ipynb to generate deepfake\_resnet.pth._Place the .pth file inside the root directory or models/ folder.

### 4\. Run the Application

Bash

`   streamlit run app.py   `

The application will launch in your browser at http://localhost:8501.

📊 Performance & Results
------------------------

*   **Training Accuracy:** ~89%
    
*   **Validation Accuracy:** ~88%
    
*   **F1-Score:** 0.89
    

_Note: Metrics may vary slightly based on hyperparameter tuning and random seeds._

### Visual Explanation (Grad-CAM)

The model looks for specific high-frequency artifacts often found in diffusion models:

*   **AI Artifacts:** Over-smoothed textures, unnatural lighting gradients, and aliasing in eyes/hair.
    
*   **Real Features:** Natural sensor noise and consistent depth of field.
    

🛠️ Technologies Used
---------------------

*   **Core Logic:** Python
    
*   **Deep Learning:** PyTorch, Torchvision
    
*   **Data Processing:** NumPy, Pandas
    
*   **Computer Vision:** OpenCV (cv2), PIL
    
*   **Visualization:** Matplotlib, Seaborn
    
*   **Frontend:** Streamlit
    

🤝 Acknowledgements
-------------------

*   **Dataset:** [CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) 
    
*   **Model:** ResNet architecture by Microsoft Research.
    

📜 License
----------

This project is licensed under the **MIT License**. Feel free to use it for research and educational purposes.

### 👤 Author

**Utkarsh**

*   [https://github.com/utkarshmishra2507](https://github.com/utkarshmishra2507)
    
*   [https://www.linkedin.com/in/utkarsh-mishra25/](https://www.linkedin.com/in/utkarsh-mishra25/)
