# Indian YouTube Virality Prediction Engine 🚀

*An end-to-end machine learning project to classify the virality archetype of a YouTube video in the Indian market and forecast its 7-day view performance using just its first-day stats. The system is deployed as an interactive web application with Streamlit.*

---

### **Live Application Screenshot**

![Streamlit App Screenshot](https://github.com/Sourabh1710/Youtube-Virality-Prediction-Engine/blob/main/charts/Streamlit%20Upper%20Half.png)
![Streamlit App Screenshot](https://github.com/Sourabh1710/Youtube-Virality-Prediction-Engine/blob/main/charts/Streamlit%20Lower%20Half.png)


---

### Table of Contents
* [Problem Statement](#problem-statement)
* [Key Features & Solution](#key-features--solution)
* [Tech Stack](#tech-stack)
* [Project Pipeline](#project-pipeline) 
* [Key Results & Insights](#key-results--insights)
* [How to Run This Project](#how-to-run-this-project)
* [Project Structure](#project-structure)

---

### Problem Statement

The Indian YouTube market is one of the largest and most competitive in the world. For content creators, marketers, and brands, the ability to identify a potential viral hit early is a massive strategic advantage. However, simply forecasting view counts is not enough. Different videos exhibit vastly different growth patterns; a news clip might explode in popularity and fade quickly, while a comedy sketch builds momentum slowly over a week.

The objective of this project is to move beyond simple forecasting and develop a system that provides **strategic intelligence**. The goal is to:
1.  **Classify the *type* of virality** a video will have (its "archetype") based on its initial 24-hour performance.
2.  **Generate a tailored forecast** specific to that predicted growth pattern.

This allows for more effective resource allocation—for instance, deciding whether to maximize ad spend on a short-lived "explosive hit" or to continue promoting a "steady climb" video with long-term potential.

---

### Key Features & Solution

To address the problem, a multi-stage machine learning pipeline was developed:

*   **Unsupervised Archetype Discovery:** Used **K-Means Clustering with a Dynamic Time Warping (DTW)** metric on `tslearn` to analyze the *shape* of view growth over time, automatically discovering four distinct virality archetypes from the data.
*   **Context-Aware Feature Engineering:** Enriched the dataset with India-specific features, including language detection on titles and keyword analysis for high-impact topics like "Bollywood," "Cricket," and "News."
*   **Archetype Classification:** Trained a powerful **XGBoost Classifier** to predict a video's future archetype using only its first-day performance metrics and engineered features.
*   **Specialized Deep Learning Forecasts:** Deployed separate, expert **LSTM (Long Short-Term Memory)** neural networks for each major archetype. This tailored approach provides a more accurate forecast than a single generic model.
*   **Interactive Deployment:** The entire pipeline is wrapped in a user-friendly web application using **Streamlit**, allowing users to input a video's stats and receive an instant, two-part prediction in real-time.

---

### Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006400?style=for-the-badge&logo=xgboost&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![tslearn](https://img.shields.io/badge/tslearn-DB6F87?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)


---

### Project Pipeline

The project follows a systematic end-to-end machine learning workflow:

1.  **Data Ingestion & Cleaning:** Loaded the `INvideos.csv` dataset, handled encoding issues, parsed date-time columns, and cleaned invalid `video_id` entries.
2.  **Exploratory Data Analysis (EDA):** Mapped `category_id` to readable names using the associated JSON file and analyzed the distribution of top categories and channels.
3.  **Time-Series Transformation:** Converted the flat dataset into a collection of normalized view-count time series for each unique video.
4.  **Unsupervised Clustering (Archetype Discovery):** Applied K-Means clustering with a Dynamic Time Warping (DTW) metric to group videos into four distinct, meaningful virality archetypes.
5.  **Feature Engineering:** Created a rich feature set for the classifier, including initial performance stats, video metadata, and context-aware flags (e.g., `has_bollywood_keyword`, `is_from_top_channel`).
6.  **Supervised Classification:** Trained an XGBoost classifier to predict the virality archetype based on the engineered first-day features.
7.  **Deep Learning Forecasting:** Trained separate, specialized LSTM models for each major archetype to forecast the 7-day view trajectory.
8.  **Deployment:** Integrated the entire pipeline into an interactive Streamlit web application where models are loaded and used for live inference.

---


### Key Results & Insights

#### 1. Discovery of Indian YouTube Archetypes
The clustering algorithm successfully identified four primary patterns of virality, providing a new lens through which to analyze content strategy.

![Virality Archetypes Plot](https://github.com/Sourabh1710/Youtube-Virality-Prediction-Engine/blob/main/charts/virality_archetypes_plot.png)

#### 2. Diagnostic Power of the Classifier
The XGBoost classifier achieved an initial accuracy of **43%**, significantly outperforming a random 25% baseline. More importantly, its confusion matrix serves as a powerful diagnostic tool, revealing:
*   A strong performance in identifying the most common archetype, "The Standard Takeoff."
*   A clear path for future improvement by addressing class imbalance, particularly for the rare "Niche Anomaly" archetype.

![Confusion Matrix](https://github.com/Sourabh1710/Youtube-Virality-Prediction-Engine/blob/main/charts/Confusion_Matrix.png)

This analytical approach—diagnosing model weaknesses to inform the next iteration—is a core tenet of practical data science.

---

### How to Run This Project

Follow these steps to set up and run the project on your local machine.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
    cd YOUR_REPO_NAME
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate it
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    All required libraries are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch the Streamlit App:**
    The application script is located in the `script` directory.
    ```bash
    streamlit run script/app.py
    ```
    The application will open in your web browser.

---

### Project Structure
```
.
├── charts/
│   ├── app_screenshot.png
│   ├── confusion_matrix.png
│   └── virality_archetypes_plot.png
│
├── data/
│   ├── INvideos.csv
│   └── IN_category_id.json
│
├── models/
│   ├── lstm_model_cluster_0.h5
│   ├── scaler_cluster_0.pkl
│   ├── ... (other models & scalers) ...
│   └── xgb_classifier.joblib
│
├── notebook/
│   └── YouTube_Virality_Analysis.ipynb
│
├── script/
│   └── app.py
│
├── README.md
└── requirements.txt
```
---

### 👤 Author
**Sourabh Sonker**                                                                                                                 
**Aspiring Data Scientist**

