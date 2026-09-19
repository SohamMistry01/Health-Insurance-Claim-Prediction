<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.44-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Gemini_AI-2.0_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

# 🏥 Health Insurance Claim Prediction System

> An end-to-end Machine Learning web application that predicts annual health insurance premium costs based on individual health profiles — powered by a **Bayesian-optimized Random Forest** model and deployed via **Streamlit**.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack & Libraries](#-tech-stack--libraries)
- [Project Architecture](#-project-architecture)
- [Dataset](#-dataset)
- [Model Pipeline](#-model-pipeline)
- [Application Pages](#-application-pages)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [Project Structure](#-project-structure)
- [Future Scope](#-future-scope)
- [Contributing](#-contributing)

---

## 🔍 Overview

The **Health Insurance Claim Prediction System** is a machine learning-based web application designed to estimate the annual medical premium cost for individuals. Users input their health parameters — age, height, weight, medical history, and more — and the system returns a personalized premium estimate across multiple coverage tiers (₹5 Lakhs to ₹1 Crore).

The project follows a complete ML lifecycle:

```
Data Collection → EDA → Feature Engineering → Model Training → Hyperparameter Tuning → Deployment
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **ML-Powered Predictions** | Random Forest Regressor trained on 986 records with 11 health features |
| 🔬 **Bayesian Hyperparameter Tuning** | Optuna-based optimization across 50 trials for best model performance |
| 📊 **Interactive Visualizations** | 8+ Plotly charts — heatmaps, scatter plots, histograms, KDE, violin plots |
| 💬 **AI Health Assistant** | Integrated Gemini 2.0 Flash chatbot for real-time medical Q&A |
| 📐 **BMI Calculator** | Auto-computed BMI with WHO-standard classification table |
| 💰 **Multi-Tier Coverage** | Premium estimation across 5 insurance cover levels (₹5L–₹1Cr) |
| 🎨 **Custom UI** | Styled with custom background imagery and responsive layout |
| 🔎 **Data Explorer** | Browse the raw dataset, summary statistics, and correlation matrix |
| 📝 **Insight Buttons** | One-click statistical insights for every visualization |
| ☁️ **Codespaces Ready** | Pre-configured devcontainer for instant cloud development |

---

## 🛠 Tech Stack & Libraries

### Core Framework
| Library | Version | Purpose |
|---|---|---|
| **Streamlit** | 1.44.1 | Web application framework & interactive UI |
| **Python** | 3.11 | Programming language |

### Machine Learning & Data Science
| Library | Version | Purpose |
|---|---|---|
| **scikit-learn** | 1.3.0 | Random Forest, Gradient Boosting, XGBoost model training, train-test split, evaluation metrics |
| **XGBoost** | — | Gradient boosted decision tree regressor (evaluated during model selection) |
| **Optuna** | — | Bayesian hyperparameter optimization with 50-trial studies |
| **pandas** | 1.5.3 | Data loading, manipulation, and feature engineering |
| **NumPy** | 1.26.4 | Numerical computations and array operations |
| **joblib** | 1.4.2 | Model serialization — saving/loading the `.pkl` model file |

### Visualization
| Library | Version | Purpose |
|---|---|---|
| **Plotly** | 5.14.1 | Interactive charts — heatmaps, histograms, scatter plots, violin plots |
| **Kaleido** | 0.2.1 | Static image export engine for Plotly figures |
| **Matplotlib** | — | Exploratory plots in the Jupyter notebook |
| **Seaborn** | — | Statistical visualizations during EDA |

### AI & Utilities
| Library | Version | Purpose |
|---|---|---|
| **google-generativeai** | Latest | Gemini 2.0 Flash integration for the AI health assistant chatbot |
| **python-dotenv** | Latest | Secure environment variable management for API keys |
| **base64** (stdlib) | — | Background image encoding for custom Streamlit styling |

---

## 🏗 Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                   │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Home    │  │ Predictions  │  │  Visualizations  │  │
│  │  (About,  │  │  (User Input │  │ (8+ Interactive  │  │
│  │  Dataset, │  │   → Model    │  │  Plotly Charts)  │  │
│  │ AI Chat)  │  │   → Result)  │  │                  │  │
│  └───────────┘  └──────┬───────┘  └──────────────────┘  │
│                        │                                │
│  ┌─────────────────────┴────────────────────────────-┐  │
│  │              Model Overview Page                  │  │
│  │     (Embedded Jupyter Notebook as HTML)           │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────┐
│                    ML BACKEND                           │
│  ┌──────────────┐  ┌────────────┐  ┌────────────────┐   │
│  │  Random      │  │  Bayesian  │  │  Feature       │   │
│  │  Forest      │  │  Tuning    │  │  Engineering   │   │
│  │  Regressor   │  │  (Optuna)  │  │  (BMI)         │   │
│  │  (.pkl)      │  │  50 Trials │  │                │   │
│  └──────────────┘  └────────────┘  └────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │         Medicalpremium.csv (986 × 11)            │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────┐
│                   EXTERNAL SERVICES                     │
│         Google Gemini 2.0 Flash API (AI Chat)           │
└─────────────────────────────────────────────────────────┘
```

---

## 📂 Dataset

**File:** `Medicalpremium.csv` — 986 records × 11 features

### Feature Dictionary

| # | Feature | Type | Description | Range |
|---|---|---|---|---|
| 1 | `Age` | Continuous | Age of the policyholder | 18 – 66 |
| 2 | `Diabetes` | Binary (0/1) | Whether the person has diabetes | 0, 1 |
| 3 | `BloodPressureProblems` | Binary (0/1) | Whether the person has BP issues | 0, 1 |
| 4 | `AnyTransplants` | Binary (0/1) | History of organ transplants | 0, 1 |
| 5 | `AnyChronicDiseases` | Binary (0/1) | Presence of chronic conditions | 0, 1 |
| 6 | `Height` | Continuous | Height in cm | 145 – 188 |
| 7 | `Weight` | Continuous | Weight in kg | 51 – 132 |
| 8 | `KnownAllergies` | Binary (0/1) | Whether the person has known allergies | 0, 1 |
| 9 | `HistoryOfCancerInFamily` | Binary (0/1) | Family cancer history | 0, 1 |
| 10 | `NumberOfMajorSurgeries` | Discrete | Count of major surgeries undergone | 0 – 3 |
| 11 | `PremiumPrice` | Continuous | **Target variable** — Annual premium (₹) | ₹15,000 – ₹40,000 |

### Engineered Feature

| Feature | Formula | Purpose |
|---|---|---|
| `BMI` | `Weight / (Height/100)²` | Body Mass Index — captures weight-height relationship in a single metric |

### Key Dataset Statistics

- **Mean Premium Price:** ₹24,337
- **Mean Age:** ~42 years
- **Transplant Rate:** ~5.6% of policyholders
- **Chronic Disease Rate:** ~18% of policyholders

---

## 🧠 Model Pipeline

### 1. Data Preprocessing
- Loaded the CSV dataset using `pandas`
- Engineered **BMI** feature from Height and Weight
- No missing values — the dataset is clean

### 2. Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- **80-20 split** with a fixed random seed for reproducibility

### 3. Model Evaluation — Ensemble Comparison

Three ensemble regressors were evaluated head-to-head:

| Model | Metrics Evaluated |
|---|---|
| 🌲 **Random Forest Regressor** | MAE, RMSE, R² |
| 🚀 **XGBoost Regressor** | MAE, RMSE, R² |
| 📈 **Gradient Boosting Regressor** | MAE, RMSE, R² |

### 4. Hyperparameter Tuning — Bayesian Optimization (Optuna)

Both **Random Forest** and **Gradient Boosting** underwent Optuna-based Bayesian optimization with **50 trials** and **5-fold cross-validation**.

#### Random Forest Search Space
| Hyperparameter | Search Range |
|---|---|
| `n_estimators` | 100 – 500 |
| `criterion` | `squared_error`, `absolute_error`, `friedman_mse` |
| `max_depth` | 5 – 20 |
| `min_samples_split` | 2 – 10 |
| `min_samples_leaf` | 1 – 5 |
| `max_features` | `sqrt`, `log2`, `None` |

#### Gradient Boosting Search Space
| Hyperparameter | Search Range |
|---|---|
| `n_estimators` | 100 – 500 |
| `learning_rate` | 0.01 – 0.1 (log scale) |
| `max_depth` | 3 – 7 |
| `min_samples_split` | 2 – 10 |
| `min_samples_leaf` | 1 – 5 |
| `subsample` | 0.7 – 1.0 |
| `max_features` | `sqrt`, `log2` |

### 5. Final Model Selection

The **Bayesian-optimized Random Forest Regressor** was selected as the production model and serialized using `joblib`:

```python
joblib.dump(rf_best_model, 'new_rf_model.pkl')
```

### 6. Feature Importance Analysis

Post-training feature importance extraction identifies which health factors most significantly influence premium pricing — with **Age** being the strongest predictor (correlation = 0.70 with PremiumPrice).

---

## 📱 Application Pages

### 🏠 Home Page (`Home.py`)
The landing page with three sidebar-navigated sections:

- **About** — Project overview and quick-links to other pages
- **Dataset Info** — Interactive dataset explorer with adjustable row count, summary statistics, and correlation matrix
- **Chat with Gemini** — AI-powered medical assistant using Google's Gemini 2.0 Flash model with context-aware health guidance

### 💰 Predict Page (`pages/Predict.py`)
The core prediction interface:

- **Input Form** — Age, diabetes, blood pressure, transplants, chronic diseases, height, weight, allergies, cancer history, and major surgeries
- **BMI Auto-Calculation** — Instantly computed and displayed
- **Coverage Selector** — Choose from 5 insurance tiers:
  | Tier | Coverage | Premium Multiplier |
  |---|---|---|
  | ₹5 Lakhs | Basic | 0.40× |
  | ₹10 Lakhs | Standard | 0.50× |
  | ₹20 Lakhs | Enhanced | 0.60× |
  | ₹50 Lakhs | Premium | 0.75× |
  | ₹1 Crore | Supreme | 1.00× |
- **BMI Status Checker** — WHO-standard BMI classification (8 categories from *Severely Underweight* to *Obesity Class III*)
- **Model Prediction** — One-click premium estimation using the serialized Random Forest model

### 📊 Visualizations Page (`pages/Visualizations.py`)
Eight interactive Plotly visualizations with one-click insight buttons:

1. **Correlation Heatmap** — Full feature correlation matrix (Cividis colorscale)
2. **Age Distribution** — Histogram + KDE overlay
3. **Premium Price Distribution** — Histogram + Violin plot (probability density)
4. **Age vs. Premium Scatter Plot** — Positive trend visualization
5. **Premium Category × Transplants** — Grouped bar chart
6. **Premium Category × Surgeries** — Grouped bar chart
7. **Premium Category × Age Group** — Grouped bar chart
8. **Weight Category × Age Group** — Grouped bar chart

### 🔍 Model Overview Page (`pages/Model_Overview.py`)
Renders the complete Jupyter notebook (`model2.ipynb`) as an embedded HTML document — allowing users to review the entire model development process, from data loading through hyperparameter tuning to final model evaluation.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- A Google Gemini API key (for the AI chatbot feature)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/SohamMistry01/Health-Insurance-Claim-Prediction.git
cd Health-Insurance-Claim-Prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
# Create a .env file in the root directory:
echo 'API_KEY = "your-gemini-api-key-here"' > .env

# 5. Launch the application
streamlit run Home.py
```

The app will be available at **http://localhost:8501**

### Using GitHub Codespaces

This project includes a `.devcontainer` configuration for **instant cloud setup**:

1. Click the **"Code"** button on GitHub → **"Codespaces"** → **"Create codespace"**
2. The environment auto-installs dependencies and launches Streamlit on port `8501`
3. A preview window opens automatically

---

## 🔐 Environment Variables

| Variable | Description | Required |
|---|---|---|
| `API_KEY` | Google Gemini API key for the AI chatbot | Yes (for Chat feature) |

Create a `.env` file in the project root:
```env
API_KEY = "your-google-gemini-api-key"
```

> ⚠️ **Note:** The `.env` file is listed in `.gitignore` to prevent accidental key exposure.

---

## 📁 Project Structure

```
Health-Insurance-Claim-Prediction/
│
├── 📄 Home.py                        # Main entry point — About, Dataset, AI Chat
├── 📄 requirements.txt               # Python dependencies
├── 📄 Medicalpremium.csv             # Training dataset (986 × 11)
├── 📄 model2.ipynb                   # Jupyter notebook — full ML pipeline
├── 🖼️ bg_2.jpg                       # Custom background image
├── 📄 .env                           # API keys (gitignored)
├── 📄 .gitignore                     # Git exclusions
│
├── 📂 pages/                         # Streamlit multi-page app
│   ├── 📄 Predict.py                 # Premium prediction interface
│   ├── 📄 Visualizations.py          # 8+ interactive Plotly charts
│   ├── 📄 Model_Overview.py          # Embedded notebook viewer
│   ├── 📄 model2.html                # Pre-rendered notebook HTML
│   ├── 📄 new_rf_model.pkl           # Serialized Random Forest model
│   └── 📄 Medicalpremium.csv         # Dataset copy (used by pages)
│
├── 📂 .devcontainer/                 # GitHub Codespaces config
│   └── 📄 devcontainer.json          # Python 3.11, auto-install, port 8501
│
├── 🖼️ heatmap.png                    # Exported correlation heatmap
├── 🖼️ scatterplot.png                # Exported scatter plot
├── 🖼️ countplot1.png                 # Exported count plot (transplants)
├── 🖼️ countplot2.png                 # Exported count plot (surgeries)
├── 🖼️ countplot3.png                 # Exported count plot (age groups)
└── 🖼️ countplot4.png                 # Exported count plot (weight groups)
```

---

## 🔮 Future Scope

| Area | Enhancement |
|---|---|
| 🧪 **Model Improvement** | Incorporate neural networks (ANN/DNN) and stacking ensembles for higher accuracy |
| 📈 **Larger Dataset** | Scale to 10k+ records with demographic diversity for better generalization |
| 🌍 **Region-Specific Pricing** | Factor in geographic location, lifestyle, and occupation-based risk profiles |
| 🔄 **Real-Time Retraining** | Implement MLOps pipeline (MLflow / DVC) for continuous model updates |
| 📱 **Mobile App** | Build a Flutter/React Native companion app for on-the-go predictions |
| 🔒 **Authentication** | Add user login, prediction history, and personalized dashboards |
| 📊 **SHAP Explainability** | Integrate SHAP values for transparent, per-prediction feature explanations |
| 🏥 **Claim Probability** | Extend from premium prediction to claim likelihood estimation |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Commit** your changes: `git commit -m "Add your feature"`
4. **Push** to the branch: `git push origin feature/your-feature`
5. **Open** a Pull Request

---

<p align="center">
  <sub>Built with ❤️ using Python, Streamlit, and Scikit-Learn</sub>
</p>
