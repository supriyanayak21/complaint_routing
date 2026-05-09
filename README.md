# AI-Powered Complaint Auto-Routing System
## Project Overview
This project is an AI/ML-powered complaint management and auto-routing system that processes multilingual complaints submitted through text, audio, or video. The system automatically predicts complaint priority, estimates resolution time (ETA), assigns the best officer, and finds similar past complaints using machine learning and NLP techniques.

The system is built completely using local/offline models without using any external APIs.
## Tech Stack
## Frontend
HTML
CSS
Bootstrap
## Backend
Flask
Machine Learning
Scikit-learn
Sentence Transformers
Whisper
## Database
CSV Dataset (for prototype)

## Machine Learning Tasks
| Task                     | Technique Used      |
| ------------------------ | ------------------- |
| Priority Prediction      | Logistic Regression |
| Text Vectorization       | TF-IDF              |
| Similar Complaint Search | Sentence Embeddings |
| ETA Prediction           | Regression          |
| Speech-to-Text           | Whisper             |

complaint-routing/
│
├── app.py
├── train.py
├── dataset.csv
├── requirements.txt
│
├── models/
│   ├── priority_model.pkl
│   └── vectorizer.pkl
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│
└── uploads/

# Installation
## 1. Clone Repository
    git clone <your-github-repo-link>
    cd complaint-routing
## 2. Create Virtual Environment
    python -m venv venv
    venv\Scripts\activate
## 3. Install Dependencies
    pip install -r requirements.txt
## Train the ML Model
    python train.py
  This will:

train the ML model
generate TF-IDF vectors
save trained model files

## Run Flask Application
    python app.py
# Workflow
User Complaint
      ↓
Text/Audio Processing
      ↓
Feature Extraction
      ↓
Priority Prediction
      ↓
Officer Routing
      ↓
ETA Estimation
      ↓
Similar Complaint Search

## Evaluation Metrics
Classification Metrics
Accuracy
Precision
Recall
F1-Score
Regression Metrics
Mean Absolute Error (MAE)


## Future Enhancements
Real database integration
User authentication
Admin dashboard
Real-time notifications
Geo-location support
Mobile application
Deep learning-based NLP models


## Advantages
Reduces manual complaint handling
Faster complaint routing
Improves resolution efficiency
Supports multilingual complaints
Works completely offline


## License

This project is created for educational and assignment purposes.
