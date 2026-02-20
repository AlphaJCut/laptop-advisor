# 💻 Smart Laptop Advisor

An AI-powered laptop price prediction and recommendation system built with Python, Machine Learning, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)

## 🎯 Project Overview

Smart Laptop Advisor is an end-to-end machine learning project that helps users:
1. **Predict laptop prices** based on specifications
2. **Get personalized recommendations** based on use case and budget
3. **Analyze market trends** with interactive visualizations
4. **Find best deals** with value analysis

## 🚀 Features

### 1. Price Prediction
- 3 ML models compared (Linear Regression, Random Forest, Gradient Boosting)
- Best model: **Gradient Boosting** with R² = 0.92
- Feature engineering with 21 features
- Real-time predictions via web interface

### 2. Recommendation System
- Content-based filtering
- 6 use case profiles: Gaming, Office, Creative, Student, Ultraportable, All-Rounder
- Budget filtering
- Match scoring algorithm

### 3. Market Analysis
- Price distribution by brand/type
- Feature correlation analysis
- Interactive Plotly visualizations


## 📁 Project Structure

```
smart-laptop-advisor/
├── data/
│   ├── download_data.py       # Script tải dataset từ Kaggle
│   ├── processed/             # Data sau khi xử lý (auto-generated)
│   └── .gitkeep
├── models/
│   ├── price_model.pkl        # Trained model (auto-generated)
│   ├── recommender.pkl        # Recommendation system (auto-generated)
│   └── .gitkeep
├── src/
│   ├── data_preprocessing.py  # Data cleaning & feature engineering
│   ├── price_model.py         # Price prediction models
│   ├── recommender.py         # Recommendation system
│   └── utils.py               # Utility functions
├── notebooks/
│   └── laptop_EDA.ipynb          # Exploratory Data Analysis
├── app.py                     # Streamlit web application
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/smart-laptop-advisor.git
cd smart-laptop-advisor
```

### 2. Create virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Kaggle API & Download Dataset
```bash
# Bước 1: Tạo Kaggle API Key
# - Đăng nhập https://www.kaggle.com
# - Vào Settings → API → Click "Create Legacy API Key"
# - File kaggle.json sẽ tự động tải về

# Bước 2: Di chuyển kaggle.json vào đúng vị trí
# Linux/Mac:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows:
# Copy kaggle.json vào C:\Users\<YourUsername>\.kaggle\

# Bước 3: Tải dataset
cd data
python download_data.py
cd ..
```

### 5. Run the preprocessing and training
```bash
# Preprocess data
python src/data_preprocessing.py

# Train models
python src/price_model.py

# Build recommender
python src/recommender.py
```

### 6. Run the Streamlit app
```bash
streamlit run app.py
```

## 📊 Model Performance

| Model | Test RMSE | Test R² | CV RMSE |
|-------|-----------|---------|---------|
| Linear Regression | $2,146.30 | 0.70 | $2,778.66 |
| Random Forest | $1,254.22 | 0.90 | $1,747.42 |
| **Gradient Boosting** | **$1,088.20** | **0.92** | **$1,453.60** |

## 🔑 Key Features Used

Top 10 most important features for price prediction:
1. Performance Score
2. RAM (GB)
3. Storage (GB)
4. Brand
5. Weight (kg)
6. PPI (Pixels Per Inch)
7. GPU
8. Processor
9. Laptop Type
10. Operating System

## 🎨 Tech Stack

- **Python 3.8+**
- **Machine Learning**: Scikit-learn (Linear Regression, Random Forest, Gradient Boosting)
- **Data Processing**: Pandas, NumPy
- **Web Application**: Streamlit
- **Visualization**: Plotly, Matplotlib, Seaborn

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Recommended - Free)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

### Option 2: Hugging Face Spaces (Free)
1. Create a new Space on Hugging Face
2. Select Streamlit as SDK
3. Upload your files
4. Done!

### Option 3: Render/Railway (Free tier available)
1. Connect your GitHub repo
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `streamlit run app.py`

## 📈 Future Improvements

- [ ] Add more ML models (Neural Networks, CatBoost)
- [ ] Implement collaborative filtering
- [ ] Add real-time price scraping
- [ ] Build REST API with FastAPI
- [ ] Add user authentication
- [ ] Deploy with Docker

## 👨‍💻 Skills Demonstrated

This project demonstrates proficiency in:
- ✅ **Data Preprocessing** - Cleaning, feature engineering
- ✅ **Machine Learning** - Regression, model comparison, hyperparameter tuning
- ✅ **Recommendation Systems** - Content-based filtering, clustering
- ✅ **Web Development** - Streamlit interactive apps
- ✅ **Data Visualization** - Plotly charts, EDA
- ✅ **Software Engineering** - Modular code, OOP, documentation

## 📄 License

MIT License - feel free to use this project for your portfolio!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built with ❤️ for AI/ML Portfolio**
