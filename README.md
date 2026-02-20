# Smart Laptop Advisor

Dự đoán giá laptop và gợi ý laptop phù hợp bằng Machine Learning với Scikit-learn và Streamlit.

## Kết quả

- **Best Model:** Random Forest
- **Test R2 Score:** 0.83
- **Test RMSE:** ~8,500 INR
- **Features:** 21 đặc trưng
- **Dataset:** 893 laptops (Indian Market)

## Dataset

Project sử dụng dataset từ Kaggle:

| Dataset | Số bản ghi | Nguồn |
|---------|------------|-------|
| Laptop Price Estimation | 893 | [Kaggle](https://www.kaggle.com/datasets/alhamdulliah123/laptop-price-estimation-using-feature-scaling) |

**Các cột chính trong dataset:**
- Company, Product, Cpu, Ram, Gpu, OpSys, Inches, Price
- Cpu_brand, Gpu_brand, HDD, SSD

## Phân tích dữ liệu (EDA)

Chi tiết về notebook phân tích dữ liệu được mô tả trong file:
- [EDA_Documentation.docx](docs/EDA_Documentation.docx) - Tài liệu mô tả chi tiết các biểu đồ và kết quả phân tích

Notebook `notebooks/laptop_EDA.ipynb` chứa toàn bộ quy trình:
1. Import Libraries
2. Load Dataset  
3. Exploratory Data Analysis
4. Data Preprocessing
5. Feature Engineering
6. Model Training
7. Model Evaluation
8. Feature Importance
9. Conclusion

## Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/smart-laptop-advisor.git
cd smart-laptop-advisor
```

### 2. Tạo virtual environment (Python 3.8+)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 4. Setup Kaggle API và tải dataset

```bash
# Bước 1: Tạo Kaggle API Key
# - Đăng nhập https://www.kaggle.com
# - Vào Settings -> API -> Click "Create New API Token"
# - File kaggle.json sẽ tự động tải về

# Bước 2: Di chuyển kaggle.json
# Windows: Copy vào C:\Users\<Username>\.kaggle\
# Linux/Mac:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Bước 3: Tải dataset
cd data
python download_data.py
cd ..
```

### 5. Tiền xử lý dữ liệu và huấn luyện model

```bash
python src/data_preprocessing.py
python src/price_model.py
python src/recommender.py
```

### 6. Chạy ứng dụng web

```bash
streamlit run app.py
```

## Sử dụng

### Tab 1: Dự đoán giá laptop

Chọn cấu hình laptop (Brand, CPU, GPU, RAM, Storage, Screen Size) và nhấn "Predict Price".

### Tab 2: Gợi ý laptop

| Tùy chọn | Mô tả |
|----------|-------|
| Use Case | Chọn mục đích sử dụng (Gaming, Office, Creative, Student, Programming, Portable) |
| Maximum Budget | Ngân sách tối đa (VND hoặc INR) |
| Number of recommendations | Số lượng laptop muốn hiển thị (3-10). Ví dụ: chọn 5 sẽ hiển thị top 5 laptop phù hợp nhất |

Hệ thống sẽ tính điểm phù hợp (Match Score) cho từng laptop dựa trên:
- Yêu cầu tối thiểu của use case (RAM, GPU score, CPU score)
- Trọng số các đặc trưng phù hợp với mục đích sử dụng
- Lọc theo ngân sách

### Tab 3: Phân tích thị trường

Xem biểu đồ phân tích giá theo thương hiệu, RAM, hiệu năng và ma trận tương quan.

## Cấu trúc thư mục

```
smart-laptop-advisor/
├── data/
│   ├── download_data.py          # Script tải dataset từ Kaggle
│   ├── laptop_prices.csv         # Dataset gốc (sau khi tải)
│   └── processed/
│       ├── processed_data.csv    # Dữ liệu đã xử lý
│       ├── preprocessor.pkl      # Preprocessor đã train
│       ├── X_train.npy           # Training features
│       ├── X_test.npy            # Test features
│       ├── y_train.npy           # Training labels
│       └── y_test.npy            # Test labels
├── docs/
│   └── EDA_Documentation.docx    # Tài liệu mô tả EDA
├── models/
│   ├── price_model.pkl           # Model dự đoán giá
│   ├── recommender.pkl           # Hệ thống gợi ý
│   ├── feature_importance.csv    # Độ quan trọng đặc trưng
│   └── model_comparison.csv      # So sánh các model
├── notebooks/
│   └── laptop_EDA.ipynb          # Notebook phân tích dữ liệu
├── src/
│   ├── __init__.py               # Module init
│   ├── data_preprocessing.py     # Tiền xử lý dữ liệu
│   ├── price_model.py            # Huấn luyện model giá
│   ├── recommender.py            # Hệ thống gợi ý
│   └── utils.py                  # Hàm tiện ích
├── app.py                        # Ứng dụng Streamlit
├── requirements.txt              # Thư viện cần thiết
├── .gitignore                    # Git ignore
└── README.md                     # Tài liệu
```

## Công nghệ

- Python 3.8+
- Scikit-learn (Linear Regression, Random Forest, Gradient Boosting)
- Pandas, NumPy
- Streamlit
- Plotly, Matplotlib, Seaborn
- Kaggle API

## Model Architecture

```
3 Models được so sánh:

1. Linear Regression
   - Baseline model
   - R2 ~ 0.70

2. Random Forest (Best)
   - n_estimators: 100
   - max_depth: 15
   - R2 ~ 0.83

3. Gradient Boosting
   - n_estimators: 100
   - learning_rate: 0.1
   - R2 ~ 0.80
```

## Feature Engineering

```
Các đặc trưng được tạo:
- processor_score: Điểm CPU (1-10)
- gpu_score: Điểm GPU (1-10)
- performance_score: Điểm hiệu năng tổng hợp
- portability_score: Điểm di động
- value_score: Điểm giá trị
- is_gaming: Laptop gaming hay không
- is_ultraportable: Laptop siêu mỏng nhẹ
- storage_gb: Tổng dung lượng (SSD + HDD)
```

## Tính năng chính

| Tab | Chức năng |
|-----|-----------|
| Price Prediction | Dự đoán giá dựa trên cấu hình |
| Smart Recommendations | Gợi ý laptop theo nhu cầu và ngân sách |
| Market Analysis | Phân tích thị trường với biểu đồ |

## Hỗ trợ tiền tệ

- **VND:** Việt Nam Đồng (mặc định)
- **INR:** Indian Rupee
- Tỷ giá: 1 INR = 287 VND (có thể tùy chỉnh)

## License

MIT License