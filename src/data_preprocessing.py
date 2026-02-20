"""
Data Preprocessing Module for Smart Laptop Advisor

Module này tự động phát hiện và xử lý các cột trong dataset,
hỗ trợ nhiều format dataset laptop khác nhau từ Kaggle.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import os
import re


class DataPreprocessor:
    """Handle all data preprocessing tasks."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = None
        self.categorical_columns = []
        self.numerical_columns = []
        
        # Mapping các tên cột phổ biến (Kaggle laptop datasets)
        self.column_mappings = {
            # Price columns
            'price': ['Price', 'price', 'price_usd', 'price_euros', 'Price_euros', 'MRP'],
            # Brand columns  
            'brand': ['Company', 'Brand', 'brand', 'company', 'Manufacturer'],
            # Type columns
            'laptop_type': ['TypeName', 'Type', 'Category', 'type', 'laptop_type'],
            # Processor columns
            'processor': ['Cpu', 'CPU', 'cpu', 'Processor', 'processor'],
            # RAM columns (GB as integer)
            'ram_gb': ['Ram', 'RAM', 'ram', 'Memory', 'ram_gb'],
            # Storage columns
            'ssd': ['SSD', 'ssd'],
            'hdd': ['HDD', 'hdd'],
            # GPU columns
            'gpu': ['Gpu', 'GPU', 'gpu', 'Graphics', 'graphics'],
            # Screen columns
            'screen_size': ['Inches', 'inches', 'ScreenSize', 'Display_Size', 'screen_size'],
            # OS columns
            'os': ['OpSys', 'OS', 'os', 'Operating_System', 'opsys'],
            # Product name
            'product': ['Product', 'product', 'Model', 'model']
        }
        
    def auto_detect_columns(self, df: pd.DataFrame):
        """Tự động phát hiện và chuẩn hóa tên cột."""
        print("\n🔍 Auto-detecting columns...")
        
        column_map = {}
        
        for standard_name, possible_names in self.column_mappings.items():
            for col in df.columns:
                if col in possible_names or col.lower() in [p.lower() for p in possible_names]:
                    column_map[col] = standard_name
                    print(f"   {col} → {standard_name}")
                    break
        
        # Rename columns
        df_renamed = df.rename(columns=column_map)
        
        # Detect target column (price) - exclude LogPrice
        for col in df_renamed.columns:
            if col.lower() == 'price' or (col.lower().startswith('price') and 'log' not in col.lower()):
                self.target_column = col
                break
        
        if self.target_column is None:
            # Tìm cột số có giá trị lớn nhất (có thể là giá)
            numeric_cols = df_renamed.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                max_col = df_renamed[numeric_cols].max().idxmax()
                if df_renamed[max_col].mean() > 100:  # Likely a price column
                    self.target_column = max_col
                    print(f"   Detected target column: {max_col}")
        
        # Calculate total storage (SSD + HDD)
        if 'ssd' in df_renamed.columns or 'hdd' in df_renamed.columns:
            ssd_val = df_renamed['ssd'] if 'ssd' in df_renamed.columns else 0
            hdd_val = df_renamed['hdd'] if 'hdd' in df_renamed.columns else 0
            df_renamed['storage_gb'] = ssd_val + hdd_val
            # If storage is 0, default to 256
            df_renamed.loc[df_renamed['storage_gb'] == 0, 'storage_gb'] = 256
            print(f"   Created storage_gb = SSD + HDD")
        
        # Define columns that should always be numerical
        force_numerical = ['ram_gb', 'storage_gb', 'screen_size', 'weight_kg', 
                          'ssd', 'hdd', 'inches']
        
        # Define columns that should always be categorical
        force_categorical = ['brand', 'processor', 'gpu', 'os', 'laptop_type',
                            'product', 'cpu_brand', 'gpu_brand']
        
        # Columns to skip
        skip_columns = ['logprice', 'log_price']
        
        # Auto-detect categorical vs numerical
        for col in df_renamed.columns:
            col_lower = col.lower()
            
            # Skip target column and log columns
            if col == self.target_column or any(skip in col_lower for skip in skip_columns):
                continue
            
            # Check forced lists first
            if any(col_lower == f or col_lower.replace('_', '') == f.replace('_', '') for f in force_numerical):
                if col not in self.numerical_columns:
                    self.numerical_columns.append(col)
            elif any(col_lower == f or col_lower.replace('_', '') == f.replace('_', '') for f in force_categorical) or df_renamed[col].dtype == 'object':
                if col not in self.categorical_columns:
                    self.categorical_columns.append(col)
            elif df_renamed[col].nunique() < 15:
                if col not in self.categorical_columns:
                    self.categorical_columns.append(col)
            else:
                if col not in self.numerical_columns:
                    self.numerical_columns.append(col)
        
        print(f"\n📊 Detected columns:")
        print(f"   Target: {self.target_column}")
        print(f"   Categorical ({len(self.categorical_columns)}): {self.categorical_columns}")
        print(f"   Numerical ({len(self.numerical_columns)}): {self.numerical_columns}")
        
        return df_renamed
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load dataset from CSV file."""
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")
        print(f"   Columns: {df.columns.tolist()}")
        return df
    
    def explore_data(self, df: pd.DataFrame) -> dict:
        """Perform exploratory data analysis."""
        eda_results = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numerical_stats': df.describe().to_dict(),
            'categorical_stats': {}
        }
        
        for col in self.categorical_columns:
            if col in df.columns:
                eda_results['categorical_stats'][col] = {
                    'unique_values': df[col].nunique(),
                    'top_values': df[col].value_counts().head(5).to_dict()
                }
        
        return eda_results
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and handle missing values."""
        df_clean = df.copy()
        
        # Remove duplicates
        initial_len = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        print(f"Removed {initial_len - len(df_clean)} duplicates")
        
        # Handle missing values
        for col in self.numerical_columns:
            if col in df_clean.columns and df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        for col in self.categorical_columns:
            if col in df_clean.columns and df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        # Remove outliers in target (using IQR method)
        if self.target_column and self.target_column in df_clean.columns:
            Q1 = df_clean[self.target_column].quantile(0.25)
            Q3 = df_clean[self.target_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df_clean[self.target_column] < lower_bound) | 
                       (df_clean[self.target_column] > upper_bound)).sum()
            print(f"Found {outliers} price outliers (keeping them for now)")
        
        return df_clean
    
    def extract_numeric_value(self, value):
        """Trích xuất giá trị số từ string."""
        if pd.isna(value):
            return 0
        if isinstance(value, (int, float)):
            return float(value)
        # Tìm số trong string
        numbers = re.findall(r'[\d.]+', str(value))
        if numbers:
            return float(numbers[0])
        return 0
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones (flexible for different datasets)."""
        df_feat = df.copy()
        
        # ============ RAM ============
        # Check if ram_gb already exists (from column mapping)
        if 'ram_gb' in df_feat.columns:
            # Already numeric, just ensure it's float
            df_feat['ram_gb'] = pd.to_numeric(df_feat['ram_gb'], errors='coerce').fillna(8)
        else:
            # Try to find RAM column
            for col in df_feat.columns:
                if 'ram' in col.lower():
                    df_feat['ram_gb'] = pd.to_numeric(df_feat[col], errors='coerce').fillna(8)
                    break
            else:
                df_feat['ram_gb'] = 8  # Default
        
        # ============ STORAGE ============
        # Check if storage_gb already created in auto_detect
        if 'storage_gb' not in df_feat.columns:
            # Try SSD + HDD
            ssd_val = 0
            hdd_val = 0
            for col in df_feat.columns:
                if col.lower() == 'ssd':
                    ssd_val = pd.to_numeric(df_feat[col], errors='coerce').fillna(0)
                elif col.lower() == 'hdd':
                    hdd_val = pd.to_numeric(df_feat[col], errors='coerce').fillna(0)
            
            df_feat['storage_gb'] = ssd_val + hdd_val
            df_feat.loc[df_feat['storage_gb'] == 0, 'storage_gb'] = 256  # Default
        
        # Ensure storage_gb is numeric
        df_feat['storage_gb'] = pd.to_numeric(df_feat['storage_gb'], errors='coerce').fillna(256)
        
        # ============ SCREEN SIZE ============
        if 'screen_size' in df_feat.columns:
            df_feat['screen_size'] = pd.to_numeric(df_feat['screen_size'], errors='coerce').fillna(15.6)
        else:
            for col in df_feat.columns:
                if 'inch' in col.lower():
                    df_feat['screen_size'] = pd.to_numeric(df_feat[col], errors='coerce').fillna(15.6)
                    break
            else:
                df_feat['screen_size'] = 15.6
        
        # ============ WEIGHT ============
        if 'weight_kg' not in df_feat.columns:
            # Dataset này không có weight, dùng default
            df_feat['weight_kg'] = 2.0
        
        # ============ RESOLUTION / PIXELS ============
        # Dataset này không có resolution, estimate từ screen size
        # Assume Full HD for most laptops
        df_feat['total_pixels'] = 1920 * 1080
        df_feat['ppi'] = np.sqrt(df_feat['total_pixels']) / df_feat['screen_size'].replace(0, 15.6)
        
        # ============ PROCESSOR SCORE ============
        processor_col = None
        for col in df_feat.columns:
            if col.lower() in ['processor', 'cpu']:
                processor_col = col
                break
        
        def get_processor_score(cpu_str):
            """Calculate processor score from CPU string."""
            if pd.isna(cpu_str):
                return 5
            cpu = str(cpu_str).lower()
            
            # Check for core count patterns
            if '14 core' in cpu or '16 core' in cpu:
                return 9
            elif '12 core' in cpu or 'octa core' in cpu:
                return 8
            elif '10 core' in cpu:
                return 7
            elif 'hexa core' in cpu or '6 core' in cpu:
                return 6
            elif 'quad core' in cpu or '4 core' in cpu:
                return 5
            elif 'dual core' in cpu or '2 core' in cpu:
                return 3
            
            # Check for Intel/AMD naming
            if 'i9' in cpu or 'ryzen 9' in cpu:
                return 9
            elif 'i7' in cpu or 'ryzen 7' in cpu or 'm2' in cpu or 'm3' in cpu:
                return 7
            elif 'i5' in cpu or 'ryzen 5' in cpu or 'm1' in cpu:
                return 5
            elif 'i3' in cpu or 'ryzen 3' in cpu:
                return 4
            elif 'celeron' in cpu or 'pentium' in cpu:
                return 2
            
            return 5  # Default
        
        if processor_col:
            df_feat['processor_score'] = df_feat[processor_col].apply(get_processor_score)
        else:
            df_feat['processor_score'] = 5
        
        # ============ GPU SCORE ============
        gpu_col = None
        for col in df_feat.columns:
            if col.lower() == 'gpu':
                gpu_col = col
                break
        
        def get_gpu_score(gpu_str):
            """Calculate GPU score from GPU string."""
            if pd.isna(gpu_str):
                return 3
            gpu = str(gpu_str).lower()
            
            # NVIDIA RTX 40 series
            if any(x in gpu for x in ['4090', '4080']):
                return 10
            elif any(x in gpu for x in ['4070', '4060']):
                return 8
            elif '4050' in gpu:
                return 7
            
            # NVIDIA RTX 30 series
            elif any(x in gpu for x in ['3090', '3080']):
                return 9
            elif '3070' in gpu:
                return 8
            elif '3060' in gpu:
                return 7
            elif '3050' in gpu:
                return 6
            
            # NVIDIA RTX 20 series & GTX
            elif any(x in gpu for x in ['2080', '2070']):
                return 7
            elif any(x in gpu for x in ['2060', '2050', '1660']):
                return 5
            elif any(x in gpu for x in ['1650', '1060', 'mx']):
                return 4
            
            # AMD
            elif 'rx 6' in gpu or 'radeon rx' in gpu:
                return 5
            elif 'radeon' in gpu:
                return 3
            
            # Integrated
            elif any(x in gpu for x in ['intel', 'iris', 'uhd', 'integrated', 'hd graphics']):
                return 2
            elif 'apple' in gpu or 'm1' in gpu or 'm2' in gpu:
                return 5
            
            return 3  # Default
        
        if gpu_col:
            df_feat['gpu_score'] = df_feat[gpu_col].apply(get_gpu_score)
        else:
            df_feat['gpu_score'] = 3
        
        # ============ DERIVED SCORES ============
        # Performance score
        df_feat['performance_score'] = (
            df_feat['processor_score'] * 0.4 +
            df_feat['gpu_score'] * 0.3 +
            (df_feat['ram_gb'] / 64) * 10 * 0.2 +
            (df_feat['storage_gb'] / 2048) * 10 * 0.1
        )
        
        # Portability score
        df_feat['portability_score'] = 10 - (df_feat['weight_kg'] * 2)
        df_feat['portability_score'] = df_feat['portability_score'].clip(0, 10)
        
        # Value score (performance per price unit)
        price_col = self.target_column or 'price'
        if price_col in df_feat.columns:
            price_values = pd.to_numeric(df_feat[price_col], errors='coerce').fillna(1)
            # Normalize price (assume INR, divide by 1000)
            df_feat['value_score'] = df_feat['performance_score'] / (price_values / 10000).replace(0, 1)
        else:
            df_feat['value_score'] = df_feat['performance_score']
        
        # ============ BINARY FLAGS ============
        # Is gaming laptop
        df_feat['is_gaming'] = 0
        
        # Check product name for "gaming"
        for col in df_feat.columns:
            if col.lower() == 'product':
                df_feat['is_gaming'] = df_feat[col].str.lower().str.contains('gaming', na=False).astype(int)
                break
        
        # Also check GPU for gaming cards
        if gpu_col:
            gaming_gpu = df_feat[gpu_col].str.contains('RTX|GTX|RX 6', case=False, na=False)
            df_feat['is_gaming'] = (df_feat['is_gaming'] | gaming_gpu).astype(int)
        
        # Is ultraportable
        df_feat['is_ultraportable'] = (
            (df_feat['weight_kg'] < 1.5) & 
            (df_feat['screen_size'] <= 14)
        ).astype(int)
        
        print(f"\n✅ Feature engineering completed!")
        print(f"   New features: ram_gb, storage_gb, screen_size, weight_kg")
        print(f"   Scores: processor_score, gpu_score, performance_score, portability_score, value_score")
        print(f"   Flags: is_gaming, is_ultraportable")
        
        return df_feat
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        df_encoded = df.copy()
        
        for col in self.categorical_columns:
            if col in df_encoded.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    # Handle unseen labels
                    le = self.label_encoders[col]
                    df_encoded[f'{col}_encoded'] = df_encoded[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        
        return df_encoded
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for model training."""
        # Columns to use for modeling
        feature_cols = (
            [f'{col}_encoded' for col in self.categorical_columns if col in df.columns] +
            [col for col in self.numerical_columns if col in df.columns] +
            ['processor_score', 'gpu_score', 'performance_score', 
             'portability_score', 'total_pixels', 'ppi', 'is_gaming', 'is_ultraportable',
             'ram_gb', 'storage_gb', 'screen_size', 'weight_kg']
        )
        
        # Filter only existing columns and remove duplicates
        self.feature_columns = list(dict.fromkeys([col for col in feature_cols if col in df.columns]))
        
        # Remove target column from features if present
        if self.target_column in self.feature_columns:
            self.feature_columns.remove(self.target_column)
        
        X = df[self.feature_columns]
        
        # Get target variable
        if self.target_column and self.target_column in df.columns:
            y = df[self.target_column]
        else:
            # Try to find price column
            for col in df.columns:
                if 'price' in col.lower() and 'log' not in col.lower():
                    y = df[col]
                    break
            else:
                raise ValueError("Could not find target (price) column!")
        
        print(f"\n📊 Features: {len(self.feature_columns)} columns")
        print(f"🎯 Target: {self.target_column}")
        
        return X, y
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Scale numerical features."""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor state."""
        state = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'target_column': self.target_column
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.label_encoders = state['label_encoders']
        self.scaler = state['scaler']
        self.feature_columns = state['feature_columns']
        self.categorical_columns = state['categorical_columns']
        self.numerical_columns = state['numerical_columns']
        print(f"Preprocessor loaded from {filepath}")


def preprocess_pipeline(data_path: str, output_dir: str = None):
    """Run the complete preprocessing pipeline."""
    preprocessor = DataPreprocessor()
    
    # Load data
    df = preprocessor.load_data(data_path)
    
    # Auto-detect and standardize columns
    df = preprocessor.auto_detect_columns(df)
    
    # EDA
    eda = preprocessor.explore_data(df)
    print("\n=== Exploratory Data Analysis ===")
    print(f"Shape: {eda['shape']}")
    print(f"Missing values: {sum(eda['missing_values'].values())}")
    
    # Clean data
    df_clean = preprocessor.clean_data(df)
    
    # Feature engineering
    df_features = preprocessor.feature_engineering(df_clean)
    
    # Encode categorical features
    df_encoded = preprocessor.encode_features(df_features, fit=True)
    
    # Prepare features
    X, y = preprocessor.prepare_features(df_encoded)
    
    # Scale features
    X_scaled = preprocessor.scale_features(X, fit=True)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_scaled, y)
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Save preprocessor
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        preprocessor.save_preprocessor(os.path.join(output_dir, 'preprocessor.pkl'))
        
        # Save processed data
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        # Save full processed dataframe
        df_encoded.to_csv(os.path.join(output_dir, 'processed_data.csv'), index=False)
        print(f"\n✅ Processed data saved to {output_dir}")
    
    return preprocessor, df_encoded, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Run preprocessing pipeline
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    data_path = os.path.join(project_dir, "data", "laptop_prices.csv")
    output_dir = os.path.join(project_dir, "data", "processed")
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print("=" * 60)
        print("❌ Dataset not found!")
        print(f"   Expected: {data_path}")
        print("\n💡 Please run download_data.py first:")
        print("   cd data")
        print("   python download_data.py")
        print("=" * 60)
    else:
        preprocessor, df, X_train, X_test, y_train, y_test = preprocess_pipeline(
            data_path, output_dir
        )
        
        print("\n=== Feature Columns ===")
        print(preprocessor.feature_columns)
