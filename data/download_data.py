"""
Download Laptop Price Dataset from Kaggle
==========================================

Script này tự động tải dataset từ Kaggle sử dụng Kaggle API.

SETUP KAGGLE API (chỉ cần làm 1 lần):
-------------------------------------
1. Đăng nhập Kaggle: https://www.kaggle.com
2. Vào Settings → API → Click "Create Legacy API Key"
3. File kaggle.json sẽ tự động tải về
4. Di chuyển file vào đúng vị trí:

   Linux/Mac:
   ----------
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json

   Windows:
   --------
   Copy kaggle.json vào C:\\Users\\<YourUsername>\\.kaggle\\

5. Chạy script này: python download_data.py

Dataset: https://www.kaggle.com/datasets/alhamdulliah123/laptop-price-estimation-using-feature-scaling

Author: AI Fresher Project
"""

import os
import sys
import shutil


# ===================== CONFIGURATION =====================
DATASET_NAME = "alhamdulliah123/laptop-price-estimation-using-feature-scaling"
OUTPUT_FILENAME = "laptop_prices.csv"
# =========================================================


def check_kaggle_setup():
    """Kiểm tra Kaggle API đã được setup chưa."""
    
    # Kiểm tra kaggle.json
    home = os.path.expanduser("~")
    kaggle_json = os.path.join(home, ".kaggle", "kaggle.json")
    
    if not os.path.exists(kaggle_json):
        print("=" * 60)
        print("❌ CHƯA SETUP KAGGLE API!")
        print("=" * 60)
        print("\nVui lòng làm theo các bước sau:")
        print("\n📌 Bước 1: Tạo API Key")
        print("   - Đăng nhập https://www.kaggle.com")
        print("   - Click vào avatar → Settings")
        print("   - Kéo xuống phần API → Click 'Create Legacy API Key'")
        print("   - File kaggle.json sẽ tự động tải về")
        print("\n📌 Bước 2: Di chuyển file kaggle.json")
        
        if sys.platform == "win32":
            user_path = os.path.expanduser("~")
            print(f"   - Copy file vào: {user_path}\\.kaggle\\kaggle.json")
        else:
            print("   - Chạy lệnh sau:")
            print("     mkdir -p ~/.kaggle")
            print("     mv ~/Downloads/kaggle.json ~/.kaggle/")
            print("     chmod 600 ~/.kaggle/kaggle.json")
        
        print(f"\n📁 File cần có tại: {kaggle_json}")
        print("=" * 60)
        return False
    
    print("✅ Kaggle API đã được setup!")
    return True


def install_kaggle():
    """Cài đặt kaggle package nếu chưa có."""
    try:
        import kaggle
        print("✅ Kaggle package đã được cài đặt!")
        return True
    except ImportError:
        print("📦 Đang cài đặt kaggle package...")
        os.system(f"{sys.executable} -m pip install kaggle -q")
        print("✅ Đã cài đặt kaggle package!")
        return True


def download_dataset():
    """Tải dataset từ Kaggle."""
    
    # Thư mục lưu data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"\n📥 Đang tải dataset: {DATASET_NAME}")
    print(f"📁 Lưu vào: {current_dir}")
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Authenticate
        api = KaggleApi()
        api.authenticate()
        print("✅ Xác thực Kaggle thành công!")
        
        # Download dataset
        api.dataset_download_files(
            DATASET_NAME,
            path=current_dir,
            unzip=True
        )
        
        print("✅ Tải dataset thành công!")
        
        # Liệt kê files đã tải
        print("\n📄 Các file đã tải:")
        for f in os.listdir(current_dir):
            if f.endswith('.csv'):
                filepath = os.path.join(current_dir, f)
                size = os.path.getsize(filepath) / 1024  # KB
                print(f"   - {f} ({size:.1f} KB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi tải dataset: {str(e)}")
        print("\n💡 Gợi ý:")
        print("   - Kiểm tra kết nối internet")
        print("   - Kiểm tra file kaggle.json đúng vị trí")
        print("   - Thử tải thủ công từ Kaggle và đặt vào thư mục data/")
        return False


def rename_and_cleanup():
    """Đổi tên file CSV và dọn dẹp."""
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Tìm file CSV đã tải
    csv_files = [f for f in os.listdir(current_dir) 
                 if f.endswith('.csv') and f != OUTPUT_FILENAME]
    
    if not csv_files:
        # Kiểm tra xem đã có file đúng tên chưa
        if os.path.exists(os.path.join(current_dir, OUTPUT_FILENAME)):
            print(f"✅ File {OUTPUT_FILENAME} đã tồn tại!")
            return True
        print("❌ Không tìm thấy file CSV nào!")
        return False
    
    # Đổi tên file đầu tiên tìm được
    for csv_file in csv_files:
        old_path = os.path.join(current_dir, csv_file)
        new_path = os.path.join(current_dir, OUTPUT_FILENAME)
        
        # Backup nếu đã tồn tại
        if os.path.exists(new_path):
            os.remove(new_path)
        
        shutil.move(old_path, new_path)
        print(f"✅ Đã đổi tên: {csv_file} → {OUTPUT_FILENAME}")
        break
    
    # Xóa các file CSV khác (nếu có)
    for f in os.listdir(current_dir):
        if f.endswith('.csv') and f != OUTPUT_FILENAME:
            os.remove(os.path.join(current_dir, f))
            print(f"🗑️ Đã xóa file thừa: {f}")
    
    return True


def verify_dataset():
    """Kiểm tra dataset đã tải đúng chưa."""
    try:
        import pandas as pd
    except ImportError:
        print("📦 Đang cài đặt pandas...")
        os.system(f"{sys.executable} -m pip install pandas -q")
        import pandas as pd
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, OUTPUT_FILENAME)
    
    if not os.path.exists(filepath):
        print(f"❌ File {OUTPUT_FILENAME} không tồn tại!")
        return False
    
    try:
        df = pd.read_csv(filepath)
        
        print("\n" + "=" * 60)
        print("📊 THÔNG TIN DATASET")
        print("=" * 60)
        print(f"📁 File: {OUTPUT_FILENAME}")
        print(f"📏 Số dòng: {len(df):,}")
        print(f"📐 Số cột: {len(df.columns)}")
        print(f"\n📋 Các cột trong dataset:")
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            print(f"   {i:2}. {col:<25} ({dtype}) - {null_count} null")
        
        print(f"\n📈 Preview dữ liệu:")
        print(df.head(3).to_string())
        
        print("\n" + "=" * 60)
        print("✅ Dataset sẵn sàng sử dụng!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ Lỗi đọc dataset: {str(e)}")
        return False


def main():
    """Main function."""
    print("=" * 60)
    print("🚀 SMART LAPTOP ADVISOR - DOWNLOAD DATASET")
    print("=" * 60)
    print(f"📦 Dataset: {DATASET_NAME}")
    print("=" * 60)
    
    # Step 1: Check Kaggle setup
    if not check_kaggle_setup():
        sys.exit(1)
    
    # Step 2: Install kaggle package
    if not install_kaggle():
        sys.exit(1)
    
    # Step 3: Download dataset
    if not download_dataset():
        sys.exit(1)
    
    # Step 4: Rename and cleanup
    rename_and_cleanup()
    
    # Step 5: Verify
    if not verify_dataset():
        sys.exit(1)
    
    print("\n🎉 Hoàn tất! Tiếp theo chạy các lệnh sau:")
    print("-" * 40)
    print("cd ..")
    print("python src/data_preprocessing.py")
    print("python src/price_model.py")
    print("python src/recommender.py")
    print("streamlit run app.py")
    print("-" * 40)


if __name__ == "__main__":
    main()
