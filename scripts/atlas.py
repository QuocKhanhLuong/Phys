import os
import sys
import subprocess
import getpass
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    # Cố gắng import hàm từ file atlas.py (bạn cần đặt nó vào src)
    from src.data_utils.atlas_utils import bidsify_indi_atlas
    print("✓ Tải hàm 'bidsify_indi_atlas' thành công.")
except ImportError as e:
    print(f"CẢNH BÁO: Không thể import 'bidsify_indi_atlas': {e}")
    bidsify_indi_atlas = None
except Exception as e:
    print(f"LỖI không xác định khi import: {e}")
    bidsify_indi_atlas = None

# --- Bước 2: Định nghĩa các đường dẫn ---
# (Giả định bạn chạy script này từ thư mục gốc 'Phys')

# Đường dẫn đến file đã tải xuống bằng gdown
encrypted_file = project_root / "data" / "ATLAS" / "ATLAS_R2.0_encrypted.tar.gz"

# Nơi lưu file giải mã .tar.gz
decrypted_file = project_root / "data" / "ATLAS" / "ATLAS_R2.0.tar.gz"

# Thư mục để giải nén (nó sẽ tạo ra 'ATLAS_2' bên trong đây)
extract_dir = project_root / "data" / "ATLAS"

# Đường dẫn đến thư mục 'ATLAS_2' sau khi giải nén
bids_input_dir = extract_dir / "ATLAS_2"

# Nơi lưu trữ dataset theo chuẩn BIDS (để 'train.py' sử dụng)
bids_output_dir = project_root / "data" / "ATLAS_BIDS" 


def decrypt_data():
    """
    Sử dụng logic từ ISLES_Example.ipynb để giải mã.
    """
    print(f"Bắt đầu giải mã file: {encrypted_file}")
    if not encrypted_file.exists():
        print(f"LỖI: Không tìm thấy file đã mã hóa tại: {encrypted_file}")
        print("Bạn đã tải file về 'Phys/data/ATLAS/' chưa?")
        return False

    try:
        # Lấy mật khẩu từ environment variable hoặc người dùng
        password = os.environ.get('ATLAS_PASSWORD')
        if not password:
        password = getpass.getpass("Nhập mật khẩu giải mã ATLAS: ")
        else:
            print("Sử dụng mật khẩu từ ATLAS_PASSWORD environment variable.")
        
        # Xây dựng lệnh openssl
        decrypt_command = [
            'openssl', 'aes-256-cbc', '-md', 'sha256',
            '-d', '-a', '-in', str(encrypted_file),
            '-out', str(decrypted_file),
            '-pass', f'pass:{password}'
        ]
        
        # Gọi lệnh
        subprocess.run(decrypt_command, check=True)
        print(f"Giải mã thành công! Đã lưu tại: {decrypted_file}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"LỖI giải mã: {e}")
        print("Sai mật khẩu hoặc lỗi openssl. Thử lại.")
        if decrypted_file.exists():
            os.remove(decrypted_file) # Xóa file lỗi nếu có
        return False
    except Exception as e:
        print(f"Lỗi không xác định: {e}")
        return False

def extract_data():
    """
    Giải nén file .tar.gz
    """
    print(f"Bắt đầu giải nén: {decrypted_file}")
    if not decrypted_file.exists():
        print(f"LỖI: Không tìm thấy file đã giải mã: {decrypted_file}")
        return False
        
    try:
        # Lệnh tar: -xzf [file] -C [thư mục giải nén]
        # -C (chữ C viết hoa) chỉ định thư mục đích
        extract_command = [
            'tar', '-xzf', str(decrypted_file),
            '-C', str(extract_dir)
        ]
        
        subprocess.run(extract_command, check=True)
        print(f"Giải nén thành công vào: {extract_dir}")
        print(f"Thư mục '{bids_input_dir}' nên xuất hiện.")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"LỖI giải nén: {e}")
        return False

def main():
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(bids_output_dir, exist_ok=True)

    # Bước 1: Giải mã
    if not decrypted_file.exists():
        if not decrypt_data():
            sys.exit(1) # Dừng nếu giải mã lỗi
    else:
        print(f"Đã tìm thấy file giải mã: {decrypted_file}. Bỏ qua bước giải mã.")

    # Bước 2: Giải nén
    if not bids_input_dir.exists():
        if not extract_data():
            sys.exit(1) # Dừng nếu giải nén lỗi
    else:
        print(f"Đã tìm thấy thư mục: {bids_input_dir}. Bỏ qua bước giải nén.")

    # Bước 3: Chuyển đổi BIDS
    if bidsify_indi_atlas is not None:
        print(f"Bắt đầu chuyển đổi BIDS-ify...")
        print(f"  Nguồn: {bids_input_dir}")
        print(f"  Đích:  {bids_output_dir}")
        
        # Gọi hàm từ src/atlas_utils.py
        bidsify_indi_atlas(str(bids_input_dir), str(bids_output_dir))
        
        print("\nHoàn tất! Dữ liệu ATLAS đã sẵn sàng tại:")
        print(f"{bids_output_dir / 'train'}")
        print(f"{bids_output_dir / 'test'}")
    else:
        print("\nDừng lại. Không thể BIDS-ify vì không tìm thấy hàm 'bidsify_indi_atlas'.")
        print("Hãy đảm bảo bạn đã copy 'atlas.py' vào 'src/atlas_utils.py'.")

if __name__ == "__main__":
    main()