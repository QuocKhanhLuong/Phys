"""
Script để kiểm tra việc nạp dữ liệu từ database ACDC.
Chạy script này để đảm bảo đường dẫn và cấu trúc dữ liệu đúng.
"""

import os
from data_utils import load_acdc_volumes

# Đường dẫn đến database (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DATASET_ROOT = os.path.join(PROJECT_ROOT, 'database')
TRAIN_DATA_PATH = os.path.join(BASE_DATASET_ROOT, 'training')
TEST_DATA_PATH = os.path.join(BASE_DATASET_ROOT, 'testing')

def test_data_structure():
    """Kiểm tra cấu trúc thư mục database."""
    print("="*80)
    print("KIỂM TRA CẤU TRÚC DATABASE")
    print("="*80)
    
    # Kiểm tra thư mục gốc
    if os.path.exists(BASE_DATASET_ROOT):
        print(f"✓ Thư mục database tồn tại: {BASE_DATASET_ROOT}")
    else:
        print(f"✗ KHÔNG tìm thấy thư mục database: {BASE_DATASET_ROOT}")
        return False
    
    # Kiểm tra thư mục training
    if os.path.exists(TRAIN_DATA_PATH):
        num_train_patients = len([d for d in os.listdir(TRAIN_DATA_PATH) 
                                 if os.path.isdir(os.path.join(TRAIN_DATA_PATH, d))])
        print(f"✓ Thư mục training tồn tại: {TRAIN_DATA_PATH}")
        print(f"  Số lượng bệnh nhân training: {num_train_patients}")
    else:
        print(f"✗ KHÔNG tìm thấy thư mục training: {TRAIN_DATA_PATH}")
        return False
    
    # Kiểm tra thư mục testing
    if os.path.exists(TEST_DATA_PATH):
        num_test_patients = len([d for d in os.listdir(TEST_DATA_PATH) 
                                if os.path.isdir(os.path.join(TEST_DATA_PATH, d))])
        print(f"✓ Thư mục testing tồn tại: {TEST_DATA_PATH}")
        print(f"  Số lượng bệnh nhân testing: {num_test_patients}")
    else:
        print(f"✗ KHÔNG tìm thấy thư mục testing: {TEST_DATA_PATH}")
        return False
    
    return True


def test_data_loading():
    """Kiểm tra việc nạp dữ liệu từ một số bệnh nhân."""
    print("\n" + "="*80)
    print("KIỂM TRA NẠP DỮ LIỆU")
    print("="*80)
    
    # Nạp một số volume training để test
    print(f"\nĐang nạp 3 bệnh nhân đầu tiên từ tập training...")
    try:
        train_volumes, train_masks = load_acdc_volumes(
            directory=TRAIN_DATA_PATH,
            target_size=(224, 224),
            max_patients=3
        )
        
        if len(train_volumes) > 0:
            print(f"✓ Nạp thành công {len(train_volumes)} volumes")
            print(f"  Shape của volume đầu tiên: {train_volumes[0].shape}")
            print(f"  Shape của mask đầu tiên: {train_masks[0].shape}")
            
            # Kiểm tra các thống kê cơ bản
            import numpy as np
            print(f"\n  Thống kê volume đầu tiên:")
            print(f"    - Min: {np.min(train_volumes[0]):.4f}")
            print(f"    - Max: {np.max(train_volumes[0]):.4f}")
            print(f"    - Mean: {np.mean(train_volumes[0]):.4f}")
            
            print(f"\n  Thống kê mask đầu tiên:")
            print(f"    - Các class có trong mask: {np.unique(train_masks[0])}")
            
            return True
        else:
            print("✗ Không nạp được volume nào")
            return False
            
    except Exception as e:
        print(f"✗ Lỗi khi nạp dữ liệu: {e}")
        return False


def test_data_format():
    """Kiểm tra format của một file bệnh nhân mẫu."""
    print("\n" + "="*80)
    print("KIỂM TRA FORMAT FILE")
    print("="*80)
    
    # Lấy bệnh nhân đầu tiên trong training
    try:
        patient_folders = sorted([d for d in os.listdir(TRAIN_DATA_PATH) 
                                 if os.path.isdir(os.path.join(TRAIN_DATA_PATH, d))])
        
        if patient_folders:
            first_patient = patient_folders[0]
            patient_path = os.path.join(TRAIN_DATA_PATH, first_patient)
            
            print(f"\nKiểm tra bệnh nhân mẫu: {first_patient}")
            
            # Liệt kê các file
            files = os.listdir(patient_path)
            nii_files = [f for f in files if f.endswith('.nii') or f.endswith('.nii.gz')]
            cfg_files = [f for f in files if f.endswith('.cfg')]
            
            print(f"  - Số file .nii: {len(nii_files)}")
            print(f"  - Số file .cfg: {len(cfg_files)}")
            
            # Kiểm tra Info.cfg
            info_cfg_path = os.path.join(patient_path, 'Info.cfg')
            if os.path.exists(info_cfg_path):
                print(f"  ✓ File Info.cfg tồn tại")
                
                # Đọc nội dung
                import configparser
                parser = configparser.ConfigParser()
                with open(info_cfg_path, 'r') as f:
                    config_string = '[DEFAULT]\n' + f.read()
                parser.read_string(config_string)
                
                ed_frame = parser['DEFAULT'].get('ED', 'N/A')
                es_frame = parser['DEFAULT'].get('ES', 'N/A')
                
                print(f"    - ED frame: {ed_frame}")
                print(f"    - ES frame: {es_frame}")
            else:
                print(f"  ✗ Không tìm thấy file Info.cfg")
            
            return True
        else:
            print("✗ Không tìm thấy bệnh nhân nào")
            return False
            
    except Exception as e:
        print(f"✗ Lỗi: {e}")
        return False


def main():
    """Chạy tất cả các test."""
    print("\n" + "="*80)
    print("TEST DATA LOADING - ACDC DATASET")
    print("="*80)
    print(f"\nĐường dẫn database: {BASE_DATASET_ROOT}")
    
    # Chạy các test
    results = []
    
    results.append(("Cấu trúc database", test_data_structure()))
    
    if results[0][1]:  # Chỉ tiếp tục nếu cấu trúc đúng
        results.append(("Format file", test_data_format()))
        results.append(("Nạp dữ liệu", test_data_loading()))
    
    # Tổng kết
    print("\n" + "="*80)
    print("TỔNG KẾT")
    print("="*80)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ TẤT CẢ CÁC TEST ĐỀU THÀNH CÔNG!")
        print("Bạn có thể bắt đầu training bằng lệnh: python train.py")
    else:
        print("\n✗ MỘT SỐ TEST THẤT BẠI!")
        print("Vui lòng kiểm tra lại đường dẫn và cấu trúc database.")
    
    print("="*80)


if __name__ == "__main__":
    main()

