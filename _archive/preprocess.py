from data_utils import preprocess_brats21_dataset, update_metadata_with_patient_info
import os
import sys

if __name__ == '__main__':
    INPUT_DIR = './BraTS21'
    OUTPUT_DIR = './BraTS21_preprocessed'
    TARGET_SIZE = (224, 224)
    NUM_WORKERS = 8
    
    if len(sys.argv) > 1 and sys.argv[1] == '--update-metadata':
        print(f"Updating metadata: {OUTPUT_DIR}")
        update_metadata_with_patient_info(OUTPUT_DIR)
        sys.exit(0)
    
    print("="*60)
    print("BraTS21 Preprocessing")
    print("="*60)
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}, Workers: {NUM_WORKERS}")
    print("="*60)
    
    processed, failed = preprocess_brats21_dataset(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        target_size=TARGET_SIZE,
        max_patients=None,
        num_workers=NUM_WORKERS,
        skip_existing=True
    )
    
    print(f"\nDone. Processed: {processed}, Failed: {failed}")

