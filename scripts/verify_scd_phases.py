import os
import glob
import nibabel as nib
import numpy as np
from tqdm import tqdm
import sys

# Define root data directory
DATA_ROOT = "data/SCD"

def get_lv_volume_and_labels(mask_path):
    """Returns LV pixel count (Class 1) and set of unique labels."""
    try:
        img = nib.load(mask_path)
        data = img.get_fdata()
        # count class 1
        lv_vol = np.sum(data == 1)
        labels = np.unique(data).astype(int)
        return lv_vol, set(labels)
    except Exception as e:
        print(f"Error loading {mask_path}: {e}")
        return None, None

def verify_phases():
    print(f"Verifying SCD Phases in {DATA_ROOT}...")
    
    # Find all patients
    # Structure: data/SCD/training/SCD0000101/...
    subsets = ['training', 'validate', 'testing']
    
    patient_dirs = []
    for subset in subsets:
        path = os.path.join(DATA_ROOT, subset)
        if os.path.exists(path):
            p_dirs = [os.path.join(path, d) for d in os.listdir(path) if d.startswith("SCD")]
            patient_dirs.extend(p_dirs)
            
    print(f"Found {len(patient_dirs)} patient directories.")
    
    correct_vol_order = 0
    wrong_vol_order = 0
    total_pairs = 0
    
    label_consistency = {
        'ED_has_MYO': 0,
        'ED_no_MYO': 0,
        'ES_has_MYO': 0,
        'ES_no_MYO': 0
    }
    
    swapped_candidates = []

    for p_dir in tqdm(patient_dirs):
        p_id = os.path.basename(p_dir)
        
        # Find GT files
        ed_gt = glob.glob(os.path.join(p_dir, "*_ED_*_gt.nii.gz"))
        es_gt = glob.glob(os.path.join(p_dir, "*_ES_*_gt.nii.gz"))
        
        if len(ed_gt) == 1 and len(es_gt) == 1:
            ed_path = ed_gt[0]
            es_path = es_gt[0]
            
            ed_vol, ed_labels = get_lv_volume_and_labels(ed_path)
            es_vol, es_labels = get_lv_volume_and_labels(es_path)
            
            if ed_vol is not None and es_vol is not None:
                total_pairs += 1
                
                # Check 1: Physiological Rule (ED Volume > ES Volume)
                if ed_vol > es_vol:
                    correct_vol_order += 1
                else:
                    wrong_vol_order += 1
                    swapped_candidates.append({
                        'id': p_id,
                        'reason': f"Vol ED ({ed_vol}) < Vol ES ({es_vol})",
                        'path': p_dir
                    })
                    
                # Check 2: Label Consistency (Tracking MYO - Class 2)
                has_myo_ed = 2 in ed_labels
                has_myo_es = 2 in es_labels
                
                if has_myo_ed: label_consistency['ED_has_MYO'] += 1
                else: label_consistency['ED_no_MYO'] += 1
                
                if has_myo_es: label_consistency['ES_has_MYO'] += 1
                else: label_consistency['ES_no_MYO'] += 1

    print(f"\n{'='*60}")
    print("VERIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"Total Patient Pairs Checked: {total_pairs}")
    
    print(f"\n1. VOLUMETRIC CHECK (Physiological: ED Vol > ES Vol)")
    print(f"   PASS: {correct_vol_order} patients ({(correct_vol_order/total_pairs)*100:.1f}%)")
    print(f"   FAIL: {wrong_vol_order} patients ({(wrong_vol_order/total_pairs)*100:.1f}%)")
    
    if wrong_vol_order > 0:
        print(f"\n   [WARNING] Potential Swaps based on Volume:")
        for cand in swapped_candidates:
            print(f"    - {cand['id']}: {cand['reason']}")
            
    print(f"\n2. LABEL CONTENT CHECK (MYO = Class 2)")
    print(f"   ED Frames: {label_consistency['ED_has_MYO']} have MYO, {label_consistency['ED_no_MYO']} no MYO")
    print(f"   ES Frames: {label_consistency['ES_has_MYO']} have MYO, {label_consistency['ES_no_MYO']} no MYO")
    
    # Heuristic for previous issue
    # Previously: "ED has MYO" was a sign of being ES. 
    # Ideal state based on previous fix: ED should NOT have MYO, ES should have MYO (if labeled).
    
    warn_ed_myo = label_consistency['ED_has_MYO']
    warn_es_no_myo = label_consistency['ES_no_MYO']
    
    if warn_ed_myo > 0:
        print(f"\n   [WARNING] {warn_ed_myo} ED files contain MYO label (Possible ES?).")
        
    print(f"\n{'='*60}")

if __name__ == "__main__":
    verify_phases()
