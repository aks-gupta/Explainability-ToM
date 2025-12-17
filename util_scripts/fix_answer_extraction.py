import json
import os
import re
import shutil
from pathlib import Path
import pickle as pkl

DATASET_FOLDER = "harmful-requests_llama3-2-90b-instruct_30_counterfactuals_TEMPLATE_BASED"

def extract_answer_from_text(text):
    if not text or not isinstance(text, str):
        return None
    
    text_lower = text.lower()
    
    patterns = [
        r'so\s+the\s+answer\s+is\s+(yes|no)',
        r'therefore[,\s]+the\s+answer\s+is\s+(yes|no)',
        r'thus[,\s]+the\s+answer\s+is\s+(yes|no)',
        r'hence[,\s]+the\s+answer\s+is\s+(yes|no)',
        r'the\s+answer\s+is\s+(yes|no)',
        r'answer:\s*(yes|no)',
        r'\*\*(yes|no)\*\*'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(1)
    
    return None

def fix_nested_dict(obj, path="", fixed_count=0, total_neither=0):
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            
            if key.endswith('_answer') and value == 'neither':
                expl_key = key.replace('_answer', '_explanation')
                if expl_key in obj and obj[expl_key]:
                    total_neither += 1
                    extracted = extract_answer_from_text(obj[expl_key])
                    if extracted:
                        print(f"  {new_path}: Fixing 'neither' -> '{extracted}'")
                        obj[key] = extracted
                        fixed_count += 1
            
            elif isinstance(value, (dict, list)):
                fixed_count, total_neither = fix_nested_dict(value, new_path, fixed_count, total_neither)
    
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            new_path = f"{path}[{i}]"
            if isinstance(item, (dict, list)):
                fixed_count, total_neither = fix_nested_dict(item, new_path, fixed_count, total_neither)
    
    return fixed_count, total_neither

def get_file_type(filename):
    if 'simulation_question_answers' in filename:
        return 'simqa'
    elif 'task_qa_simulation_questions' in filename:
        return 'taskqa_sim'
    elif 'task_qa_out' in filename:
        return 'taskqa'
    else:
        return 'other'

def fix_json_file(file_path):
    path_obj = Path(file_path)
    parent_dir = path_obj.parent.name
    filename = path_obj.name
    file_type = get_file_type(filename)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fixed_count, total_neither = fix_nested_dict(data)
    
    if total_neither > 0:
        print(f"\n  [{parent_dir}] {file_type}: {filename}")
        print(f"    Found {total_neither} 'neither' → Fixed {fixed_count} → Remaining {total_neither - fixed_count}")
    
    return data, fixed_count

def process_directory(base_dir):
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Directory not found: {base_dir}")
        return
    
    dataset_folders = [d for d in base_path.iterdir() if d.is_dir()]
    dataset_folders.sort()
    
    total_files_updated = 0
    total_fixes_made = 0
    dataset_summary = {}
    
    for dataset_folder in dataset_folders:
        dataset_name = dataset_folder.name
        print(f"\n{'='*80}")
        print(f"📁 DATASET: {dataset_name}")
        print(f"{'='*80}")
        
        version_folders = [v for v in dataset_folder.iterdir() if v.is_dir() and v.name.startswith('v')]
        version_folders.sort(key=lambda x: int(x.name[1:]) if x.name[1:].isdigit() else 999)
        
        dataset_updated = 0
        dataset_fixed = 0
        
        for version_folder in version_folders:
            version_name = version_folder.name
            print(f"\n  📂 Version: {version_name}")
            
            json_files = [f for f in version_folder.glob("*.json") if '_old' not in f.name]
            
            version_updated = 0
            version_fixed = 0
            
            for json_file in json_files:
                try:
                    fixed_data, fixed_count = fix_json_file(str(json_file))
                    
                    if fixed_count > 0:
                        backup_path = json_file.with_name(json_file.stem + '_old.json')
                        shutil.copy2(json_file, backup_path)
                        
                        with open(json_file, 'w', encoding='utf-8') as f:
                            json.dump(fixed_data, f, indent=4, ensure_ascii=False)
                        
                        print(f"    ✅ Updated and backed up")
                        version_updated += 1
                        version_fixed += fixed_count
                        
                except Exception as e:
                    print(f"    ❌ Error: {e}")
            
            if version_updated > 0:
                print(f"    📊 {version_name} Summary: {version_updated} file(s) updated, {version_fixed} answer(s) fixed")
            
            dataset_updated += version_updated
            dataset_fixed += version_fixed
        
        if dataset_updated > 0:
            dataset_summary[dataset_name] = {'files': dataset_updated, 'fixes': dataset_fixed}
            print(f"\n  ✨ {dataset_name} Total: {dataset_updated} file(s) updated, {dataset_fixed} answer(s) fixed")
        else:
            print(f"  ✓ No fixes needed in {dataset_name}")
        
        total_files_updated += dataset_updated
        total_fixes_made += dataset_fixed
    
    return total_files_updated, total_fixes_made, dataset_summary

def process_single_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset folder not found: {dataset_path}")
        return 0, 0
    
    dataset_name = dataset_path.name
    print(f"\n{'='*80}")
    print(f"📁 DATASET: {dataset_name}")
    print(f"{'='*80}")
    
    version_folders = [v for v in dataset_path.iterdir() if v.is_dir() and v.name.startswith('v')]
    
    if not version_folders:
        print("❌ No version folders (v1, v2, etc.) found in this dataset!")
        return 0, 0
    
    version_folders.sort(key=lambda x: int(x.name[1:]) if x.name[1:].isdigit() else 999)
    
    total_updated = 0
    total_fixed = 0
    
    for version_folder in version_folders:
        version_name = version_folder.name
        print(f"\n  📂 Version: {version_name}")
        
        json_files = [f for f in version_folder.glob("*.json") if '_old' not in f.name]
        
        version_updated = 0
        version_fixed = 0
        
        for json_file in json_files:
            if 'precision' in json_file.name or 'count' in json_file.name:
                continue
            
            try:
                fixed_data, fixed_count = fix_json_file(str(json_file))
                
                if fixed_count > 0:
                    backup_path = json_file.with_name(json_file.stem + '_old.json')
                    shutil.copy2(json_file, backup_path)
                    
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(fixed_data, f, indent=4, ensure_ascii=False)
                    
                    version_updated += 1
                    version_fixed += fixed_count
                
                pkl_file = json_file.with_suffix('.pkl')
                if pkl_file.exists():
                    pkl_backup = pkl_file.with_name(pkl_file.stem + '_old.pkl')
                    if not pkl_backup.exists():
                        shutil.copy2(pkl_file, pkl_backup)
                    
                    with open(pkl_file, 'wb') as f:
                        pkl.dump(fixed_data, f)
                    
                    if fixed_count > 0:
                        print(f"    ✅ Updated JSON and PKL (backed up)")
                    
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        if version_updated > 0:
            print(f"    📊 {version_name} Summary: {version_updated} file(s) updated, {version_fixed} answer(s) fixed")
        else:
            print(f"    ✓ No fixes needed in {version_name}")
        
        total_updated += version_updated
        total_fixed += version_fixed
    
    print(f"\n  ✨ {dataset_name} Total: {total_updated} file(s) updated, {total_fixed} answer(s) fixed")
    
    return total_updated, total_fixed

def main():
    import sys
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print("\n" + "="*80)
    print("🔧 Answer Extraction Fix Script")
    print("="*80)
    
    if len(sys.argv) > 1:
        dataset_folder = sys.argv[1]
    else:
        dataset_folder = DATASET_FOLDER
    
    if dataset_folder.startswith('outputs/'):
        dataset_folder = dataset_folder.replace('outputs/', '')
    
    dataset_path = project_root / "outputs" / dataset_folder
    
    print(f"📍 Target: {dataset_path}")
    print("="*80)
    
    total_files, total_fixes = process_single_dataset(dataset_path)
    
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    
    if total_files > 0:
        print(f"\n✅ Updated {total_files} file(s) with {total_fixes} answer(s) fixed")
        print(f"💾 All original files backed up with '_old.json' suffix")
        print(f"✨ Ready for sim precision calculations!")
    else:
        print("\n✓ No fixes needed - all files already have correct answers!")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

