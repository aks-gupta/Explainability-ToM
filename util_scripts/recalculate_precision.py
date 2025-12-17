import json
import sys
from pathlib import Path
import shutil

DATASET_FOLDER = "harmful-requests_llama3-2-90b-instruct_30_counterfactuals_TEMPLATE_BASED"

def calculate_precision_from_json(version_path):
    version_path = Path(version_path)
    
    simqa_files = [f for f in version_path.glob("*_simulation_question_answers_out_*.json") if '_old' not in f.name]
    taskqa_sim_files = [f for f in version_path.glob("*_task_qa_simulation_questions_out_*.json") if '_old' not in f.name]
    
    if not simqa_files or not taskqa_sim_files:
        return None
    
    simqa_file = simqa_files[0]
    taskqa_sim_file = taskqa_sim_files[0]
    
    with open(simqa_file, 'r') as f:
        simqa_data = json.load(f)
    
    with open(taskqa_sim_file, 'r') as f:
        taskqa_sim_data = json.load(f)
    
    count = 0
    simans_count = {}
    taskans_count = {}
    
    for qid in simqa_data:
        if isinstance(simqa_data[qid], list):
            for i, simqa_item in enumerate(simqa_data[qid]):
                pred_ans = simqa_item.get('pred_ans', 'neither')
                simans_count[count] = [str(pred_ans)]
                
                if qid in taskqa_sim_data and i < len(taskqa_sim_data[qid]):
                    taskqa_ans = taskqa_sim_data[qid][i].get('pred_ans', 'neither')
                else:
                    taskqa_ans = 'neither'
                taskans_count[count] = [str(taskqa_ans)]
                
                count += 1
    
    old_files = ['simans_count.json', 'taskans_count.json', 'precision_results.json']
    for old_file in old_files:
        old_path = version_path / old_file
        if old_path.exists():
            backup_path = version_path / f"{old_file.replace('.json', '_old.json')}"
            shutil.copy2(old_path, backup_path)
    
    with open(version_path / 'simans_count.json', 'w') as f:
        json.dump(simans_count, f)
    
    with open(version_path / 'taskans_count.json', 'w') as f:
        json.dump(taskans_count, f)
    
    ex_simulatable_count = 0
    ex_correct_simul_count = 0
    unknown_count_simqa = 0
    unknown_count_taskqa = 0
    unknown_set = set()
    
    for exidx in range(count):
        simqa_ann = simans_count[exidx][0]
        taskqa_pred = taskans_count[exidx][0]
        
        if simqa_ann in ['no', 'yes']:
            ex_simulatable_count += 1
            if simqa_ann == taskqa_pred:
                ex_correct_simul_count += 1
        else:
            unknown_count_simqa += 1
            unknown_set.add(simqa_ann)
        
        if taskqa_pred not in ['yes', 'no']:
            unknown_count_taskqa += 1
            unknown_set.add(taskqa_pred)
    
    precision = (ex_correct_simul_count / ex_simulatable_count * 100) if ex_simulatable_count > 0 else 0
    
    results = {
        "correctly_simulated": ex_correct_simul_count,
        "simulatable_count": ex_simulatable_count,
        "total_comparisons": count,
        "precision_percentage": round(precision, 1),
        "unknown_count_simqa": unknown_count_simqa,
        "unknown_count_taskqa": unknown_count_taskqa,
        "unknown_set": sorted(list(unknown_set))
    }
    
    with open(version_path / 'precision_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Correctly simulated: {ex_correct_simul_count}, Simulatable: {ex_simulatable_count}")
    print(f"Unknown count simqa: {unknown_count_simqa}")
    print(f"Unknown count taskqa: {unknown_count_taskqa}")
    print(f"Precision: {round(precision, 1)}%")
    
    return precision

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    if len(sys.argv) >= 2:
        dataset_folder = sys.argv[1]
    else:
        dataset_folder = DATASET_FOLDER
    
    if dataset_folder.startswith('outputs/'):
        dataset_folder = dataset_folder.replace('outputs/', '')
    
    dataset_path = project_root / "outputs" / dataset_folder
    
    if not dataset_path.exists():
        print(f"Folder not found: {dataset_path}")
        return
    
    print(f"Calculating precision for folder: {dataset_path}")
    print("="*80)
    
    version_folders = sorted([v for v in dataset_path.iterdir() if v.is_dir() and v.name.startswith('v')],
                            key=lambda x: int(x.name[1:]) if x.name[1:].isdigit() else 999)
    
    if not version_folders:
        print("No version folders found")
        return
    
    for version_folder in version_folders:
        print(f"\n{version_folder.name}:")
        precision = calculate_precision_from_json(version_folder)
        if precision is None:
            print("  Missing required files")

if __name__ == "__main__":
    main()
