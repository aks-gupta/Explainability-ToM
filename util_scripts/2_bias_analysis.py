import json
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================
# UTILITY FUNCTIONS
# ============================================

def norm(ans):
    """Normalize answer to 'yes', 'no', or 'other'."""
    x = (ans or "").strip().lower()
    if x.startswith("y"):
        return "yes"
    if x.startswith("n"):
        return "no"
    return "other"


def load_json(path):
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def compute_confusion_matrix(st):
    """Compute confusion matrix components from stats dictionary."""
    TP = st["sim_pred_switch"] - st["sim_pred_but_no_switch"]
    FP = st["sim_pred_but_no_switch"]
    FN = st["sim_no_pred_but_switched"]
    TN = st["total"] - (TP + FP + FN)
    
    if TP < 0:
        TP = 0
    
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN}


# ============================================
# DATA LOADING
# ============================================

def load_templates(domain):
    """Load fixed question templates."""
    fixed_qs_path = f'templates/{domain}_fixed_qs.json'
    return load_json(fixed_qs_path)['questions']


def load_counterfactuals(domain):
    """Load counterfactual data."""
    counterfactuals_path = f'templates/counterfactuals_output_{domain}.json'
    return load_json(counterfactuals_path)


def load_model_outputs(subfolder, domain):
    """Load taskqa, simqa, and tasksim files from subfolder."""
    outputs = {}
    
    for file in os.listdir(subfolder):
        file_path = os.path.join(subfolder, file)
        
        if file.startswith(f"{domain}_task_qa_out") and file.endswith('.json'):
            outputs['taskqa'] = load_json(file_path)
        elif file.startswith(f"{domain}_simulation_question") and file.endswith('.json'):
            outputs['simqa'] = load_json(file_path)
        elif file.startswith(f"{domain}_task_qa_simulation") and file.endswith('.json'):
            outputs['tasksim'] = load_json(file_path)
    
    return outputs


def group_data(outputs, cf_data, fixed_qs):
    """Group results by integer keys from taskqa, simqa, tasksim, and cf_data."""
    grouped = {}
    
    def add_to_group(d, type_name):
        for k, v in d.items():
            ik = int(k)
            if ik not in grouped:
                grouped[ik] = {}
            grouped[ik][type_name] = v
    
    # Add model outputs
    for key in ['taskqa', 'simqa', 'tasksim']:
        if key in outputs:
            add_to_group(outputs[key], key)
    
    # Add counterfactual data
    template_map = {qs["template_id"]: qs for qs in fixed_qs}
    
    for k, v in cf_data.items():
        ik = int(k)
        grouped.setdefault(ik, {})
        grouped[ik]["cf"] = {
            "question": v["question"],
            "counterfactual_questions": v["counterfactual_questions"],
            "template_id": v["template_id"],
            "qid": v["qid"]
        }
        
        tpl_id = v["template_id"]
        if tpl_id in template_map:
            grouped[ik]["template"] = template_map[tpl_id]
    
    return grouped


# ============================================
# STATISTICS COMPUTATION
# ============================================

def initialize_stats():
    """Initialize statistics dictionary."""
    return {
        "yes": 0, "no": 0, "total": 0,
        "switched": 0,
        "sim_pred_switch": 0,
        "sim_pred_but_no_switch": 0,
        "sim_no_pred_but_switched": 0,
    }


def initialize_qid_stats():
    """Initialize QID-specific statistics dictionary."""
    return {
        "yes": 0, "no": 0, "total": 0,
        "switched": 0,
        "sim_pred": 0,
        "sim_pred_switch": 0,
        "sim_pred_but_no_switch": 0,
        "sim_no_pred_but_switched": 0,
        "TP": 0, "FP": 0, "FN": 0, "TN": 0
    }


def compute_statistics(grouped):
    """Compute all statistics from grouped data."""
    stats_a = defaultdict(initialize_stats)
    stats_b = defaultdict(initialize_stats)
    cf_b_stats = defaultdict(initialize_stats)
    orig_to_cf_switch = defaultdict(lambda: defaultdict(int))
    orig_to_cf_qid = {0: defaultdict(lambda: defaultdict(int)),
                      1: defaultdict(lambda: defaultdict(int))}
    
    global_stats = {
        "total": 0,
        "switched": 0,
        "sim_pred_switch": 0,
        "sim_pred_but_no_switch": 0,
        "sim_no_pred_but_switched": 0
    }
    
    qid_stats_A = {0: initialize_qid_stats(), 1: initialize_qid_stats()}
    qid_stats_B = {0: initialize_qid_stats(), 1: initialize_qid_stats()}
    
    for item in grouped.values():
        tpl = item.get("template", {})
        cf = item.get("cf", {})
        
        question_text = cf.get("question", "")
        qid = cf.get("qid")
        
        possible_a = tpl.get("variables", {}).get("possible_values", {}).get("a", [])
        possible_b = tpl.get("variables", {}).get("possible_values", {}).get("b", [])
        
        used_a = next((a for a in possible_a if a in question_text), None)
        used_b = next((b for b in possible_b if b in question_text), None)
        
        if used_a is None or used_b is None:
            continue
        
        og = norm(item.get("taskqa", {}).get("pred_ans", None))
        sim_list = [norm(a.get("pred_ans")) for a in item.get("simqa", [])]
        tasksim_list = [norm(a.get("pred_ans")) for a in item.get("tasksim", [])]
        
        if not sim_list or not tasksim_list:
            continue
        
        cf_questions = item.get("cf", {}).get("counterfactual_questions", [])
        
        for idx, (sim_ans, task_ans) in enumerate(zip(sim_list, tasksim_list)):
            switched = (task_ans != og)
            sim_pred = (sim_ans != og)
            
            # Update basic a/b stats
            for stats_dict in [stats_a[used_a], stats_b[used_b]]:
                stats_dict["total"] += 1
                if og == "yes":
                    stats_dict["yes"] += 1
                elif og == "no":
                    stats_dict["no"] += 1
                
                if switched:
                    stats_dict["switched"] += 1
                if sim_pred:
                    stats_dict["sim_pred_switch"] += 1
                if sim_pred and not switched:
                    stats_dict["sim_pred_but_no_switch"] += 1
                if switched and not sim_pred:
                    stats_dict["sim_no_pred_but_switched"] += 1
            
            # Update counterfactual-b stats
            cf_b = None
            if idx < len(cf_questions):
                cf_q = cf_questions[idx]
                cf_b = next((b for b in possible_b if b in cf_q), None)
            
            if cf_b is not None:
                orig_to_cf_switch[used_b][cf_b] += 1 if switched else 0
                
                bucket = cf_b_stats[cf_b]
                bucket["total"] += 1
                if og == "yes":
                    bucket["yes"] += 1
                elif og == "no":
                    bucket["no"] += 1
                
                if switched:
                    bucket["switched"] += 1
                if sim_pred:
                    bucket["sim_pred_switch"] += 1
                if sim_pred and not switched:
                    bucket["sim_pred_but_no_switch"] += 1
                if switched and not sim_pred:
                    bucket["sim_no_pred_but_switched"] += 1
            
            # Update global stats
            global_stats["total"] += 1
            if switched:
                global_stats["switched"] += 1
            if sim_pred:
                global_stats["sim_pred_switch"] += 1
            if sim_pred and not switched:
                global_stats["sim_pred_but_no_switch"] += 1
            if switched and not sim_pred:
                global_stats["sim_no_pred_but_switched"] += 1
            
            # Update qid-specific stats for Option A
            if qid in (0, 1) and used_a:
                qid_stats_A[qid]["total"] += 1
                if switched:
                    qid_stats_A[qid]["switched"] += 1
                if sim_pred:
                    qid_stats_A[qid]["sim_pred"] += 1
                
                if sim_pred and switched:
                    qid_stats_A[qid]["TP"] += 1
                elif sim_pred and not switched:
                    qid_stats_A[qid]["FP"] += 1
                elif (not sim_pred) and switched:
                    qid_stats_A[qid]["FN"] += 1
                else:
                    qid_stats_A[qid]["TN"] += 1
            
            # Update qid-specific stats for Option B
            if qid in (0, 1) and used_b:
                qid_stats_B[qid]["total"] += 1
                if switched:
                    qid_stats_B[qid]["switched"] += 1
                if sim_pred:
                    qid_stats_B[qid]["sim_pred"] += 1
                
                if sim_pred and switched:
                    qid_stats_B[qid]["TP"] += 1
                elif sim_pred and not switched:
                    qid_stats_B[qid]["FP"] += 1
                elif (not sim_pred) and switched:
                    qid_stats_B[qid]["FN"] += 1
                else:
                    qid_stats_B[qid]["TN"] += 1
            
            # Update qid-specific orig_to_cf switch matrix
            if qid in (0, 1) and cf_b:
                orig_to_cf_qid[qid][used_b][cf_b] += 1 if switched else 0
    
    return {
        'stats_a': stats_a,
        'stats_b': stats_b,
        'cf_b_stats': cf_b_stats,
        'orig_to_cf_switch': orig_to_cf_switch,
        'orig_to_cf_qid': orig_to_cf_qid,
        'global_stats': global_stats,
        'qid_stats_A': qid_stats_A,
        'qid_stats_B': qid_stats_B
    }


# ============================================
# PRINTING FUNCTIONS
# ============================================

def print_templates_info(fixed_qs):
    """Print template information."""
    for qs in fixed_qs:
        option_a = qs['variables']['possible_values']['a']
        option_b = qs['variables']['possible_values']['b']
        print(f"Template ID: {qs['template_id']}, Possible [a]: {option_a}, Possible [b]: {option_b}")


def print_table(title, stats_dict):
    """Print statistics table."""
    print("\n==============================")
    print(title)
    print("==============================")
    print(f"{'Option':25} | YES%   | NO%    | Sw%    | SimPred% | Pred_NoSw% | NoPred_Sw%")
    print("-" * 95)
    
    for opt, st in stats_dict.items():
        tot = st["total"] or 1
        yes_p = st["yes"] / tot
        no_p = st["no"] / tot
        sw_p = st["switched"] / tot
        sp_p = st["sim_pred_switch"] / tot
        sp_no = st["sim_pred_but_no_switch"] / tot
        no_sp = st["sim_no_pred_but_switched"] / tot
        
        print(f"{opt:25} | {yes_p:0.2f} | {no_p:0.2f} | {sw_p:0.2f} | {sp_p:0.2f} | {sp_no:0.2f} | {no_sp:0.2f}")


def print_confusion_matrix(label, stats_dict):
    """Print confusion matrix for each option."""
    print(f"\n==============================")
    print(f"CONFUSION MATRICES for {label}")
    print(f"==============================")
    
    for opt, st in stats_dict.items():
        matrix = compute_confusion_matrix(st)
        print(f"\nOption: {opt}")
        print("┌───────────────┬─────────┬────────┐")
        print(f"│               │ Pred Sw │ Pred No │")
        print("├───────────────┼─────────┼────────┤")
        print(f"│ Actual Sw     │   {matrix['TP']:3d}   │   {matrix['FN']:3d}   │")
        print("├───────────────┼─────────┼────────┤")
        print(f"│ Actual No Sw  │   {matrix['FP']:3d}   │   {matrix['TN']:3d}   │")
        print("└───────────────┴─────────┴────────┘")


def print_global_confusion_matrix(global_stats):
    """Print global confusion matrix."""
    TP = global_stats["sim_no_pred_but_switched"]
    FP = global_stats["sim_pred_but_no_switch"]
    FN = global_stats["sim_no_pred_but_switched"]
    TN = global_stats["total"] - (TP + FP + FN)
    
    print("\n==============================")
    print("GLOBAL CONFUSION MATRIX")
    print("==============================")
    print("┌───────────────┬─────────┬────────┐")
    print("│               │ Pred Sw │ Pred No │")
    print("├───────────────┼─────────┼────────┤")
    print(f"│ Actual Sw     │   {TP:3d}   │   {FN:3d}   │")
    print("├───────────────┼─────────┼────────┤")
    print(f"│ Actual No Sw  │   {FP:3d}   │   {TN:3d}   │")
    print("└───────────────┴─────────┴────────┘")
    print("\nGLOBAL SUMMARY:")
    print(global_stats)
    
    return TP, FP, FN, TN


def print_cf_confusion_matrices(cf_b_stats, orig_to_cf_switch, orig_to_cf_qid):
    """Print counterfactual confusion matrices."""
    print("\n\n==============================")
    print("CONFUSION MATRIX #1: ACTUAL vs PREDICTED SWITCH (per CF-b)")
    print("==============================")
    print(f"{'CF_b':25} |  TP  |  FP  |  TN  |  FN")
    print("-" * 60)
    
    for cf_b, st in cf_b_stats.items():
        matrix = compute_confusion_matrix(st)
        print(f"{cf_b:25} | {matrix['TP']:4} | {matrix['FP']:4} | {matrix['TN']:4} | {matrix['FN']:4}")
    
    print("\n\n==============================")
    print("CONFUSION MATRIX #2: ORIGINAL_b → CF_b SWITCH MATRIX")
    print("==============================")
    
    all_cf_b = sorted(set(cf_b_stats.keys()))
    all_orig_b = sorted(set(orig_to_cf_switch.keys()))
    
    header = " " * 20 + " | " + " | ".join([f"{b[:12]:12}" for b in all_cf_b])
    print(header)
    print("-" * len(header))
    
    for ob in all_orig_b:
        row = f"{ob[:20]:20} | " + " | ".join([
            f"{orig_to_cf_switch[ob].get(cb, 0):12}" for cb in all_cf_b
        ])
        print(row)
    
    # QID-specific matrices
    for qid in (0, 1):
        print(f"\n\n---- QID = {qid} ----")
        orig_b_qid = sorted(orig_to_cf_qid[qid].keys())
        cf_b_qid = sorted({cb for ob in orig_b_qid for cb in orig_to_cf_qid[qid][ob].keys()})
        
        header = " " * 20 + " | " + " | ".join([f"{b[:12]:12}" for b in cf_b_qid])
        print(header)
        print("-" * len(header))
        
        for ob in orig_b_qid:
            row = f"{ob[:20]:20} | " + " | ".join([
                f"{orig_to_cf_qid[qid][ob].get(cb, 0):12}" for cb in cf_b_qid
            ])
            print(row)
    
    print("\n\n==============================")
    print("CONFUSION MATRIX #3: CF-b SIMPLE SWITCH COUNTS")
    print("==============================")
    print(f"{'CF_b':25} | Switch | NoSwitch")
    print("-" * 60)
    
    for cf_b, st in cf_b_stats.items():
        sw = st["switched"]
        nosw = st["total"] - sw
        print(f"{cf_b:25} | {sw:6} | {nosw:8}")


# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def save_global_confusion_heatmap(TP, FP, FN, TN, output_dir):
    """Save global confusion matrix heatmap."""
    cm = np.array([[TP, FN], [FP, TN]])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=["Pred Switch", "Pred No Switch"],
        yticklabels=["Actual Switch", "Actual No Switch"],
        cbar=False
    )
    plt.title("Global Confusion Matrix – Switch Prediction")
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "global_confusion_matrix.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")


def save_option_a_aggregate_confusion(stats_a, output_dir):
    """Save aggregated Option A confusion matrix."""
    A_TP = A_FP = A_FN = A_TN = 0
    
    for a_val, st in stats_a.items():
        if st["total"] == 0:
            continue
        
        matrix = compute_confusion_matrix(st)
        A_TP += matrix["TP"]
        A_FP += matrix["FP"]
        A_FN += matrix["FN"]
        A_TN += matrix["TN"]
    
    cmA = np.array([[A_TP, A_FN], [A_FP, A_TN]])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cmA,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Pred Switch", "Pred No Switch"],
        yticklabels=["Actual Switch", "Actual No Switch"],
        cbar=False
    )
    plt.title("Confusion Matrix – Aggregated over Option [A]")
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "confusion_matrix_optionA_aggregate.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")


def save_option_a_bias_heatmap(stats_a, output_dir):
    """Save Option A bias metrics heatmap."""
    matrix_rows = []
    row_labels = []
    
    for a_val, st in stats_a.items():
        tot = st["total"] or 1
        Sw = st["switched"] / tot
        FP = st["sim_pred_but_no_switch"] / tot
        FN = st["sim_no_pred_but_switched"] / tot
        
        matrix_rows.append([Sw, FP, FN])
        row_labels.append(a_val)
    
    df_a_bias = pd.DataFrame(matrix_rows, index=row_labels, columns=["Sw%", "FP%", "FN%"])
    
    plt.figure(figsize=(12, 9))
    sns.heatmap(df_a_bias, annot=True, cmap="Oranges", linewidths=.5, fmt=".2f", cbar=True)
    plt.title("Option-A Bias Metrics (Switch / FP / FN) Normalized", fontsize=16)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "optionA_bias_heatmap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Saved] {out_path}")


def save_cf_heatmaps(cf_b_stats, orig_to_cf_switch, orig_to_cf_qid, output_dir):
    """Save counterfactual-related heatmaps."""
    cf_b_list = sorted(cf_b_stats.keys())
    
    # 1. Switch rate heatmap
    switch_rates = [cf_b_stats[b]["switched"] / (cf_b_stats[b]["total"] or 1) for b in cf_b_list]
    
    plt.figure(figsize=(12, 1.5))
    sns.heatmap(
        [switch_rates],
        annot=True,
        xticklabels=cf_b_list,
        yticklabels=["Switch Rate"],
        cmap="Reds",
        vmin=0,
        vmax=1
    )
    plt.title("Switch Rate per CF-b")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_cf_b_switch_rate.png"))
    plt.close()
    
    # 2. Original to CF switch matrix
    orig_b_list = sorted(orig_to_cf_switch.keys())
    matrix = [[orig_to_cf_switch[ob].get(cb, 0) for cb in cf_b_list] for ob in orig_b_list]
    
    plt.figure(figsize=(18, 10))
    sns.heatmap(
        matrix,
        annot=True,
        xticklabels=cf_b_list,
        yticklabels=orig_b_list,
        cmap="Reds"
    )
    plt.title("Original_b → CF_b Switch Count Matrix")
    plt.xlabel("Counterfactual b")
    plt.ylabel("Original b")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_orig_to_cf_switch.png"))
    plt.close()
    
    # 3. Simulation confusion metrics
    cf_b_TP = []
    cf_b_FP = []
    cf_b_FN = []
    
    for b in cf_b_list:
        matrix = compute_confusion_matrix(cf_b_stats[b])
        cf_b_TP.append(matrix["TP"])
        cf_b_FP.append(matrix["FP"])
        cf_b_FN.append(matrix["FN"])
    
    metric_matrix = np.array([cf_b_TP, cf_b_FP, cf_b_FN])
    
    plt.figure(figsize=(14, 4))
    sns.heatmap(
        metric_matrix,
        annot=True,
        xticklabels=cf_b_list,
        yticklabels=["TP", "FP", "FN"],
        cmap="Purples"
    )
    plt.title("Simulation Model Confusion Metrics per CF-b")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_cf_b_sim_confusion.png"))
    plt.close()
    
    # 4. Combined metrics
    combined_matrix = []
    for b in cf_b_list:
        st = cf_b_stats[b]
        tot = st["total"] or 1
        sw = st["switched"] / tot
        fp = st["sim_pred_but_no_switch"] / tot
        fn = st["sim_no_pred_but_switched"] / tot
        combined_matrix.append([sw, fp, fn])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        np.array(combined_matrix),
        annot=True,
        xticklabels=["Sw%", "FP%", "FN%"],
        yticklabels=cf_b_list,
        cmap="Oranges",
        vmin=0,
        vmax=1
    )
    plt.title("CF-b Bias Metrics (Switch / FP / FN) Normalized")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_cf_b_metrics.png"))
    plt.close()
    
    # 5. QID-specific heatmaps
    for qid in (0, 1):
        orig_to_cf_dict = orig_to_cf_qid[qid]
        orig_list = sorted(orig_to_cf_dict.keys())
        cf_list = sorted({cb for ob in orig_list for cb in orig_to_cf_dict[ob].keys()})
        
        if not orig_list or not cf_list:
            print(f"[Warning] QID {qid} has empty b-switch matrix. Skipping heatmap.")
            continue
        
        matrix = [[orig_to_cf_dict[ob].get(cb, 0) for cb in cf_list] for ob in orig_list]
        df_matrix = pd.DataFrame(matrix, index=orig_list, columns=cf_list)
        
        plt.figure(figsize=(18, 10))
        sns.heatmap(
            df_matrix,
            annot=True,
            cmap="Reds",
            xticklabels=cf_list,
            yticklabels=orig_list,
            fmt="d",
            linewidths=.5
        )
        plt.title(f"Original_b → CF_b Switch Count Matrix (QID = {qid})", fontsize=16)
        plt.xlabel("Counterfactual b")
        plt.ylabel("Original b")
        plt.tight_layout()
        
        outpath = os.path.join(output_dir, f"qid_{qid}_orig_to_cf_switch_heatmap.png")
        plt.savefig(outpath, dpi=300)
        plt.close()
        print(f"[Saved] {outpath}")


def save_qid_pie_charts(qid_stats, option_label, output_dir):
    """Save QID-specific pie charts."""
    def save_pie(qid, labels, sizes, title, filename):
        plt.figure(figsize=(5, 5))
        plt.pie(
            sizes,
            labels=[f"{l} ({s})" for l, s in zip(labels, sizes)],
            autopct="%1.1f%%",
            startangle=140
        )
        plt.title(f"{title} — QID {qid} (Option {option_label})")
        plt.tight_layout()
        out = os.path.join(output_dir, f"{filename}_{option_label.lower()}_qid{qid}.png")
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[Saved] {out}")
    
    for qid in (0, 1):
        st = qid_stats[qid]
        tot = st["total"] or 1
        
        # Actual switch breakdown
        save_pie(
            qid,
            ["Switch", "No Switch"],
            [st["switched"], tot - st["switched"]],
            "Actual Switch Breakdown",
            "actual_switch"
        )
        
        # Simulation predictions
        save_pie(
            qid,
            ["Pred Switch", "Pred No"],
            [st["sim_pred"], tot - st["sim_pred"]],
            "Simulation Predictions",
            "sim_predictions"
        )
        
        # Confusion matrix slices
        save_pie(
            qid,
            ["TP", "FP", "FN", "TN"],
            [st["TP"], st["FP"], st["FN"], st["TN"]],
            "Confusion Matrix Slices",
            "confusion_slices"
        )

# -----------------------------------------------
# SNIPPET 1: Add this function after save_qid_pie_charts() function
# -----------------------------------------------

def compute_simulatability_by_option(grouped):
    """
    Compute simulatability statistics for each option [a] and [b] value, split by QID.
    
    CORRECTED DEFINITION:
    Simulatable = sim_ans == task_ans (simulation's answer matches the actual task answer for CF)
    This is independent of whether a switch occurred.
    """
    # Structure: {qid: {option_value: {"simulatable": count, "not_simulatable": count}}}
    simulatability_a = {0: defaultdict(lambda: {"simulatable": 0, "not_simulatable": 0}),
                        1: defaultdict(lambda: {"simulatable": 0, "not_simulatable": 0})}
    
    simulatability_b = {0: defaultdict(lambda: {"simulatable": 0, "not_simulatable": 0}),
                        1: defaultdict(lambda: {"simulatable": 0, "not_simulatable": 0})}
    
    for item in grouped.values():
        tpl = item.get("template", {})
        cf = item.get("cf", {})
        
        qid = cf.get("qid")
        if qid not in (0, 1):
            continue
        
        question_text = cf.get("question", "")
        possible_a = tpl.get("variables", {}).get("possible_values", {}).get("a", [])
        possible_b = tpl.get("variables", {}).get("possible_values", {}).get("b", [])
        
        used_a = next((a for a in possible_a if a in question_text), None)
        used_b = next((b for b in possible_b if b in question_text), None)
        
        if used_a is None or used_b is None:
            continue
        
        sim_list = [norm(a.get("pred_ans")) for a in item.get("simqa", [])]
        tasksim_list = [norm(a.get("pred_ans")) for a in item.get("tasksim", [])]
        
        if not sim_list or not tasksim_list:
            continue
        
        for sim_ans, task_ans in zip(sim_list, tasksim_list):
            # CORRECTED: Simulatable if simulation answer matches task answer for CF
            is_simulatable = (sim_ans == task_ans)
            
            # Update option A stats
            if is_simulatable:
                simulatability_a[qid][used_a]["simulatable"] += 1
            else:
                simulatability_a[qid][used_a]["not_simulatable"] += 1
            
            # Update option B stats
            if is_simulatable:
                simulatability_b[qid][used_b]["simulatable"] += 1
            else:
                simulatability_b[qid][used_b]["not_simulatable"] += 1
    
    return simulatability_a, simulatability_b

def compute_cf_simulatability_by_original_b(grouped):
    """
    For each original option [b], compute simulatability stats for each counterfactual [b].
    Split by QID.
    
    Returns: {qid: {original_b: {cf_b: {"simulatable": count, "not_simulatable": count}}}}
    """
    cf_simulatability = {
        0: defaultdict(lambda: defaultdict(lambda: {"simulatable": 0, "not_simulatable": 0})),
        1: defaultdict(lambda: defaultdict(lambda: {"simulatable": 0, "not_simulatable": 0}))
    }
    
    for item in grouped.values():
        tpl = item.get("template", {})
        cf = item.get("cf", {})
        
        qid = cf.get("qid")
        if qid not in (0, 1):
            continue
        
        question_text = cf.get("question", "")
        possible_b = tpl.get("variables", {}).get("possible_values", {}).get("b", [])
        
        # Get original b value
        used_b = next((b for b in possible_b if b in question_text), None)
        if used_b is None:
            continue
        
        sim_list = [norm(a.get("pred_ans")) for a in item.get("simqa", [])]
        tasksim_list = [norm(a.get("pred_ans")) for a in item.get("tasksim", [])]
        cf_questions = item.get("cf", {}).get("counterfactual_questions", [])
        
        if not sim_list or not tasksim_list:
            continue
        
        for idx, (sim_ans, task_ans) in enumerate(zip(sim_list, tasksim_list)):
            # Get the CF b value for this counterfactual
            cf_b = None
            if idx < len(cf_questions):
                cf_q = cf_questions[idx]
                cf_b = next((b for b in possible_b if b in cf_q), None)
            
            if cf_b is None:
                continue
            
            # Simulatable if simulation answer matches task answer
            is_simulatable = (sim_ans == task_ans)
            
            if is_simulatable:
                cf_simulatability[qid][used_b][cf_b]["simulatable"] += 1
            else:
                cf_simulatability[qid][used_b][cf_b]["not_simulatable"] += 1
    
    return cf_simulatability


# -----------------------------------------------
# SNIPPET 3: Function to save CF simulatability pie charts
# Add this after compute_cf_simulatability_by_original_b()
# -----------------------------------------------

def save_cf_simulatability_pie_charts(cf_simulatability, output_dir):
    """
    Save pie charts showing simulatability for each CF option [b] grouped by original option [b].
    One chart per (qid, original_b, cf_b) combination.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for qid in (0, 1):
        qid_data = cf_simulatability[qid]
        
        for original_b in sorted(qid_data.keys()):
            cf_data = qid_data[original_b]
            
            for cf_b in sorted(cf_data.keys()):
                stats = cf_data[cf_b]
                simulatable = stats["simulatable"]
                not_simulatable = stats["not_simulatable"]
                total = simulatable + not_simulatable
                
                if total == 0:
                    continue
                
                # Create pie chart
                plt.figure(figsize=(7, 7))
                sizes = [simulatable, not_simulatable]
                labels = [f"Simulatable ({simulatable})", f"Not Simulatable ({not_simulatable})"]
                colors = ['#3498db', '#e74c3c']  # Blue for simulatable, red for not
                explode = (0.05, 0)
                
                plt.pie(
                    sizes,
                    labels=labels,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors,
                    explode=explode,
                    shadow=True,
                    textprops={'fontsize': 11, 'weight': 'bold'}
                )
                
                # Calculate percentage
                sim_pct = (simulatable / total) * 100
                not_sim_pct = (not_simulatable / total) * 100
                
                plt.title(
                    f"CF Simulatability: {original_b} → {cf_b}\n"
                    f"QID {qid} | Not Simulatable: {not_sim_pct:.1f}%",
                    fontsize=13,
                    weight='bold',
                    pad=20
                )
                plt.tight_layout()
                
                # Safe filename
                safe_orig = original_b.replace('/', '_').replace(' ', '_').replace('-', '_')
                safe_cf = cf_b.replace('/', '_').replace(' ', '_').replace('-', '_')
                filename = f"cf_sim_qid{qid}_{safe_orig}_to_{safe_cf}.png"
                out_path = os.path.join(output_dir, filename)
                plt.savefig(out_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"[Saved] {out_path}")


# -----------------------------------------------
# SNIPPET 4: Function to save summary heatmap of CF simulatability
# Add this after save_cf_simulatability_pie_charts()
# -----------------------------------------------

def save_cf_simulatability_heatmap(cf_simulatability, output_dir):
    """
    Save heatmap showing NOT simulatable percentage for each (original_b, cf_b) pair.
    One heatmap per QID.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for qid in (0, 1):
        qid_data = cf_simulatability[qid]
        
        if not qid_data:
            continue
        
        # Get all original and CF b values
        all_original_b = sorted(qid_data.keys())
        all_cf_b = sorted(set(
            cf_b for orig_data in qid_data.values() 
            for cf_b in orig_data.keys()
        ))
        
        if not all_original_b or not all_cf_b:
            continue
        
        # Build matrix of NOT simulatable percentages
        matrix = []
        for orig_b in all_original_b:
            row = []
            for cf_b in all_cf_b:
                stats = qid_data[orig_b].get(cf_b, {"simulatable": 0, "not_simulatable": 0})
                total = stats["simulatable"] + stats["not_simulatable"]
                
                if total > 0:
                    not_sim_pct = (stats["not_simulatable"] / total) * 100
                else:
                    not_sim_pct = 0
                
                row.append(not_sim_pct)
            matrix.append(row)
        
        # Create heatmap
        plt.figure(figsize=(max(14, len(all_cf_b) * 0.8), max(8, len(all_original_b) * 0.6)))
        
        # Create DataFrame for better labels
        df_matrix = pd.DataFrame(matrix, index=all_original_b, columns=all_cf_b)
        
        sns.heatmap(
            df_matrix,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn_r',  # Red = high not simulatable, Green = low not simulatable
            vmin=0,
            vmax=100,
            cbar_kws={'label': 'Not Simulatable (%)'},
            linewidths=0.5,
            linecolor='gray'
        )
        
        plt.title(f'Counterfactual Not Simulatable % | QID {qid}', fontsize=16, weight='bold', pad=15)
        plt.xlabel('Counterfactual [b]', fontsize=13, weight='bold')
        plt.ylabel('Original [b]', fontsize=13, weight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        filename = f"cf_not_simulatable_heatmap_qid{qid}.png"
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[Saved] {out_path}")


# -----------------------------------------------
# SNIPPET 5: Function to save grouped bar chart
# Add this after save_cf_simulatability_heatmap()
# -----------------------------------------------

def save_cf_not_simulatable_grouped_bars(cf_simulatability, output_dir):
    """
    For each original [b], create a grouped bar chart showing NOT simulatable %
    for each CF [b] option.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for qid in (0, 1):
        qid_data = cf_simulatability[qid]
        
        for original_b in sorted(qid_data.keys()):
            cf_data = qid_data[original_b]
            
            if not cf_data:
                continue
            
            # Prepare data
            cf_options = []
            not_sim_percentages = []
            
            for cf_b in sorted(cf_data.keys()):
                stats = cf_data[cf_b]
                total = stats["simulatable"] + stats["not_simulatable"]
                
                if total > 0:
                    not_sim_pct = (stats["not_simulatable"] / total) * 100
                    cf_options.append(cf_b)
                    not_sim_percentages.append(not_sim_pct)
            
            if not cf_options:
                continue
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(max(10, len(cf_options) * 0.5), 6))
            
            # Color based on percentage
            colors = []
            for pct in not_sim_percentages:
                if pct >= 70:
                    colors.append('#e74c3c')  # Red - high not simulatable
                elif pct >= 40:
                    colors.append('#f39c12')  # Orange
                else:
                    colors.append('#2ecc71')  # Green - low not simulatable
            
            bars = ax.bar(range(len(cf_options)), not_sim_percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for i, (bar, pct) in enumerate(zip(bars, not_sim_percentages)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{pct:.1f}%',
                       ha='center', va='bottom', fontsize=9, weight='bold')
            
            ax.set_ylabel('Not Simulatable (%)', fontsize=12, weight='bold')
            ax.set_xlabel('Counterfactual [b] Options', fontsize=12, weight='bold')
            ax.set_title(f'Not Simulatable % for CF [b] | Original: {original_b} | QID {qid}', 
                        fontsize=13, weight='bold')
            ax.set_xticks(range(len(cf_options)))
            ax.set_xticklabels(cf_options, rotation=45, ha='right')
            ax.set_ylim(0, 105)
            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            safe_orig = original_b.replace('/', '_').replace(' ', '_').replace('-', '_')
            filename = f"cf_not_sim_bars_{safe_orig}_qid{qid}.png"
            out_path = os.path.join(output_dir, filename)
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[Saved] {out_path}")



# -----------------------------------------------
# SNIPPET 2: Add this function after compute_simulatability_by_option()
# -----------------------------------------------

def save_simulatability_pie_charts(simulatability_stats, option_label, output_dir):
    """
    Save pie charts showing simulatability percentage for each option value, split by QID.
    
    Args:
        simulatability_stats: dict with structure {qid: {option_value: {"simulatable": count, "not_simulatable": count}}}
        option_label: "A" or "B"
        output_dir: directory to save charts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for qid in (0, 1):
        qid_data = simulatability_stats[qid]
        
        # Sort options by name for consistent ordering
        sorted_options = sorted(qid_data.keys())
        
        for option_val in sorted_options:
            stats = qid_data[option_val]
            simulatable = stats["simulatable"]
            not_simulatable = stats["not_simulatable"]
            total = simulatable + not_simulatable
            
            if total == 0:
                continue
            
            # Create pie chart
            plt.figure(figsize=(6, 6))
            sizes = [simulatable, not_simulatable]
            labels = [f"Simulatable ({simulatable})", f"Not Simulatable ({not_simulatable})"]
            colors = ['#2ecc71', '#e74c3c']  # Green for simulatable, red for not
            explode = (0.05, 0)  # Slightly separate the simulatable slice
            
            plt.pie(
                sizes,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=colors,
                explode=explode,
                shadow=True
            )
            
            # Calculate percentage for title
            sim_pct = (simulatable / total) * 100
            
            plt.title(f"Simulatability: {option_val}\nOption [{option_label}] | QID {qid} | {sim_pct:.1f}% Simulatable", 
                     fontsize=12, weight='bold')
            plt.tight_layout()
            
            # Safe filename (replace special characters)
            safe_option_val = option_val.replace('/', '_').replace(' ', '_')
            filename = f"simulatability_{option_label.lower()}_{safe_option_val}_qid{qid}.png"
            out_path = os.path.join(output_dir, filename)
            plt.savefig(out_path, dpi=300)
            plt.close()
            
            print(f"[Saved] {out_path}")


# -----------------------------------------------
# SNIPPET 3: Add this function after save_simulatability_pie_charts()
# -----------------------------------------------

def save_simulatability_summary_charts(simulatability_stats, option_label, output_dir):
    """
    Save summary bar charts showing simulatability percentage for all options in one chart per QID.
    
    Args:
        simulatability_stats: dict with structure {qid: {option_value: {"simulatable": count, "not_simulatable": count}}}
        option_label: "A" or "B"
        output_dir: directory to save charts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for qid in (0, 1):
        qid_data = simulatability_stats[qid]
        
        if not qid_data:
            continue
        
        # Sort options by simulatability percentage (descending)
        option_percentages = []
        for option_val, stats in qid_data.items():
            total = stats["simulatable"] + stats["not_simulatable"]
            if total > 0:
                pct = (stats["simulatable"] / total) * 100
                option_percentages.append((option_val, pct, stats["simulatable"], total))
        
        # Sort by percentage descending
        option_percentages.sort(key=lambda x: x[1], reverse=True)
        
        if not option_percentages:
            continue
        
        # Extract data for plotting
        options = [x[0] for x in option_percentages]
        percentages = [x[1] for x in option_percentages]
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, max(6, len(options) * 0.4)))
        
        # Color bars based on percentage (gradient from red to green)
        colors = []
        for pct in percentages:
            if pct >= 80:
                colors.append('#2ecc71')  # Green
            elif pct >= 60:
                colors.append('#f39c12')  # Orange
            else:
                colors.append('#e74c3c')  # Red
        
        bars = ax.barh(options, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add percentage labels on bars
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f'{pct:.1f}%', 
                   ha='left', va='center', fontsize=10, weight='bold')
        
        ax.set_xlabel('Simulatability (%)', fontsize=12, weight='bold')
        ax.set_ylabel(f'Option [{option_label}] Values', fontsize=12, weight='bold')
        ax.set_title(f'Simulatability by Option [{option_label}] | QID {qid}', fontsize=14, weight='bold')
        ax.set_xlim(0, 105)
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"simulatability_summary_{option_label.lower()}_qid{qid}.png"
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[Saved] {out_path}")
        
# ============================================
# MAIN ANALYSIS FUNCTION
# ============================================

def analyze_by_type(domain='DOMAIN'):
    """
    Analyze answers breakdown by question type (e.g., task_qa, sim_qa, task_sim)
    """
    # Load data
    fixed_qs = load_templates(domain)
    cf_data = load_counterfactuals(domain)
    
    print_templates_info(fixed_qs)
    
    # Process subfolders
    folder_name = f'outputs/{domain}_llama3-2-90b-instruct_30_counterfactuals_TEMPLATE_BASED'
    subfolders = [f.path for f in os.scandir(folder_name) if f.is_dir()]
    
    for subfolder in subfolders:
        if not os.path.basename(subfolder).startswith("v7"):
            continue
        
        # Load model outputs
        outputs = load_model_outputs(subfolder, domain)
        
        if not all(key in outputs for key in ['taskqa', 'simqa', 'tasksim']):
            continue
        
        # Group data
        grouped = group_data(outputs, cf_data, fixed_qs)
        
        # Compute statistics
        stats = compute_statistics(grouped)
        
        # Print tables
        print_table("OPTION [a] BIAS BREAKDOWN", stats['stats_a'])
        print_table("OPTION [b] BIAS BREAKDOWN", stats['stats_b'])
        print_table("COUNTERFACTUAL-B BREAKDOWN", stats['cf_b_stats'])
        
        # Print confusion matrices
        TP, FP, FN, TN = print_global_confusion_matrix(stats['global_stats'])
        print_cf_confusion_matrices(
            stats['cf_b_stats'],
            stats['orig_to_cf_switch'],
            stats['orig_to_cf_qid']
        )
        
        # Create output directory
        output_dir = "bias_diagrams"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save visualizations
        save_global_confusion_heatmap(TP, FP, FN, TN, output_dir)
        save_option_a_aggregate_confusion(stats['stats_a'], output_dir)
        save_option_a_bias_heatmap(stats['stats_a'], output_dir)
        save_cf_heatmaps(
            stats['cf_b_stats'],
            stats['orig_to_cf_switch'],
            stats['orig_to_cf_qid'],
            output_dir
        )
        
        # Save QID pie charts
        qid_output_dir = "qid_pies"
        os.makedirs(qid_output_dir, exist_ok=True)
        save_qid_pie_charts(stats['qid_stats_A'], 'A', qid_output_dir)
        save_qid_pie_charts(stats['qid_stats_B'], 'B', qid_output_dir)
        
        # Compute and save simulatability pie charts
        print("\n" + "="*60)
        print("GENERATING SIMULATABILITY CHARTS")
        print("="*60)
        
        simulatability_a, simulatability_b = compute_simulatability_by_option(grouped)
        
        # Create output directory for simulatability charts
        sim_output_dir = "qid_pies"
        os.makedirs(sim_output_dir, exist_ok=True)
        
        # Save individual pie charts for each option value
        print("\nGenerating individual simulatability pie charts for Option A...")
        save_simulatability_pie_charts(simulatability_a, 'A', sim_output_dir)
        
        print("\nGenerating individual simulatability pie charts for Option B...")
        save_simulatability_pie_charts(simulatability_b, 'B', sim_output_dir)
        
        # Save summary bar charts
        print("\nGenerating simulatability summary charts...")
        save_simulatability_summary_charts(simulatability_a, 'A', sim_output_dir)
        save_simulatability_summary_charts(simulatability_b, 'B', sim_output_dir)
        
        print(f"\n[Complete] All simulatability charts saved to {sim_output_dir}/")
        
        print("\n" + "="*60)
        print("GENERATING SIMULATABILITY CHARTS")
        print("="*60)
        
        simulatability_a, simulatability_b = compute_simulatability_by_option(grouped)
        
        # Create output directory for simulatability charts
        sim_output_dir = "simulatability_charts"
        os.makedirs(sim_output_dir, exist_ok=True)
        
        # Save individual pie charts for each option value
        print("\nGenerating individual simulatability pie charts for Option A...")
        save_simulatability_pie_charts(simulatability_a, 'A', sim_output_dir)
        
        print("\nGenerating individual simulatability pie charts for Option B...")
        save_simulatability_pie_charts(simulatability_b, 'B', sim_output_dir)
        
        # Save summary bar charts
        print("\nGenerating simulatability summary charts...")
        save_simulatability_summary_charts(simulatability_a, 'A', sim_output_dir)
        save_simulatability_summary_charts(simulatability_b, 'B', sim_output_dir)
        
        print(f"\n[Complete] All simulatability charts saved to {sim_output_dir}/")
        
        # NEW: Compute and save counterfactual-specific simulatability
        print("\n" + "="*60)
        print("GENERATING COUNTERFACTUAL SIMULATABILITY CHARTS")
        print("="*60)
        
        cf_simulatability = compute_cf_simulatability_by_original_b(grouped)
        
        cf_sim_output_dir = "cf_simulatability_charts"
        os.makedirs(cf_sim_output_dir, exist_ok=True)
        
        # print("\nGenerating CF simulatability pie charts...")
        # save_cf_simulatability_pie_charts(cf_simulatability, cf_sim_output_dir)
        
        print("\nGenerating CF simulatability heatmaps...")
        save_cf_simulatability_heatmap(cf_simulatability, cf_sim_output_dir)
        
        print("\nGenerating CF not simulatable grouped bar charts...")
        save_cf_not_simulatable_grouped_bars(cf_simulatability, cf_sim_output_dir)
        
        print(f"\n[Complete] All CF simulatability charts saved to {cf_sim_output_dir}/")


# Set DOMAIN as a module-level variable that can be imported
DOMAIN = 'sycophancy'

if __name__ == "__main__":
    analyze_by_type(DOMAIN)