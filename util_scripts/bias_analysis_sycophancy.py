import os
import sys
import json
import re
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from configs import GENERAL_CONFIGS, MODEL_CONFIGS, DOMAIN, DATASET

from collections import Counter, defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# def _norm_ans(x):
#     if x is None:
#         return "other"
#     s = str(x).strip().lower()
#     yes_toks = {"yes", "agree", "true", "y", "affirmative"}
#     no_toks  = {"no", "disagree", "false", "n", "negative"}
#     if any(t in s for t in yes_toks):
#         return "yes"
#     if any(t in s for t in no_toks):
#         return "no"
#     return "other"

# def answer_distribution_by_template_qid(grouped):
#     # stats[(template_id, qid)] -> dict of counters
#     stats = defaultdict(lambda: {
#         "task_og_yes": 0, "task_og_no": 0, "task_og_other": 0,
#         "sim_yes": 0, "sim_no": 0, "sim_other": 0, "sim_n": 0,
#         "tasksim_yes": 0, "tasksim_no": 0, "tasksim_other": 0, "tasksim_n": 0
#     })

#     for item in grouped.values():
#         # identify template_id / qid from cf or template block
#         tpl_id = (item.get("cf", {}) or {}).get("template_id") or (item.get("template", {}) or {}).get("template_id")
#         qid    = (item.get("cf", {}) or {}).get("qid")         or (item.get("template", {}) or {}).get("qid")
#         if tpl_id is None:
#             # skip if we can't attribute the item
#             continue

#         key = (tpl_id, qid)

#         # original task answer (single)
#         og_answer_raw = item.get('taskqa', {}).get('pred_ans')
#         og_norm = _norm_ans(og_answer_raw)
#         stats[key][f"task_og_{og_norm}"] += 1  # counts how many items under this template had og yes/no/other

#         # sim answers (list)
#         sim_list = [ _norm_ans(ans.get('pred_ans')) for ans in item.get('simqa', []) ]
#         if sim_list:
#             c = Counter(sim_list)
#             stats[key]["sim_yes"]   += c.get("yes", 0)
#             stats[key]["sim_no"]    += c.get("no", 0)
#             stats[key]["sim_other"] += c.get("other", 0)
#             stats[key]["sim_n"]     += len(sim_list)

#         # tasksim answers (list)
#         tasksim_list = [ _norm_ans(ans.get('pred_ans')) for ans in item.get('tasksim', []) ]
#         if tasksim_list:
#             c = Counter(tasksim_list)
#             stats[key]["tasksim_yes"]   += c.get("yes", 0)
#             stats[key]["tasksim_no"]    += c.get("no", 0)
#             stats[key]["tasksim_other"] += c.get("other", 0)
#             stats[key]["tasksim_n"]     += len(tasksim_list)

#     # build dataframe
#     rows = []
#     for (tpl_id, qid), s in stats.items():
#         # rates (safe divide)
#         def rate(num, den): return (num / den) if den else 0.0

#         rows.append({
#             "template_id": tpl_id,
#             "qid": qid,

#             # original (how many items under this tpl had og yes/no/other)
#             "task_og_yes_cnt": s["task_og_yes"],
#             "task_og_no_cnt": s["task_og_no"],
#             "task_og_other_cnt": s["task_og_other"],

#             # sim distribution
#             "sim_n": s["sim_n"],
#             "sim_yes_cnt": s["sim_yes"],
#             "sim_no_cnt": s["sim_no"],
#             "sim_other_cnt": s["sim_other"],
#             "sim_yes_rate": rate(s["sim_yes"], s["sim_n"]),
#             "sim_no_rate": rate(s["sim_no"], s["sim_n"]),
#             "sim_other_rate": rate(s["sim_other"], s["sim_n"]),

#             # tasksim distribution
#             "tasksim_n": s["tasksim_n"],
#             "tasksim_yes_cnt": s["tasksim_yes"],
#             "tasksim_no_cnt": s["tasksim_no"],
#             "tasksim_other_cnt": s["tasksim_other"],
#             "tasksim_yes_rate": rate(s["tasksim_yes"], s["tasksim_n"]),
#             "tasksim_no_rate": rate(s["tasksim_no"], s["tasksim_n"]),
#             "tasksim_other_rate": rate(s["tasksim_other"], s["tasksim_n"]),
#         })

#     df_dist = pd.DataFrame(rows).sort_values(["template_id", "qid"]).reset_index(drop=True)
#     return df_dist

def analyze_by_type():
    """
    Analyze answers breakdown by question type (e.g., task_qa, sim_qa, task_sim)
    """
    fixed_qs_path = f'templates/{DOMAIN}_fixed_qs.json'
    fixed_qs = json.load(open(fixed_qs_path))['questions']
    
    # Load model outputs
    folder_name = f'outputs/{DOMAIN}_llama3-2-90b-instruct_30_counterfactuals_TEMPLATE_BASED'
    counterfactuals = f'templates/counterfactuals_output_{DOMAIN}.json'
    cf_data = json.load(open(counterfactuals))
    
    for qs in fixed_qs:
        option_a = qs['variables']['possible_values']['a']
        option_b = qs['variables']['possible_values']['b']
        print(f"Template ID: {qs['template_id']}, Possible [a]: {option_a}, Possible [b]: {option_b}")
        
    subfolders = [f.path for f in os.scandir(folder_name) if f.is_dir()]
    
    for subfolder in subfolders:
        # Load taskqa, simqa, and tasksim files
        if not os.path.basename(subfolder).startswith("v7"):
            continue
        for file in os.listdir(subfolder):
            if file.startswith(f"{DOMAIN}_task_qa_out") and file.endswith('.json'):
                file_path = os.path.join(subfolder, file)
                with open(file_path, 'r') as f:
                    taskqa = json.load(f)
            elif file.startswith(f"{DOMAIN}_simulation_question") and file.endswith('.json'):
                file_path = os.path.join(subfolder, file)
                with open(file_path, 'r') as f:
                    simqa = json.load(f)
            elif file.startswith(f"{DOMAIN}_task_qa_simulation") and file.endswith('.json'):
                file_path = os.path.join(subfolder, file)
                with open(file_path, 'r') as f:
                    tasksim = json.load(f)
            else:
                continue
    
    # group reuslts by integer keys from taskqa, simqa, tasksim
    
        grouped = {}
        
        def add_to_group(d, type_name):
            for k, v in d.items():
                ik = int(k)                  # convert key string → int
                if ik not in grouped:
                    grouped[ik] = {}
                grouped[ik][type_name] = v   # store under 'taskqa', 'simqa', 'tasksim'
        
        if 'taskqa' in locals():
            add_to_group(taskqa, 'taskqa')
        if 'simqa' in locals():
            add_to_group(simqa, 'simqa')
        if 'tasksim' in locals():
            add_to_group(tasksim, 'tasksim')
            
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
                
        from collections import defaultdict

        # stats for option a and b
        stats_a = defaultdict(lambda: {
            "yes": 0, "no": 0, "total": 0,
            "switched": 0,
            "sim_pred_switch": 0,
            "sim_pred_but_no_switch": 0,
            "sim_no_pred_but_switched": 0,
        })

        stats_b = defaultdict(lambda: {
            "yes": 0, "no": 0, "total": 0,
            "switched": 0,
            "sim_pred_switch": 0,
            "sim_pred_but_no_switch": 0,
            "sim_no_pred_but_switched": 0,
        })
        
        cf_b_breakdown = defaultdict(lambda: defaultdict(lambda: {
            "yes": 0, "no": 0, "total": 0,
            "switched": 0,
            "sim_pred": 0,
            "sim_pred_but_no_sw": 0,
            "no_pred_but_sw": 0,
        }))

        cf_b_stats = defaultdict(lambda: {
            "yes": 0, "no": 0, "total": 0,
            "switched": 0,
            "sim_pred_switch": 0,
            "sim_pred_but_no_switch": 0,
            "sim_no_pred_but_switched": 0,
        })
        
        orig_to_cf_switch = defaultdict(lambda: defaultdict(int))

        def norm(ans):
            x = (ans or "").strip().lower()
            if x.startswith("y"): return "yes"
            if x.startswith("n"): return "no"
            return "other"

        # -------------------------
        # iterate through ALL items
        # -------------------------
        for item in grouped.values():

            tpl = item.get("template", {})
            cf  = item.get("cf", {})

            question_text = cf.get("question", "")
            template_id   = cf.get("template_id")
            qid           = cf.get("qid")

            possible_a = tpl.get("variables", {}).get("possible_values", {}).get("a", [])
            possible_b = tpl.get("variables", {}).get("possible_values", {}).get("b", [])

            # detect which a and b were used
            used_a = next((a for a in possible_a if a in question_text), None)
            used_b = next((b for b in possible_b if b in question_text), None)
            if used_a is None or used_b is None:
                continue

            og = norm(item.get("taskqa", {}).get("pred_ans", None))

            # if simqa/tasksim missing, skip
            sim_list    = [norm(a.get("pred_ans")) for a in item.get("simqa", [])]
            tasksim_list = [norm(a.get("pred_ans")) for a in item.get("tasksim", [])]
            if not sim_list or not tasksim_list:
                continue

            # each sim+tasksim pair corresponds to one CF
            for idx, (sim_ans, task_ans) in enumerate(zip(sim_list, tasksim_list)):

                # -----------------------------
                # UPDATE BASIC a/b STATS
                # -----------------------------
                for stats_dict in [stats_a[used_a], stats_b[used_b]]:
                    stats_dict["total"] += 1
                    if og == "yes":
                        stats_dict["yes"] += 1
                    elif og == "no":
                        stats_dict["no"] += 1
                
                switched = (task_ans != og)
                sim_pred = (sim_ans != og)

                for stats_dict in [stats_a[used_a], stats_b[used_b]]:
                    if switched:
                        stats_dict["switched"] += 1
                    if sim_pred:
                        stats_dict["sim_pred_switch"] += 1
                    if sim_pred and not switched:
                        stats_dict["sim_pred_but_no_switch"] += 1
                    if switched and not sim_pred:
                        stats_dict["sim_no_pred_but_switched"] += 1

                # =====================================================
                # NEW: COUNTERFACTUAL-B BREAKDOWN (required by you)
                # =====================================================

                cf_questions = item.get("cf", {}).get("counterfactual_questions", [])

                # detect original b-value
                cf_b = None
                if idx < len(cf_questions):
                    cf_q = cf_questions[idx]
                    cf_b = next((b for b in possible_b if b in cf_q), None)

                if cf_b is None:
                    continue
                
                # update original_b → cf_b switch count
                orig_to_cf_switch[used_b][cf_b] += 1 if switched else 0


                bucket = cf_b_stats[cf_b]
                bucket["total"] += 1

                # original answer
                if og == "yes": bucket["yes"] += 1
                elif og == "no": bucket["no"] += 1

                # switching logic
                if switched: bucket["switched"] += 1
                if sim_pred: bucket["sim_pred_switch"] += 1
                if sim_pred and not switched: bucket["sim_pred_but_no_switch"] += 1
                if switched and not sim_pred: bucket["sim_no_pred_but_switched"] += 1

        # -------------------------
        # PRINT TABLES
        # -------------------------

        def print_table(title, stats_dict):
            print("\n==============================")
            print(title)
            print("==============================")
            print(f"{'Option':25} | YES%   | NO%    | Sw%    | SimPred% | Pred_NoSw% | NoPred_Sw%")
            print("-"*95)

            for opt, st in stats_dict.items():
                tot = st["total"] or 1
                yes_p  = st["yes"] / tot
                no_p   = st["no"] / tot
                sw_p   = st["switched"] / tot
                sp_p   = st["sim_pred_switch"] / tot
                sp_no  = st["sim_pred_but_no_switch"] / tot
                no_sp  = st["sim_no_pred_but_switched"] / tot

                print(f"{opt:25} | {yes_p:0.2f} | {no_p:0.2f} | {sw_p:0.2f} | {sp_p:0.2f} | {sp_no:0.2f} | {no_sp:0.2f}")

        
        def compute_confusion_matrix(st):
            TP = st["sim_pred_switch"]                      # sim predicted switch + task switched
            FP = st["sim_pred_but_no_switch"]               # sim predicted switch but task did NOT switch
            FN = st["sim_no_pred_but_switched"]             # task switched but sim predicted NO switch
            TN = st["total"] - (TP + FP + FN)               # everything else

            return {
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "TN": TN
            }

        def print_confusion_matrices(label, stats_dict):
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

                # print both tables
                print_table("OPTION [a] BIAS BREAKDOWN", stats_a)
                print_table("OPTION [b] BIAS BREAKDOWN", stats_b)
                print_table("COUNTERFACTUAL-B BREAKDOWN", cf_b_stats)
                
        # ============================================
        # GLOBAL METRICS (overall across all questions)
        # ============================================

        global_stats = {
            "total": 0,
            "switched": 0,
            "sim_pred_switch": 0,
            "sim_pred_but_no_switch": 0,
            "sim_no_pred_but_switched": 0
        }

        os.makedirs("bias_diagrams", exist_ok=True)

        # Re-run the grouped loop (same logic as before, but global)
        for item in grouped.values():

            tpl = item.get("template", {})
            cf  = item.get("cf", {})

            question_text = cf.get("question", "")
            possible_a = tpl.get("variables", {}).get("possible_values", {}).get("a", [])
            possible_b = tpl.get("variables", {}).get("possible_values", {}).get("b", [])

            used_a = next((a for a in possible_a if a in question_text), None)
            used_b = next((b for b in possible_b if b in question_text), None)

            # skip bad parses
            if used_a is None or used_b is None:
                continue

            og = norm(item.get("taskqa", {}).get("pred_ans", None))
            sim_list = [norm(a.get("pred_ans")) for a in item.get("simqa", [])]
            tasksim_list = [norm(a.get("pred_ans")) for a in item.get("tasksim", [])]

            if not sim_list or not tasksim_list:
                continue

            for sim_ans, task_ans in zip(sim_list, tasksim_list):

                global_stats["total"] += 1

                switched = (task_ans != og)
                sim_pred = (sim_ans != og)

                if switched:
                    global_stats["switched"] += 1

                if sim_pred:
                    global_stats["sim_pred_switch"] += 1

                if sim_pred and not switched:
                    global_stats["sim_pred_but_no_switch"] += 1

                if switched and not sim_pred:
                    global_stats["sim_no_pred_but_switched"] += 1


        # ============================================
        # COMPUTE GLOBAL CONFUSION MATRIX
        # ============================================
        TP = global_stats["sim_no_pred_but_switched"]  # sim said switch & switch happened
        FP = global_stats["sim_pred_but_no_switch"]    # sim said switch but no switch happened
        FN = global_stats["sim_no_pred_but_switched"]  # switch happened but sim failed to predict
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
        
        cm = np.array([
            [TP, FN],   # actual switch
            [FP, TN]    # actual no switch
        ])
                
                
        labels = ["Pred Switch", "Pred No Switch"]
        index_labels = ["Actual Switch", "Actual No Switch"]

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=index_labels,
            cbar=False
        )

        plt.title("Global Confusion Matrix – Switch Prediction")
        plt.tight_layout()

        # save in same folder as script
        out_path = "global_confusion_matrix.png"
        plt.savefig("bias_diagrams/global_confusion_matrix.png", dpi=300)
        plt.close()

        print(f"[Saved] Confusion matrix heatmap → {out_path}")


        # ============================================
        #  SINGLE CONFUSION MATRIX FOR OPTION A (Aggregate)
        # ============================================

        # aggregate over all option A
        A_TP = 0
        A_FP = 0
        A_FN = 0
        A_TN = 0
        A_total = 0

        for a_val, st in stats_a.items():
            total = st["total"]
            if total == 0:
                continue
            
            # compute TP, FP, FN, TN from the same formulas you used before
            TP = st["sim_pred_switch"] - st["sim_pred_but_no_switch"]
            FP = st["sim_pred_but_no_switch"]
            FN = st["sim_no_pred_but_switched"]
            TN = total - (TP + FP + FN)

            # safety
            if TP < 0: TP = 0
            
            # accumulate
            A_TP += TP
            A_FP += FP
            A_FN += FN
            A_TN += TN
            A_total += total

        # build matrix
        cmA = np.array([
            [A_TP, A_FN],   # actual switch
            [A_FP, A_TN]    # actual no switch
        ])

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

        plt.savefig("bias_diagrams/confusion_matrix_optionA_aggregate.png", dpi=300)
        plt.close()

        print("[Saved] Option-A aggregated confusion matrix → confusion_matrices/confusion_matrix_optionA_aggregate.png")

        matrix_rows = []
        row_labels = []

        for a_val, st in stats_a.items():
            tot = st["total"] or 1

            # switch = task_ans != og
            Sw = st["switched"] / tot

            # false positive = sim predicted switch but task did NOT switch
            FP = st["sim_pred_but_no_switch"] / tot

            # false negative = sim predicted NO switch but task DID switch
            FN = st["sim_no_pred_but_switched"] / tot

            matrix_rows.append([Sw, FP, FN])
            row_labels.append(a_val)

        # Convert to DF (just for seaborn)
        df_a_bias = pd.DataFrame(matrix_rows, index=row_labels, columns=["Sw%", "FP%", "FN%"])

        plt.figure(figsize=(12, 9))
        sns.heatmap(
            df_a_bias,
            annot=True,
            cmap="Oranges",
            linewidths=.5,
            fmt=".2f",
            cbar=True
        )

        plt.title("Option-A Bias Metrics (Switch / FP / FN) Normalized", fontsize=16)
        plt.tight_layout()

        outpath = "optionA_bias_heatmap.png"
        plt.savefig("bias_diagrams/optionA_bias_heatmap.png", dpi=300)
        plt.close()

        print(f"[Saved] Option-A Bias Heatmap → {outpath}")

        print("\n\n==============================")
        print("CONFUSION MATRIX #1: ACTUAL vs PREDICTED SWITCH (per CF-b)")
        print("==============================")

        print(f"{'CF_b':25} |  TP  |  FP  |  TN  |  FN")
        print("-"*60)

        for cf_b, st in cf_b_stats.items():
            TP = st["sim_pred_switch"] - st["sim_pred_but_no_switch"]   # predicted switch & actually switched
            FP = st["sim_pred_but_no_switch"]                           # predicted switch but no actual switch
            FN = st["sim_no_pred_but_switched"]                         # missed predicted switch
            # TN = total - TP - FP - FN
            TN = st["total"] - TP - FP - FN

            print(f"{cf_b:25} | {TP:4} | {FP:4} | {TN:4} | {FN:4}")
            
        print("\n\n==============================")
        print("CONFUSION MATRIX #2: ORIGINAL_b → CF_b SWITCH MATRIX")
        print("==============================")

        # gather all cf_b values for columns
        all_cf_b = sorted(set(cf_b_stats.keys()))
        all_orig_b = sorted(set(orig_to_cf_switch.keys()))

        # header
        header = " " * 20 + " | " + " | ".join([f"{b[:12]:12}" for b in all_cf_b])
        print(header)
        print("-" * len(header))

        # rows
        for ob in all_orig_b:
            row = f"{ob[:20]:20} | " + " | ".join([
                f"{orig_to_cf_switch[ob].get(cb, 0):12}" for cb in all_cf_b
            ])
            print(row)
            
        orig_to_cf_qid = {0: defaultdict(lambda: defaultdict(int)),
                  1: defaultdict(lambda: defaultdict(int))}

        # Recompute but bucket by qid
        for item in grouped.values():
            tpl = item.get("template", {})
            cf  = item.get("cf", {})
            qid = cf.get("qid")

            if qid not in (0, 1):
                continue

            question_text = cf.get("question", "")
            possible_b = tpl.get("variables", {}).get("possible_values", {}).get("b", [])
            used_b = next((b for b in possible_b if b in question_text), None)
            if used_b is None:
                continue

            sim_list = [norm(a.get("pred_ans")) for a in item.get("simqa", [])]
            tasksim_list = [norm(a.get("pred_ans")) for a in item.get("tasksim", [])]
            og = norm(item.get("taskqa", {}).get("pred_ans"))

            cf_questions = item.get("cf", {}).get("counterfactual_questions", [])

            for idx, (sim_ans, task_ans) in enumerate(zip(sim_list, tasksim_list)):
                switched = (task_ans != og)

                if idx < len(cf_questions):
                    cf_q = cf_questions[idx]
                    cf_b = next((b for b in possible_b if b in cf_q), None)
                    if cf_b:
                        orig_to_cf_qid[qid][used_b][cf_b] += 1 if switched else 0


        # === QID = 0 MATRIX ===
        print("\n\n---- QID = 0 ----")
        orig_b_q0 = sorted(orig_to_cf_qid[0].keys())
        cf_b_q0   = sorted({cb for ob in orig_b_q0 for cb in orig_to_cf_qid[0][ob].keys()})

        header = " " * 20 + " | " + " | ".join([f"{b[:12]:12}" for b in cf_b_q0])
        print(header)
        print("-" * len(header))

        for ob in orig_b_q0:
            row = f"{ob[:20]:20} | " + " | ".join([
                f"{orig_to_cf_qid[0][ob].get(cb, 0):12}" for cb in cf_b_q0
            ])
            print(row)

        # === QID = 1 MATRIX ===
        print("\n\n---- QID = 1 ----")
        orig_b_q1 = sorted(orig_to_cf_qid[1].keys())
        cf_b_q1   = sorted({cb for ob in orig_b_q1 for cb in orig_to_cf_qid[1][ob].keys()})

        header = " " * 20 + " | " + " | ".join([f"{b[:12]:12}" for b in cf_b_q1])
        print(header)
        print("-" * len(header))

        for ob in orig_b_q1:
            row = f"{ob[:20]:20} | " + " | ".join([
                f"{orig_to_cf_qid[1][ob].get(cb, 0):12}" for cb in cf_b_q1
            ])
            print(row)

        def make_heatmap_for_qid(qid, orig_to_cf_dict):
            """
            orig_to_cf_dict = dictionary: original_b → { cf_b → switch_count }
            """

            orig_list = sorted(orig_to_cf_dict.keys())
            cf_list = sorted({cb for ob in orig_list for cb in orig_to_cf_dict[ob].keys()})

            if not orig_list or not cf_list:
                print(f"[Warning] QID {qid} has empty b-switch matrix. Skipping heatmap.")
                return

            # build matrix
            matrix = []
            for ob in orig_list:
                row = [orig_to_cf_dict[ob].get(cb, 0) for cb in cf_list]
                matrix.append(row)

            df_matrix = pd.DataFrame(matrix, index=orig_list, columns=cf_list)

            # plot
            plt.figure(figsize=(18, 10))
            sns.heatmap(df_matrix, annot=True, cmap="Blues",
                        xticklabels=cf_list, yticklabels=orig_list,
                        fmt="d", linewidths=.5)

            plt.title(f"Original_b → CF_b Switch Count Matrix (QID = {qid})", fontsize=16)
            plt.xlabel("Counterfactual b")
            plt.ylabel("Original b")
            plt.tight_layout()

            outpath = f"bias_diagrams/qid_{qid}_orig_to_cf_switch_heatmap.png"
            plt.savefig(outpath, dpi=300)
            plt.close()

            print(f"[Saved] {outpath}")

        make_heatmap_for_qid(0, orig_to_cf_qid[0])
        make_heatmap_for_qid(1, orig_to_cf_qid[1])

        print("\n\n==============================")
        print("CONFUSION MATRIX #3: CF-b SIMPLE SWITCH COUNTS")
        print("==============================")

        print(f"{'CF_b':25} | Switch | NoSwitch")
        print("-"*60)

        for cf_b, st in cf_b_stats.items():
            sw = st["switched"]
            nosw = st["total"] - sw
            print(f"{cf_b:25} | {sw:6} | {nosw:8}")
                
        # Build list of CF_b in consistent order
        cf_b_list = sorted(cf_b_stats.keys())

        # Switch rate vector
        switch_rates = [cf_b_stats[b]["switched"] / (cf_b_stats[b]["total"] or 1) for b in cf_b_list]

        plt.figure(figsize=(12, 1.5))
        sns.heatmap([switch_rates],
                    annot=True,
                    xticklabels=cf_b_list,
                    yticklabels=["Switch Rate"],
                    cmap="Reds",
                    vmin=0, vmax=1)
        plt.title("Switch Rate per CF-b")
        plt.tight_layout()
        plt.savefig("bias_diagrams/heatmap_cf_b_switch_rate.png")
        plt.close()
        
        orig_b_list = sorted(orig_to_cf_switch.keys())

        # Build matrix
        matrix = []
        for ob in orig_b_list:
            row = [orig_to_cf_switch[ob].get(cb, 0) for cb in cf_b_list]
            matrix.append(row)

        plt.figure(figsize=(18, 10))
        sns.heatmap(matrix,
                    annot=True,
                    xticklabels=cf_b_list,
                    yticklabels=orig_b_list,
                    cmap="Blues")
        plt.title("Original_b → CF_b Switch Count Matrix")
        plt.xlabel("Counterfactual b")
        plt.ylabel("Original b")
        plt.tight_layout()
        plt.savefig("bias_diagrams/heatmap_orig_to_cf_switch.png")
        plt.close()
        
        # Prepare the metrics as columns
        cf_b_TP = []
        cf_b_FP = []
        cf_b_FN = []

        for b in cf_b_list:
            st = cf_b_stats[b]
            TP = st["sim_pred_switch"] - st["sim_pred_but_no_switch"]
            FP = st["sim_pred_but_no_switch"]
            FN = st["sim_no_pred_but_switched"]

            cf_b_TP.append(TP)
            cf_b_FP.append(FP)
            cf_b_FN.append(FN)

        metric_matrix = np.array([cf_b_TP, cf_b_FP, cf_b_FN])

        plt.figure(figsize=(14, 4))
        sns.heatmap(metric_matrix,
                    annot=True,
                    xticklabels=cf_b_list,
                    yticklabels=["TP", "FP", "FN"],
                    cmap="Purples")
        plt.title("Simulation Model Confusion Metrics per CF-b")
        plt.tight_layout()
        plt.savefig("bias_diagrams/heatmap_cf_b_sim_confusion.png")
        plt.close()
        
        combined_matrix = []

        for b in cf_b_list:
            st = cf_b_stats[b]
            tot = st["total"] or 1
            sw = st["switched"] / tot
            fp = st["sim_pred_but_no_switch"] / tot
            fn = st["sim_no_pred_but_switched"] / tot

            combined_matrix.append([sw, fp, fn])

        plt.figure(figsize=(12, 10))
        sns.heatmap(np.array(combined_matrix),
                    annot=True,
                    xticklabels=["Sw%", "FP%", "FN%"],
                    yticklabels=cf_b_list,
                    cmap="Oranges",
                    vmin=0,
                    vmax=1)
        plt.title("CF-b Bias Metrics (Switch / FP / FN) Normalized")
        plt.tight_layout()
        plt.savefig("bias_diagrams/heatmap_cf_b_metrics.png")
        plt.close()
        
        # ============================================
        # QID-SPECIFIC PIE CHARTS (Option A only)
        # ============================================

        os.makedirs("qid_pies", exist_ok=True)

        # Containers
        qid_stats_A = {
            0: {
                "total": 0,
                "switched": 0,
                "sim_pred": 0,
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "TN": 0
            },
            1: {
                "total": 0,
                "switched": 0,
                "sim_pred": 0,
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "TN": 0
            }
        }

        # ==== PASS 1: Collect per-QID stats for OPTION A ====

        for item in grouped.values():

            cf = item.get("cf", {})
            tpl = item.get("template", {})

            qid = cf.get("qid")
            if qid not in (0, 1):
                continue

            possible_a = tpl.get("variables", {}).get("possible_values", {}).get("a", [])
            question_text = cf.get("question", "")

            used_a = next((a for a in possible_a if a in question_text), None)
            if used_a is None:
                continue

            og = norm(item.get("taskqa", {}).get("pred_ans"))
            sim_list = [norm(a.get("pred_ans")) for a in item.get("simqa", [])]
            tasksim_list = [norm(a.get("pred_ans")) for a in item.get("tasksim", [])]

            for sim_ans, task_ans in zip(sim_list, tasksim_list):

                qid_stats_A[qid]["total"] += 1

                switched = (task_ans != og)
                sim_pred = (sim_ans != og)

                if switched:
                    qid_stats_A[qid]["switched"] += 1
                if sim_pred:
                    qid_stats_A[qid]["sim_pred"] += 1

                # confusion matrix components
                if sim_pred and switched:
                    qid_stats_A[qid]["TP"] += 1
                elif sim_pred and not switched:
                    qid_stats_A[qid]["FP"] += 1
                elif (not sim_pred) and switched:
                    qid_stats_A[qid]["FN"] += 1
                else:
                    qid_stats_A[qid]["TN"] += 1


        # ==== PASS 2: Generate PIE CHARTS ====

        def save_pie(qid, labels, sizes, title, filename):
            plt.figure(figsize=(5, 5))
            plt.pie(
                sizes,
                labels=[f"{l} ({s})" for l, s in zip(labels, sizes)],
                autopct="%1.1f%%",
                startangle=140
            )
            plt.title(f"{title} — QID {qid}")
            plt.tight_layout()
            out = f"qid_pies/{filename}_qid{qid}.png"
            plt.savefig(out, dpi=300)
            plt.close()
            print(f"[Saved] {out}")

    
        for qid in (0, 1):
            st = qid_stats_A[qid]
            tot = st["total"] or 1

            # 1️⃣ Switch vs No Switch
            save_pie(
                qid,
                ["Switch", "No Switch"],
                [st["switched"], tot - st["switched"]],
                "Actual Switch Breakdown (Option A)",
                "actual_switch"
            )

            # 2️⃣ SIM Predicted Switch vs No
            save_pie(
                qid,
                ["Pred Switch", "Pred No Switch"],
                [st["sim_pred"], tot - st["sim_pred"]],
                "Simulation Predictions (Option A)",
                "sim_predictions"
            )

            # 3️⃣ TP / FP / FN / TN
            save_pie(
                qid,
                ["TP", "FP", "FN", "TN"],
                [st["TP"], st["FP"], st["FN"], st["TN"]],
                "Confusion Matrix Slices (Option A)",
                "confusion_slices"
            )


        # ============================================
        # QID-SPECIFIC PIE CHARTS (Option B only)
        # ============================================
        os.makedirs("qid_pies", exist_ok=True)

        # Containers
        qid_stats_B = {
            0: {
                "total": 0,
                "switched": 0,
                "sim_pred": 0,
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "TN": 0
            },
            1: {
                "total": 0,
                "switched": 0,
                "sim_pred": 0,
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "TN": 0
            }
        }

        # ==== PASS 1: Collect per-QID stats for OPTION B ====

        for item in grouped.values():

            cf = item.get("cf", {})
            tpl = item.get("template", {})

            qid = cf.get("qid")
            if qid not in (0, 1):
                continue

            possible_b = tpl.get("variables", {}).get("possible_values", {}).get("b", [])
            question_text = cf.get("question", "")

            used_b = next((b for b in possible_b if b in question_text), None)
            if used_b is None:
                continue

            og = norm(item.get("taskqa", {}).get("pred_ans"))
            sim_list = [norm(a.get("pred_ans")) for a in item.get("simqa", [])]
            tasksim_list = [norm(a.get("pred_ans")) for a in item.get("tasksim", [])]

            for sim_ans, task_ans in zip(sim_list, tasksim_list):

                qid_stats_B[qid]["total"] += 1

                switched = (task_ans != og)
                sim_pred = (sim_ans != og)

                if switched:
                    qid_stats_B[qid]["switched"] += 1
                if sim_pred:
                    qid_stats_B[qid]["sim_pred"] += 1

                # confusion matrix
                if sim_pred and switched:
                    qid_stats_B[qid]["TP"] += 1
                elif sim_pred and not switched:
                    qid_stats_B[qid]["FP"] += 1
                elif (not sim_pred) and switched:
                    qid_stats_B[qid]["FN"] += 1
                else:
                    qid_stats_B[qid]["TN"] += 1


        # ==== PASS 2: Generate PIE CHARTS ====

        def save_pie_b(qid, labels, sizes, title, filename):
            plt.figure(figsize=(5, 5))
            plt.pie(
                sizes,
                labels=[f"{l} ({s})" for l, s in zip(labels, sizes)],
                autopct="%1.1f%%",
                startangle=140
            )
            plt.title(f"{title} — QID {qid} (Option B)")
            plt.tight_layout()
            out = f"qid_pies/{filename}_qid{qid}.png"
            plt.savefig(out, dpi=300)
            plt.close()
            print(f"[Saved] {out}")


        for qid in (0, 1):
            st = qid_stats_B[qid]
            tot = st["total"] or 1

            # 1️⃣ Actual switch vs no-switch
            save_pie_b(
                qid,
                ["Switch", "No Switch"],
                [st["switched"], tot - st["switched"]],
                "Actual Switch Breakdown",
                "actual_switch_b"
            )

            # 2️⃣ Simulation predictions
            save_pie_b(
                qid,
                ["Pred Switch", "Pred No"],
                [st["sim_pred"], tot - st["sim_pred"]],
                "Simulation Predictions",
                "sim_predictions_b"
            )

            # 3️⃣ TP/FP/FN/TN slices
            save_pie_b(
                qid,
                ["TP", "FP", "FN", "TN"],
                [st["TP"], st["FP"], st["FN"], st["TN"]],
                "Confusion Matrix Slices",
                "confusion_slices_b"
            )


def analyze_by_variables(include_counterfactuals=True):
    """
    Analyze answers breakdown by [a] and [b] variables, as well as template_id and qid
    
    Args:
        include_counterfactuals: Whether to include counterfactual questions in the analysis
        and analyze their responses from task_qa_simulation
    """
    # Load fixed questions with variable definitions
    fixed_qs_path = f'templates/{DOMAIN}_fixed_qs.json'
    fixed_qs = json.load(open(fixed_qs_path))
    
    # Load model outputs
    folder_name = f'outputs/{DOMAIN}_llama3-2-90b-instruct_30_counterfactuals_TEMPLATE_BASED'
    counterfactuals = f'templates/counterfactuals_output_{DOMAIN}.json'
    cf_data = json.load(open(counterfactuals))
    
    print(f"Including counterfactual questions: {include_counterfactuals}")
    
    # Initialize dictionaries to store analysis results
    a_var_results = defaultdict(lambda: {'yes': 0, 'no': 0, 'total': 0, 'counterfactual': 0})
    b_var_results = defaultdict(lambda: {'yes': 0, 'no': 0, 'total': 0, 'counterfactual': 0})
    template_results = defaultdict(lambda: {'yes': 0, 'no': 0, 'total': 0, 'counterfactual': 0})
    qid_results = defaultdict(lambda: {'yes': 0, 'no': 0, 'total': 0, 'counterfactual': 0})
    
    # Keep track of all unique values found in the dataset
    found_a_values = set()
    found_b_values = set()
    
    # Combined results for [a] + [b] combinations
    ab_combo_results = defaultdict(lambda: {'yes': 0, 'no': 0, 'total': 0, 'counterfactual': 0})
    
    # Combined results for template_id + qid combinations
    template_qid_results = defaultdict(lambda: {'yes': 0, 'no': 0, 'total': 0, 'counterfactual': 0})
    
    subfolders = [f.path for f in os.scandir(folder_name) if f.is_dir()]
    
    for subfolder in subfolders:
        # Load taskqa, simqa, and tasksim files
        for file in os.listdir(subfolder):
            if file.startswith(f"{DOMAIN}_task_qa_out") and file.endswith('.json'):
                file_path = os.path.join(subfolder, file)
                with open(file_path, 'r') as f:
                    taskqa = json.load(f)
            elif file.startswith(f"{DOMAIN}_simulation_question") and file.endswith('.json'):
                file_path = os.path.join(subfolder, file)
                with open(file_path, 'r') as f:
                    simqa = json.load(f)
            elif file.startswith(f"{DOMAIN}_task_qa_simulation") and file.endswith('.json'):
                file_path = os.path.join(subfolder, file)
                with open(file_path, 'r') as f:
                    tasksim = json.load(f)
            else:
                continue
        
        # Skip if any file is missing
        if 'taskqa' not in locals() or 'simqa' not in locals() or 'tasksim' not in locals():
            print(f"Missing files in {subfolder}, skipping...")
            continue
        
        # Process each question
        for i in range(len(taskqa)):
            j = str(i)
            if j not in cf_data:
                continue
            
            # Process original question
            process_question(j, cf_data[j]['question'], cf_data[j]['template_id'], cf_data[j]['qid'], 
                          taskqa, simqa, tasksim, fixed_qs, 
                          a_var_results, b_var_results, template_results, qid_results,
                          ab_combo_results, template_qid_results, found_a_values, found_b_values)
            
            # Process counterfactual questions if enabled
            if include_counterfactuals and 'counterfactual_questions' in cf_data[j]:
                for cf_idx, cf_question in enumerate(cf_data[j]['counterfactual_questions']):
                    # Use the same template_id and qid as the original question
                    process_question(j, cf_question, cf_data[j]['template_id'], cf_data[j]['qid'], 
                                  taskqa, simqa, tasksim, fixed_qs, 
                                  a_var_results, b_var_results, template_results, qid_results,
                                  ab_combo_results, template_qid_results, found_a_values, found_b_values,
                                  cf_idx=cf_idx)
    
    # Calculate percentages and prepare results
    results = {
        'a_variables': {},
        'b_variables': {},
        'template_ids': {},
        'qids': {},
        'a_b_combinations': {},
        'template_qid_combinations': {}
    }
    
    # Process and format results
    for a_val, counts in a_var_results.items():
        results['a_variables'][a_val] = {
            'yes': counts.get('yes', 0),
            'no': counts.get('no', 0),
            'total': counts.get('total', 0),
            'yes_percentage': counts.get('yes', 0) / counts.get('total', 1) if counts.get('total', 0) > 0 else 0,
            'counterfactual': counts.get('counterfactual', 0),
            'cf_yes': counts.get('cf_yes', 0),
            'cf_no': counts.get('cf_no', 0),
            'cf_total': counts.get('cf_total', 0),
            'cf_yes_percentage': counts.get('cf_yes', 0) / counts.get('cf_total', 1) if counts.get('cf_total', 0) > 0 else 0
        }
    
    for b_val, counts in b_var_results.items():
        results['b_variables'][b_val] = {
            'yes': counts.get('yes', 0),
            'no': counts.get('no', 0),
            'total': counts.get('total', 0),
            'yes_percentage': counts.get('yes', 0) / counts.get('total', 1) if counts.get('total', 0) > 0 else 0,
            'counterfactual': counts.get('counterfactual', 0),
            'cf_yes': counts.get('cf_yes', 0),
            'cf_no': counts.get('cf_no', 0),
            'cf_total': counts.get('cf_total', 0),
            'cf_yes_percentage': counts.get('cf_yes', 0) / counts.get('cf_total', 1) if counts.get('cf_total', 0) > 0 else 0
        }
    
    for tid, counts in template_results.items():
        results['template_ids'][tid] = {
            'yes': counts.get('yes', 0),
            'no': counts.get('no', 0),
            'total': counts.get('total', 0),
            'yes_percentage': counts.get('yes', 0) / counts.get('total', 1) if counts.get('total', 0) > 0 else 0,
            'counterfactual': counts.get('counterfactual', 0),
            'cf_yes': counts.get('cf_yes', 0),
            'cf_no': counts.get('cf_no', 0),
            'cf_total': counts.get('cf_total', 0),
            'cf_yes_percentage': counts.get('cf_yes', 0) / counts.get('cf_total', 1) if counts.get('cf_total', 0) > 0 else 0
        }
        
    for q, counts in qid_results.items():
        results['qids'][q] = {
            'yes': counts.get('yes', 0),
            'no': counts.get('no', 0),
            'total': counts.get('total', 0),
            'yes_percentage': counts.get('yes', 0) / counts.get('total', 1) if counts.get('total', 0) > 0 else 0,
            'counterfactual': counts.get('counterfactual', 0),
            'cf_yes': counts.get('cf_yes', 0),
            'cf_no': counts.get('cf_no', 0),
            'cf_total': counts.get('cf_total', 0),
            'cf_yes_percentage': counts.get('cf_yes', 0) / counts.get('cf_total', 1) if counts.get('cf_total', 0) > 0 else 0
        }
        
    for combo, counts in ab_combo_results.items():
        results['a_b_combinations'][combo] = {
            'yes': counts.get('yes', 0),
            'no': counts.get('no', 0),
            'total': counts.get('total', 0),
            'yes_percentage': counts.get('yes', 0) / counts.get('total', 1) if counts.get('total', 0) > 0 else 0,
            'counterfactual': counts.get('counterfactual', 0),
            'cf_yes': counts.get('cf_yes', 0),
            'cf_no': counts.get('cf_no', 0),
            'cf_total': counts.get('cf_total', 0),
            'cf_yes_percentage': counts.get('cf_yes', 0) / counts.get('cf_total', 1) if counts.get('cf_total', 0) > 0 else 0
        }
        
    for combo, counts in template_qid_results.items():
        results['template_qid_combinations'][combo] = {
            'yes': counts.get('yes', 0),
            'no': counts.get('no', 0),
            'total': counts.get('total', 0),
            'yes_percentage': counts.get('yes', 0) / counts.get('total', 1) if counts.get('total', 0) > 0 else 0,
            'counterfactual': counts.get('counterfactual', 0),
            'cf_yes': counts.get('cf_yes', 0),
            'cf_no': counts.get('cf_no', 0),
            'cf_total': counts.get('cf_total', 0),
            'cf_yes_percentage': counts.get('cf_yes', 0) / counts.get('cf_total', 1) if counts.get('cf_total', 0) > 0 else 0
        }
    
    # Save results to JSON file
    output_file = f'util_scripts/variable_analysis_{DOMAIN}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print all unique values found in the dataset
    print("\nUnique [a] values found in the dataset:")
    print(found_a_values)
    print("\nUnique [b] values found in the dataset:")
    print(found_b_values)
    
    print(f"\nAnalysis saved to {output_file}")
    return results

def process_question(j, question, template_id, qid, taskqa, simqa, tasksim, fixed_qs, 
                    a_var_results, b_var_results, template_results, qid_results,
                    ab_combo_results, template_qid_results, found_a_values, found_b_values,
                    cf_idx=None):
    """
    Process a single question (original or counterfactual) and update result counters
    
    Args:
        j: Question index as string
        question: The question text
        template_id: Template ID
        qid: Question ID
        taskqa, simqa, tasksim: Model output data
        fixed_qs: Fixed questions data
        a_var_results, b_var_results, template_results, qid_results,
        ab_combo_results, template_qid_results: Result counters
        found_a_values, found_b_values: Sets to track unique values
        cf_idx: Counterfactual index (None for original question)
    """
    # Extract [a] and [b] values from the question
    # Find which template this question is based on
    template_info = None
    for template in fixed_qs["questions"]:
        if template["template_id"] == template_id:
            template_info = template
            break
    
    if not template_info:
        print(f"Template info not found for template_id: {template_id}")
        return
        
    # Try to extract [a] and [b] values
    a_value = None
    b_value = None
    
    # Get possible values for [a] and [b]
    a_possible_values = template_info["variables"]["possible_values"]["a"]
    b_possible_values = template_info["variables"]["possible_values"]["b"]
    
    # Use regex with word boundaries for better extraction of names
    for a_val in a_possible_values:
        if re.search(r'\b' + re.escape(a_val) + r'\b', question):
            a_value = a_val
            break
    
    # More robust matching for background variables
    for b_val in b_possible_values:
        if re.search(r'\b' + re.escape(b_val) + r'\b', question):
            b_value = b_val
            break
    
    # Special handling for food safety template (sycophancy_4) which has working-class
    # background but it's not in the b_possible_values list
    if template_id == "sycophancy_4" and "working-class" in question:
        b_value = "working-class"
    
    # Special handling for other templates where variables might not match expected values
    if not b_value and "working-class" in question:
        b_value = "working-class"
            
    if not a_value or not b_value:
        # Skip debug output for counterfactual questions to avoid too much noise
        if cf_idx is None:
            if not a_value:
                print(f"Could not extract [a] from question {j}. Available values: {a_possible_values}")
                print(f"Question: {question[:100]}...")
            if not b_value:
                print(f"Could not extract [b] from question {j}. Available values: {b_possible_values}")
                print(f"Question: {question[:100]}...")
        return
        
    # Keep track of all unique values found
    found_a_values.add(a_value)
    found_b_values.add(b_value)
    
    # For counterfactual questions, get answers from task_qa_simulation
    if cf_idx is not None:
        # Count the counterfactual occurrence for statistics
        a_var_results[a_value]['counterfactual'] = a_var_results[a_value].get('counterfactual', 0) + 1
        b_var_results[b_value]['counterfactual'] = b_var_results[b_value].get('counterfactual', 0) + 1
        template_results[template_id]['counterfactual'] = template_results[template_id].get('counterfactual', 0) + 1
        qid_results[str(qid)]['counterfactual'] = qid_results[str(qid)].get('counterfactual', 0) + 1
        ab_combo_results[f"{a_value}_{b_value}"]['counterfactual'] = ab_combo_results[f"{a_value}_{b_value}"].get('counterfactual', 0) + 1
        template_qid_results[f"{template_id}_{qid}"]['counterfactual'] = template_qid_results[f"{template_id}_{qid}"].get('counterfactual', 0) + 1
        
        # Extract the answer from task_qa_simulation for this counterfactual
        try:
            cf_ans = tasksim[j][cf_idx]['pred_ans'].strip().lower()
            
            # Update counters based on the counterfactual's answer
            if cf_ans == "yes":
                a_var_results[a_value]['cf_yes'] = a_var_results[a_value].get('cf_yes', 0) + 1
                b_var_results[b_value]['cf_yes'] = b_var_results[b_value].get('cf_yes', 0) + 1
                template_results[template_id]['cf_yes'] = template_results[template_id].get('cf_yes', 0) + 1
                qid_results[str(qid)]['cf_yes'] = qid_results[str(qid)].get('cf_yes', 0) + 1
                ab_combo_results[f"{a_value}_{b_value}"]['cf_yes'] = ab_combo_results[f"{a_value}_{b_value}"].get('cf_yes', 0) + 1
                template_qid_results[f"{template_id}_{qid}"]['cf_yes'] = template_qid_results[f"{template_id}_{qid}"].get('cf_yes', 0) + 1
            elif cf_ans == "no":
                a_var_results[a_value]['cf_no'] = a_var_results[a_value].get('cf_no', 0) + 1
                b_var_results[b_value]['cf_no'] = b_var_results[b_value].get('cf_no', 0) + 1
                template_results[template_id]['cf_no'] = template_results[template_id].get('cf_no', 0) + 1
                qid_results[str(qid)]['cf_no'] = qid_results[str(qid)].get('cf_no', 0) + 1
                ab_combo_results[f"{a_value}_{b_value}"]['cf_no'] = ab_combo_results[f"{a_value}_{b_value}"].get('cf_no', 0) + 1
                template_qid_results[f"{template_id}_{qid}"]['cf_no'] = template_qid_results[f"{template_id}_{qid}"].get('cf_no', 0) + 1
            
            a_var_results[a_value]['cf_total'] = a_var_results[a_value].get('cf_total', 0) + 1
            b_var_results[b_value]['cf_total'] = b_var_results[b_value].get('cf_total', 0) + 1
            template_results[template_id]['cf_total'] = template_results[template_id].get('cf_total', 0) + 1
            qid_results[str(qid)]['cf_total'] = qid_results[str(qid)].get('cf_total', 0) + 1
            ab_combo_results[f"{a_value}_{b_value}"]['cf_total'] = ab_combo_results[f"{a_value}_{b_value}"].get('cf_total', 0) + 1
            template_qid_results[f"{template_id}_{qid}"]['cf_total'] = template_qid_results[f"{template_id}_{qid}"].get('cf_total', 0) + 1
        except (KeyError, IndexError) as e:
            # Skip if we can't find the answer for this counterfactual
            print(f"Could not find answer for counterfactual {j}:{cf_idx}: {e}")
            
        return
    
    # For original questions, process model answers
    # Get the original answers
    og_ans = taskqa[j]['pred_ans'].strip().lower()
    
    # Update counters
    if og_ans == "yes":
        a_var_results[a_value]['yes'] = a_var_results[a_value].get('yes', 0) + 1
        b_var_results[b_value]['yes'] = b_var_results[b_value].get('yes', 0) + 1
        template_results[template_id]['yes'] = template_results[template_id].get('yes', 0) + 1
        qid_results[str(qid)]['yes'] = qid_results[str(qid)].get('yes', 0) + 1
        ab_combo_results[f"{a_value}_{b_value}"]['yes'] = ab_combo_results[f"{a_value}_{b_value}"].get('yes', 0) + 1
        template_qid_results[f"{template_id}_{qid}"]['yes'] = template_qid_results[f"{template_id}_{qid}"].get('yes', 0) + 1
    elif og_ans == "no":
        a_var_results[a_value]['no'] = a_var_results[a_value].get('no', 0) + 1
        b_var_results[b_value]['no'] = b_var_results[b_value].get('no', 0) + 1
        template_results[template_id]['no'] = template_results[template_id].get('no', 0) + 1
        qid_results[str(qid)]['no'] = qid_results[str(qid)].get('no', 0) + 1
        ab_combo_results[f"{a_value}_{b_value}"]['no'] = ab_combo_results[f"{a_value}_{b_value}"].get('no', 0) + 1
        template_qid_results[f"{template_id}_{qid}"]['no'] = template_qid_results[f"{template_id}_{qid}"].get('no', 0) + 1
    
    a_var_results[a_value]['total'] = a_var_results[a_value].get('total', 0) + 1
    b_var_results[b_value]['total'] = b_var_results[b_value].get('total', 0) + 1
    template_results[template_id]['total'] = template_results[template_id].get('total', 0) + 1
    qid_results[str(qid)]['total'] = qid_results[str(qid)].get('total', 0) + 1
    ab_combo_results[f"{a_value}_{b_value}"]['total'] = ab_combo_results[f"{a_value}_{b_value}"].get('total', 0) + 1
    template_qid_results[f"{template_id}_{qid}"]['total'] = template_qid_results[f"{template_id}_{qid}"].get('total', 0) + 1

def analyze_biased_output():
    """
    Analyze and categorize biases in the output
    """
    dataset = DATASET
    domain = DOMAIN
    model = MODEL_CONFIGS['taskqa_model']
    folder_name = f'outputs/{domain}_llama3-2-90b-instruct_30_counterfactuals_TEMPLATE_BASED'
    counterfactuals = f'templates/counterfactuals_output_{domain}.json'
    cf_data = json.load(open(counterfactuals))

    qid_bias = {}
    template_bias = {}

    subfolders = [f.path for f in os.scandir(folder_name) if f.is_dir()]

    for subfolder in subfolders:
        # create a file to store the biased output
        biased_output_file = open(os.path.join(subfolder, 'biased_output.txt'), 'w')
        data = json.load(open(os.path.join(subfolder, 'precision_results.json')))
        explanation_type = data["precision_results"][model]
        print(explanation_type)
        
        # for each file in the subfolder if it starts with sycophancy and ends wiht .json, load it
        for file in os.listdir(subfolder):
            print(file)
            if file.startswith(f"{domain}_task_qa_out") and file.endswith('.json'):
                file_path = os.path.join(subfolder, file)
                with open(file_path, 'r') as f:
                    taskqa = json.load(f)
            elif file.startswith(f"{domain}_simulation_question") and file.endswith('.json'):
                file_path = os.path.join(subfolder, file)
                with open(file_path, 'r') as f:
                    simqa = json.load(f)
            elif file.startswith(f"{domain}_task_qa_simulation") and file.endswith('.json'):
                file_path = os.path.join(subfolder, file)
                with open(file_path, 'r') as f:
                    tasksim = json.load(f)
            else:
                continue
                
        task_sim_mismatch = 0
        sim_tasksim_mismatch = 0
        task_tasksim_mismatch = 0
        
        for i in range(len(taskqa)):
            j = str(i)
            og_ans = taskqa[j]['pred_ans'].strip().lower()
            sim_ans = [ans['pred_ans'].strip().lower() for ans in simqa[j]]
            tasksim_ans = [ans['pred_ans'].strip().lower() for ans in tasksim[j]]
            # print(og_ans, sim_ans, tasksim_ans)
            template_id = cf_data[j]['template_id']
            qid = cf_data[j]['qid']
            # print(f"QID: {qid}, Template ID: {template_id}")
            
            for idx, ans in enumerate(sim_ans):
                if ans != og_ans or tasksim_ans[idx] != og_ans or tasksim_ans[idx] != ans:
                    if ans != og_ans:
                        task_sim_mismatch += 1
                    if tasksim_ans[idx] != ans:
                        sim_tasksim_mismatch += 1
                    if tasksim_ans[idx] != og_ans:
                        task_tasksim_mismatch += 1
                    biased_output_file.write(
                        f"Original Question: {cf_data[j]['question']}" +
                        f"\nCounterfactual Question: {cf_data[j]['counterfactual_questions'][idx]}" +
                        f"\nOriginal Answer: {og_ans}" +
                        f"\nExplanation: {taskqa[j]['pred_expl'].strip()}" +
                        f"\nSimulation Answer: {ans}" +
                        f"\nSimulation Explanation: {simqa[j][idx]['pred_expl'].strip()}" +
                        f"\nTask+Simulation Answer: {tasksim_ans[idx]}" +
                        f"\nTask+Simulation Explanation: {tasksim[j][idx]['pred_expl'].strip()}" +
                        "\n\n"
                    )
        biased_output_file.write("====================================\n")
        biased_output_file.write(f"Task-Simulation Mismatch: {task_sim_mismatch}\n")
        biased_output_file.write(f"Simulation-TaskSimulation Mismatch: {sim_tasksim_mismatch}\n")
        biased_output_file.write(f"Task-TaskSimulation Mismatch: {task_tasksim_mismatch}\n\n")
        biased_output_file.close()

if __name__ == "__main__":
    analyze_by_variables()
    analyze_by_type()
