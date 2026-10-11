# python tracking_module.py
import json
import pandas as pd
from pathlib import Path


def log_run_json(results_folder, experiment_data, kind='patch'):
    """把單一 run 的 flat 指標 dict 寫成 {results_folder}/run_metrics.json。

    這是新版 source of truth：每個 fold（patch）與每個 recon 各寫一份，
    restructure_excel.py 直接讀這些 json 產生多 sheet 統計檔，不再依賴舊 wide-format Excel。
    kind: 'patch' | 'recon'（寫進 json 的 _kind 欄，供 restructure 區分）。
    """
    out = Path(results_folder)
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for k, v in experiment_data.items():
        data[k] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
    data['_kind'] = kind
    p = out / 'run_metrics.json'
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    print(f"實驗記錄(JSON)已寫入: {p}")


def log_to_excel(excel_path_str, experiment_data, desired_order):
    excel_path = Path(excel_path_str)
    excel_path.parent.mkdir(exist_ok=True, parents=True)

    # ⭐ 核心修改：使用新的 'Experiment_name' 來命名欄位
    column_header = f"{experiment_data.get('run_timestamp')}_{experiment_data.get('Experiment_name', 'Unnamed_Experiment')}"
    
    new_log_series = pd.Series({k: str(v) for k, v in experiment_data.items()}, name=column_header)

    if excel_path.exists():
        try:
            log_df = pd.read_excel(excel_path, index_col=0, engine='openpyxl')
            log_df = log_df.join(new_log_series, how='outer')
        except Exception as e:
            print(f"⚠️ 讀取現有 Excel 檔案 {excel_path} 失敗: {e}。將建立一個新檔案。")
            log_df = new_log_series.to_frame()
    else:
        log_df = new_log_series.to_frame()

    final_order = [key for key in desired_order if key in log_df.index]
    other_keys = [key for key in log_df.index if key not in final_order]
    final_order.extend(other_keys)
    
    log_df = log_df.reindex(final_order)

    try:
        log_df.index.name = "Metric / Parameter"
        log_df.to_excel(excel_path, index=True, engine='openpyxl')
        print(f"實驗記錄已更新至: {excel_path} (行列互換格式)")
    except Exception as e:
        print(f"寫入 Excel 檔案時發生錯誤: {e}")