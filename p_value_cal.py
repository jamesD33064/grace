# coding=utf-8
import os
import time  # 新增：用於計時
import torch
import argparse
import numpy as np
import pandas as pd
from scipy import stats  # 用於統計檢定
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, Spacingd, LoadImaged, EnsureTyped, 
    Orientationd, ScaleIntensityRanged, EnsureChannelFirstd, 
    AsDiscrete
)
from monai.data import Dataset, DataLoader, pad_list_data_collate, load_decathlon_datalist
from monai.metrics import DiceMetric

# --- 引入你的模型定義 ---
from monai.networks.nets import UNETR, SwinUNETR, UNet
from model.aftunet import AFTUNET 

# ==========================================
# 1. 設定區域 (請修改這裡)
# ==========================================
MODEL_REGISTRY = {
    "Aftunet":   ("aftunet", "aftunet_20251202-044901.pth"),     # Target Model
    "UNET":      ("unet", "unet_20251123-215406.pth"),           # Baseline 1
    "UNETR":     ("unetr",   "unetr_20251122-182209.pth"),       # Baseline 2
    "SwinUNETR": ("swinunetr", "swinunetr_20251124-233859.pth"), # Baseline 3
}

DATA_DIR = r"C:\Users\irisc\Documents\CV\grace\Data"
JSON_NAME = "dataset.json"

# ==========================================
# 2. 參數與環境設定
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--spatial_size", type=int, default=64)
parser.add_argument("--N_classes", type=int, default=7)
parser.add_argument("--a_min_value", type=int, default=0)
parser.add_argument("--a_max_value", type=int, default=255)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 3. 輔助函數：動態載入模型
# ==========================================
def get_model(arch_name, weights_path, args):
    model = None
    if arch_name == "unet":
        model = UNet(
            spatial_dims=3, in_channels=1, out_channels=args.N_classes,
            channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2, norm="instance"
        )
    elif arch_name == "unetr":
        model = UNETR(
            in_channels=1, out_channels=args.N_classes, img_size=(args.spatial_size,)*3,
            feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12,
            norm_name="instance", res_block=True, dropout_rate=0.0
        )
    elif arch_name == "swinunetr":
        model = SwinUNETR(
            in_channels=1, out_channels=args.N_classes, feature_size=24, 
            use_checkpoint=True, spatial_dims=3
        )
    elif arch_name == "aftunet":
        model = AFTUNET(
            in_channels=1, out_channels=args.N_classes, img_size=(args.spatial_size,)*3,
            feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12,
            norm_name="instance", res_block=True, dropout_rate=0.0, spatial_dims=3
        )
    
    if model is None:
        raise ValueError(f"Unknown architecture: {arch_name}")

    full_path = os.path.join(DATA_DIR, weights_path)
    print(f"Loading weights from: {full_path}")
    model.load_state_dict(torch.load(full_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ==========================================
# 4. 資料載入
# ==========================================
datasets = os.path.join(DATA_DIR, JSON_NAME)
test_files = load_decathlon_datalist(datasets, True, "test")

if "label" not in test_files[0]:
    raise ValueError("錯誤：Test Set 中沒有 'label' 欄位，無法計算 Dice！")

test_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=args.a_min_value, a_max=args.a_max_value, b_min=0.0, b_max=1.0, clip=True),
    EnsureTyped(keys=["image", "label"]),
])

test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

# ==========================================
# 5. 評估與統計主迴圈
# ==========================================
all_model_scores = {}  # 儲存 Dice
all_model_times = {}   # 儲存 Time

dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
post_pred = AsDiscrete(argmax=True, to_onehot=args.N_classes)
post_label = AsDiscrete(to_onehot=args.N_classes)

print(f"\nStarting Evaluation on {len(test_ds)} cases...")

for display_name, (arch_name, weights_path) in MODEL_REGISTRY.items():
    print(f"\nEvaluating Model: {display_name} ({arch_name})...")
    
    model = get_model(arch_name, weights_path, args)
    
    current_dice_scores = []
    current_inference_times = []
    
    # GPU Warm-up (為了避免第一次推論包含初始化時間，建議先跑一次空資料，或是在統計時忽略第一筆)
    # 這裡為了簡單起見，我們直接跑，但請注意學術上嚴謹的做法通常會忽略第一筆
    
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            val_inputs, val_labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            # --- ⏱️ 開始計時 (Time Start) ---
            if torch.cuda.is_available():
                torch.cuda.synchronize() # 等待所有 GPU 任務完成
            start_time = time.time()
            
            # Inference
            val_outputs = sliding_window_inference(
                val_inputs, (args.spatial_size, args.spatial_size, args.spatial_size), 
                4, model, overlap=0.8
            )
            
            # --- ⏱️ 結束計時 (Time End) ---
            if torch.cuda.is_available():
                torch.cuda.synchronize() # 等待推論真正結束
            end_time = time.time()
            
            inference_time = end_time - start_time
            current_inference_times.append(inference_time)
            
            # --- 計算 Dice ---
            val_outputs_list = [post_pred(i) for i in val_outputs]
            val_labels_list = [post_label(i) for i in val_labels]
            
            dice_metric.reset()
            dice_metric(y_pred=val_outputs_list, y=val_labels_list)
            
            score = dice_metric.aggregate().item()
            current_dice_scores.append(score)
            
            print(f"  Case {i+1}: Dice={score:.4f}, Time={inference_time:.4f}s")
            
    all_model_scores[display_name] = current_dice_scores
    all_model_times[display_name] = current_inference_times
    
    print(f"Finished {display_name}.")
    print(f"  > Avg Dice: {np.mean(current_dice_scores):.4f}")
    print(f"  > Avg Time: {np.mean(current_inference_times):.4f} s/case")

# ==========================================
# 6. 統計比較函數
# ==========================================
def perform_statistical_test(target_name, data_dict, metric_name, higher_is_better=True):
    print(f"\n" + "="*50)
    print(f"Statistical Comparison: {metric_name} (vs {target_name})")
    print(f"Note: {'Higher' if higher_is_better else 'Lower'} is better.")
    print("="*50)
    
    if target_name not in data_dict:
        print(f"Target model {target_name} not found.")
        return

    target_data = data_dict[target_name]
    results_list = []

    for model_name, data in data_dict.items():
        if model_name == target_name:
            continue
        
        # Paired t-test
        t_stat, p_val = stats.ttest_rel(target_data, data)
        
        # 計算平均差異 (Target - Baseline)
        diff_mean = np.mean(target_data) - np.mean(data)
        
        # 判斷結果好壞
        if higher_is_better:
            # Dice: Diff > 0 代表 Target 贏
            better = diff_mean > 0
        else:
            # Time: Diff < 0 代表 Target 贏 (時間更短)
            better = diff_mean < 0
            
        results_list.append({
            "Comparison": f"{target_name} vs {model_name}",
            "Mean Diff": diff_mean,
            "t-value": t_stat,
            "p-value": p_val,
            "Significant": "Yes" if p_val < 0.05 else "No",
            "Target Wins?": "YES" if better else "NO"
        })

    df = pd.DataFrame(results_list)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df)

# ==========================================
# 7. 執行兩組統計分析
# ==========================================

# 1. 比較 Dice Score (越高越好)
perform_statistical_test("Aftunet", all_model_scores, "Dice Score", higher_is_better=True)

# 2. 比較 Inference Time (越低越好)
perform_statistical_test("Aftunet", all_model_times, "Inference Time (s)", higher_is_better=False)

print("\n[Professor's Academic Note]")
print("1. 時間統計中的負 t-value (Negative t-value) 代表 Aftunet 的時間比對照組短（速度快）。")
print("2. 確保在比較時間時，Batch Size 設為 1，這樣比較的是『單個病例的延遲 (Latency)』。")
print("3. 若某個模型雖然 Dice 顯著較高，但 Time 也顯著較高，這就是典型的 Trade-off，需在論文討論區解釋。")