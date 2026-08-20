import os
import time
import pandas as pd

MASTER_CSV = r"D:\tony dataset\all_datasets_features_12_master.csv"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_monitor():
    last_count = 0
    while True:
        clear_screen()
        print("================================================================================")
        print("           ⚡ TONY DATASET — LIVE EXTRACTION & CUT MONITOR")
        print("================================================================================")
        
        if not os.path.exists(MASTER_CSV):
            print("[!] Master CSV not found yet. Waiting for extraction daemon...")
            time.sleep(3)
            continue
            
        try:
            df = pd.read_csv(MASTER_CSV)
            total_cuts = len(df)
            unique_ds = df["dataset_id"].nunique()
            stable_cnt = (df["label"] == 0).sum()
            chatter_cnt = (df["label"] == 1).sum()
            
            delta_str = f"(+{total_cuts - last_count} new cuts)" if last_count > 0 and total_cuts > last_count else ""
            last_count = total_cuts
            
            print(f"[*] TOTAL VERIFIED CUTS  : {total_cuts:,} {delta_str}")
            print(f"[*] UNIQUE DATASETS      : {unique_ds} / 155 ({unique_ds/155*100:.1f}%)")
            print(f"[*] STABLE CUTS (Label 0): {stable_cnt:,} ({stable_cnt/total_cuts*100:.1f}%)")
            print(f"[*] CHATTER CUTS(Label 1): {chatter_cnt:,} ({chatter_cnt/total_cuts*100:.1f}%)")
            print("--------------------------------------------------------------------------------")
            print("RECENT DATASETS EXTRACTED:")
            top_ds = df["dataset_id"].value_counts().sort_index().tail(10)
            for ds_id, count in top_ds.items():
                bar = "█" * min(30, int(count / 15))
                print(f"  Dataset {ds_id:3d} : {count:4d} cuts  {bar}")
                
            print("--------------------------------------------------------------------------------")
            print("LATEST 5 CUTS PROCESSED:")
            latest = df.tail(5)[["dataset_id", "file", "omega_rpm", "axial_depth_m", "label"]].iloc[::-1]
            for _, r in latest.iterrows():
                status = "🟢 STABLE " if r["label"] == 0 else "🔴 CHATTER"
                print(f"  [{status}] DS {int(r['dataset_id']):3d} | {r['file']:<22} | {r['omega_rpm']:5.0f} RPM | {r['axial_depth_m']*1000:5.2f} mm depth")
                
            print("================================================================================")
            print("Press Ctrl+C to stop monitor. Auto-refreshing every 3s...")
        except Exception as e:
            print(f"[!] Reading CSV... ({e})")
            
        time.sleep(3)

if __name__ == "__main__":
    run_monitor()
