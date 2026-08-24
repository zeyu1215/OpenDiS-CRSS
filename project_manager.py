import subprocess
import sys
import csv
import time
import itertools

# --- 大管家的核心配置 ---
WORKER_SCRIPT = "project_worker.py"
STRESS_LOWER_BOUND = 1e7 
STRESS_UPPER_BOUND = 1e9 # 已根据你的经验拓宽至 500 MPa
TOLERANCE = 1e7          

# --- 高通量参数矩阵 (Ni 300K) ---
RADII_LIST = [350] 
MEDGE_LIST = [59520, 61205, 62890]
MSCREW_LIST = [33330, 37500, 41670]

# 生成 27 组排列组合
TASKS = list(itertools.product(RADII_LIST, MEDGE_LIST, MSCREW_LIST))

def call_worker(radius, medge, mscrew, stress, run_id):
    """大管家派发任务，并监听打工人的汇报结果"""
    cmd = [
        sys.executable, WORKER_SCRIPT, 
        "--radius", str(radius), 
        "--medge", str(medge),
        "--mscrew", str(mscrew),
        "--stress", str(stress), 
        "--id", str(run_id)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if "OUTCOME: PASS" in result.stdout:
        return True
    elif "OUTCOME: FAIL" in result.stdout:
        return False
    else:
        print(f"⚠️ Worker error! (Stress={stress:.1f})")
        print(result.stderr)
        return False 

def main():
    print("="*60)
    print(f" 🤖 ML Dataset Automated Generation Manager ({len(TASKS)} Tasks)")
    print("="*60)
    
    csv_filename = "ni_300k_training_data.csv"
    run_counter = 1
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 更新表头，加入 Mobility 参数
        writer.writerow(["Radius_b", "M_edge", "M_screw", "Critical_Stress_Pa"]) 
        
        # --- 外层循环：遍历 27 个任务 ---
        for task_idx, (r, medge, mscrew) in enumerate(TASKS, 1):
            print(f"\n🎯 [Task {task_idx}/{len(TASKS)}] Params: R={r}, Medge={medge}, Mscrew={mscrew}")
            start_time = time.time()
            
            stress_low = STRESS_LOWER_BOUND
            stress_high = STRESS_UPPER_BOUND
            
            # --- 内层循环：二分法锁定临界应力 ---
            while (stress_high - stress_low) > TOLERANCE:
                stress_mid = (stress_low + stress_high) / 2.0
                print(f"    ▶️ Testing Stress = {stress_mid:.1f} Pa")
                
                passed = call_worker(r, medge, mscrew, stress_mid, run_counter)
                run_counter += 1
                
                if passed:
                    print("      ✅ Passed (Shrinking upper bound)")
                    stress_high = stress_mid
                else:
                    print("      ❌ Stuck (Raising lower bound)")
                    stress_low = stress_mid
            
            tau_c = (stress_low + stress_high) / 2.0
            elapsed = (time.time() - start_time) / 60.0
            print(f"🎉 Task {task_idx} Complete! CRSS: {tau_c:.1f} Pa (Took: {elapsed:.1f} min)")
            
            writer.writerow([r, medge, mscrew, tau_c])
            file.flush() 
            
    print("\n✅ All 27 tasks completed, data safely saved to", csv_filename)

if __name__ == "__main__":
    main()