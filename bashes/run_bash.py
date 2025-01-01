import os
import subprocess
import re
import time

bash_script = "/home/rsofnc000/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes/real_train_keypoint_detection.sh"
checkpoint_folder = "/home/rsofnc000/checkpoint_save_folder/Real-1Task-pick_place-KP-Finetune-Batch32"
bash_argument="pick_place"
max_epochs = 90  # Set your maximum number of epochs here

def get_highest_epoch(folder):
    highest_epoch = -1
    for file in os.listdir(folder):
        match = re.match(r'model_save-(\d+)\.pt', file)
        if match:
            epoch_number = int(match.group(1))
            if epoch_number > highest_epoch:
                highest_epoch = epoch_number
    return highest_epoch

def run_bash_script():
    while True:
        result = subprocess.run(['sbatch', bash_script, bash_argument], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error submitting job: {result.stderr}")
            return
        
        # Extract job ID from sbatch output
        job_id = None
        for line in result.stdout.split('\n'):
            if "Submitted batch job" in line:
                job_id = line.split()[-1]
                break
        
        if job_id is None:
            print("Failed to get job ID from sbatch output")
            return

        print(f"Job {job_id} submitted. Waiting for completion...")
        
        
        # Poll the job status using squeue
        while True:
            result = subprocess.run(['squeue', '--job', job_id], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error checking job status: {result.stderr}")
                return

            if job_id not in result.stdout:
                print(f"Job {job_id} completed.")
                break

            time.sleep(10)  # Wait for 10 seconds before polling again
        
        
        highest_epoch = get_highest_epoch(checkpoint_folder)
        print(f"Highest epoch reached: {highest_epoch}")
        
        if highest_epoch >= max_epochs:
            print("Reached the maximum number of epochs. Exiting.")
            break
        else:
            print("Restarting the bash script...")
            time.sleep(5)  # Optional: wait for a few seconds before restarting

if __name__ == "__main__":
    run_bash_script()