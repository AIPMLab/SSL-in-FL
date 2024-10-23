import subprocess
import os

def generate_data(task, dataset_id):
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_py_path = os.path.join(current_dir, "main.py")

    # 使用双引号确保路径中包含空格时也能正确解析
    command = f'python "{main_py_path}" --work-type gen_data --task {task} --gpu 0 --dataset_id {dataset_id}'

    try:
        print(f"Running command: {command}")
        subprocess.run(command, shell=True, check=True)
        print(f"Data generation completed for task: {task}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while generating data for task {task}: {e}")

def main():
    # tasks = ["lc-biid-c10", "lc-bimb-c10", "ls-biid-c10", "ls-bimb-c10"]
    # tasks = ["lc-biid-stl10", "lc-bimb-stl10", "ls-biid-stl10", "ls-bimb-stl10"]
    tasks = ["lc-biid-fashion", "lc-bimb-fashion", "ls-biid-fashion", "ls-bimb-fashion"]

    # 由于任务使用的是STL10数据集，dataset_id = 1
    dataset_id = 2

    for task in tasks:
        print(f"Starting data generation for task: {task}")
        generate_data(task, dataset_id)

if __name__ == "__main__":
    main()
