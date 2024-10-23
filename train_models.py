import subprocess
import os


def train_model(task, gpu="0,1,2,3,4", frac_client=0.05, dataset_id=2):
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_py_path = os.path.join(current_dir, "main.py")

    # 构建运行命令，并将main_py_path用双引号包围，以处理路径中的空格
    command = (
        f'python "{main_py_path}" '
        f"--gpu {gpu} "
        f"--work-type train "
        f"--model fedmatch "
        f"--frac-client {frac_client} "
        f"--task {task} "
        f"--dataset_id {dataset_id}"  # 添加 dataset_id 参数
    )

    # 打印当前目录和执行的命令，用于调试
    print(f"Current script directory: {current_dir}")
    print(f"Running command: {command}")

    try:
        # 执行命令
        subprocess.run(command, shell=True, check=True)
        print(f"Command executed successfully for task: {task}")
    except subprocess.CalledProcessError as e:
        # 捕获并打印执行错误
        print(f"An error occurred while executing the command for task {task}: {e}")


def main():
    # 定义任务列表
    tasks = ["ls-biid-fashion"]

    for task in tasks:
        print(f"Starting training for task: {task}")
        train_model(task)
        print(f"Training completed for task: {task}")


if __name__ == "__main__":
    main()
