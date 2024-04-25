from setuptools import setup, find_packages

setup(
    name='Multi-Task-IL',
    version='0.0.1',
    include_package_data=True,
    packages=find_packages(),
    package_data={"multi_task_il.datasets": ['*'],
                  "multi_task_il.utils": ['*'],
                  "multi_task_il.models": ['*'],
                  "multi_task_il_gnn.datasets": ['*'],
                  "multi_task_il_gnn.utils": ['*'],
                  "multi_task_il_gnn.models": ['*'],
                  "train_scripts.gnn": ['*'],
                  "train_scripts": ['*']
                  }
)

if __name__ == "__main__":
    packages = find_packages()
    print(packages)
