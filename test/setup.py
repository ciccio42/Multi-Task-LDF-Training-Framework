# setup.py
from setuptools import setup, find_packages

setup(
    name='multi_task_test',
    version='0.0.1',
    include_package_data=True,
    packages=find_packages(exclude=['multi_task_test_gnn']),
)

if __name__ == "__main__":
    packages = find_packages(exclude=['test_models'])
    print(packages)
