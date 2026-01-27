import sys
import subprocess
from setuptools import setup, find_packages  # type: ignore

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

packages = find_packages(where=".", exclude=["mindie_llm_examples", "mindie_service_examples"])

setup(
    name='mindie_ais_bench_backend',
    version='0.0.1',
    description='mindie_ais_bench_backend',
    long_description=long_description,
    packages=packages,
    include_package_data=True,
    keywords='mindie_ais_bench_backend',
    install_requires=required,
    python_requires='>=3.8.0',
    entry_points={
        'ais_bench.benchmark_plugins': [
            'mindie_plugin = mindie_ais_bench_backend',
        ],
    },
)