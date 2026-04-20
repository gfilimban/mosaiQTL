from setuptools import setup, find_packages

setup(
    name='mosaiqtl',
    version='0.2.0',
    description='model-agnostic haplotype-based qtl mapping for multiparent populations',
    license='GPL-3.0-or-later',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'build-cache=mosaiqtl.genotype.genomecache:build',
        ],
    },
    install_requires=[
        'numpy>=1.21',
        'scipy>=1.7',
        'pandas>=1.3',
        'h5py>=3.0',
        'anndata>=0.8',
        'matplotlib>=3.5',
        'seaborn>=0.12',
        'pydantic>=2.0',
        'click>=8.1',
    ],
    extras_require={
        'sc': [
            'scanpy>=1.9',
        ],
        'r': [
            'rpy2>=3.4',
        ],
        'dev': [
            'pytest>=8.0',
            'pytest-cov>=5.0',
            'pytest-benchmark>=4.0',
            'hypothesis>=6.0',
            'black>=24.0',
            'flake8>=7.0',
            'mypy>=1.10',
            'interrogate>=1.7',
            'bandit>=1.7',
            'statsmodels>=0.13',
        ],
    },
)
