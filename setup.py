from setuptools import setup, find_packages

setup(
    name='mosaiqtl',
    version='0.1.0',
    description='haplotype-based qtl mapping for multiparent populations with nebula-ln single-cell modeling',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.21',
        'scipy>=1.7',
        'pandas>=1.3',
    ],
    extras_require={
        'full': [
            'h5py>=3.0',
            'scanpy>=1.9',
            'anndata>=0.8',
        ],
        'r_bridge': [
            'rpy2>=3.4',
        ],
    },
)
