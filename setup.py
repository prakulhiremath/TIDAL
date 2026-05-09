"""
TIDAL: Temporal AI for Early Detection of Latent Financial Market Instability
Setup configuration for pip-installable package.
"""

from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text()

setup(
    name="tidal-finance",
    version="0.1.0",
    author="TIDAL Research Team",
    description="Temporal AI for Early Detection of Latent Financial Market Instability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/TIDAL",
    packages=find_packages(exclude=["tests", "notebooks"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "xgboost>=1.7.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "loguru>=0.7.0",
        "rich>=13.0.0",
        "h5py>=3.8.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "plotly>=5.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tidal-train=experiments.run_tidal:main",
            "tidal-eval=evaluation.benchmark_compare:main",
            "tidal-viz=visualizations.paper_figures:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="financial instability, market surveillance, temporal AI, early warning, order book",
)
