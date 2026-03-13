from setuptools import find_packages, setup

setup(
    name="lyrics-generation-project",
    version="0.1.0",
    description="Song lyric generation with n-gram and neural language models",
    author="APS360 Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "PyYAML>=6.0.1",
        "nltk>=3.8.1",
        "huggingface-hub>=0.21.0",
        "datasets>=2.17.0",
        "scikit-learn>=1.4.0",
        "matplotlib>=3.8.2",
        "seaborn>=0.13.1",
        "sacrebleu>=2.4.0",
        "tqdm>=4.66.1",
    ],
)
