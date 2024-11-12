from setuptools import setup, find_packages

setup(
    name="surf_shelter_training_pkg",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "requests",
        "joblib",
        "scikit-learn",
        "matplotlib",
        "google-cloud-storage",
        "warcio",
        "python-dotenv",
    ],
    package_data={
        'model_trainer_v0': ['warc.paths.gz'],
    },
    python_requires=">=3.7",
)
