#!/usr/bin/env python3
"""
Setup script for the Quadcopter RL Training project.
"""

from setuptools import setup, find_packages
import os


# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    return []


setup(
    name="quadcopter-rl",
    version="1.0.0",
    description="Reinforcement Learning for Quadcopter Control using SAC and HER",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/quadcopter-rl",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="reinforcement-learning, quadcopter, SAC, HER, gymnasium",
    entry_points={
        "console_scripts": [
            "quadcopter-train-sac=training.train_sac:main",
            "quadcopter-train-her=training.train_her:main",
            "quadcopter-evaluate=evaluation.evaluate_model:main",
            "quadcopter-tune=training.hyperparameter_tuning:main",
        ],
    },
    include_package_data=True,
    package_data={
        "assets": ["**/*"],
    },
)
