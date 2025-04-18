#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="finllm-insight",
    version="0.1.0",
    author="WXYkkmml2",
    author_email="example@mail.com",
    description="Financial Report Analysis and Stock Prediction System Powered by Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WXYkkmml2/FinLLM-Insight",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "finllm-pipeline=src.pipeline:main",
            "finllm-rag=src.rag.rag_component:main",
        ],
    },
)
