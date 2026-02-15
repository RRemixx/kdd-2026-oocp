# setup.py
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="core",                 # Replace with your desired package name
    version="0.1.0",                   # Choose any initial version number
    author="anonymous",                   # Replace with your name or organization
    author_email="anonymous@example.com",  # Replace with your email
    description="Optimized Adaptive Conformal Prediction for Multi-step Time Series Forecasting",  # A short description of your package
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)