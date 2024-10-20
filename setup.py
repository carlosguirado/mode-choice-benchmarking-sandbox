from setuptools import setup, find_packages

setup(
       name="mcbs",
       version="0.1.0",
       packages=find_packages(),
       install_requires=[
           "numpy",
           "pandas",
           "scikit-learn",
           "matplotlib",
           "seaborn",
           "biogeme"
       ],
       author="Carlos Guirado",
       author_email="guirado@berkeley.edu",
       description="Mode Choice Benchmarking Sandbox (MCBS)",
       long_description=open("README.md").read(),
       long_description_content_type="text/markdown",
       url="https://github.com/carlosguirado/mode-choice-benchmarking-sandbox",
       classifiers=[
           "Programming Language :: Python :: 3",
           "License :: OSI Approved :: MIT License",
           "Operating System :: OS Independent",
       ],
   )