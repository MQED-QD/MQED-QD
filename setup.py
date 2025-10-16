# setup.py

from setuptools import setup, find_packages

setup(
    # Core package metadata
    name="mqed",
    version="0.1.0",
    description="A tool for calculating Dyadic Green's Functions in layered media.",
    author="Hsing-ta Chen Lab", # Change this to your name
    author_email="", # Change this to your email

    # Automatically find all Python packages in your project
    packages=find_packages(),

    include_package_data=True,

    # List of dependencies required for the package to run
    install_requires=[
    ],

    # Creates the command-line tool
    entry_points={
        "console_scripts": [
            "mqed_GF = mqed.Dyadic_GF.main:run_simulation",
            "mqed_RET = mqed.analysis.RET:main"
            "mqed_lindblad = mqed.Lindblad.cli:mqed_lindblad"
            "mqed_nhse = mqed.Lindblad.cli:mqed_nhse"
        ]
    },

    # Specify the minimum Python version required
    python_requires=">=3.8",
)