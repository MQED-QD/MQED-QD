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
            "mqed_RET = mqed.analysis.RET:main",
            "mqed_lindblad = mqed.Lindblad.run_quantum_dynamics:mqed_lindblad",
            "mqed_nhse = mqed.Lindblad.run_quantum_dynamics:mqed_nhse",
            "mqed_nhse_disorder = mqed.Lindblad.run_disorder:run_disorder",
            "mqed_plot_sqrt_msd = mqed.plotting.plot_sqrt_msd:main",
            "mqed_plot_msd = mqed.plotting.plot_msd:main",
            "mqed_BEM_compare = mqed.BEM.compare_const:main",
            "mqed_BEM_compare_silver = mqed.BEM.compare_silver:main",
            "mqed_BEM_compute_peff = mqed.BEM.compute_peff:main",
            "mqed_BEM_compare_dyadic = mqed.BEM.compare_BEM_dyadic:main",
            "mqed_BEM_reconstruct_GF = mqed.BEM.reconstruct_GF:main",
            "mqed_plot_IPR = mqed.plotting.plot_ipr:main",
        ]
    },

    # Specify the minimum Python version required
    python_requires=">=3.8",
)