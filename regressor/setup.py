"""Setup configuration for homotopy_regressors package."""

from setuptools import setup, find_packages

setup(
    name="homotopy_regressors",
    version="0.1.0",
    author="Rodolfo H. Rodrigo",
    author_email="",
    description="Nonlinear ODE solver via discrete homotopy series",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "sympy",
    ],
    python_requires=">=3.8",
)
