from __future__ import annotations

from setuptools import setup


setup(
    name="asyrcwa",
    version="0.1.0",
    description="Python package wrapper for the RCWA C++ backend.",
    packages=["asyrcwa"],
    install_requires=["numpy"],
    python_requires=">=3.9",
)
