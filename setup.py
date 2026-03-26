from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

version_ns = {}
exec((ROOT / "pipeline" / "version.py").read_text(encoding="utf-8"), version_ns)

setup(
    name="pipeline",
    version=version_ns["__version__"],
    description="Pipeline for 7-dimensional Telescope",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(exclude=("test", "test.*")),
    python_requires=">=3.10",  # match case syntax in checker.py
    install_requires=[
        "numpy>=1.25",
    ],
    include_package_data=True,  # include files from MANIFEST.in if present
    classifiers=[  # helps PyPI & tooling understand the project
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        # "Operating System :: OS Independent",
    ],
)
