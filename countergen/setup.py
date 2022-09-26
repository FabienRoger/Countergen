import pathlib
from setuptools import setup, find_packages

"""
To upload the package, run
python setup.py sdist
twine upload --repository-url https://upload.pypi.org/legacy/ dist/countergen-VERSION.tar.gz
"""

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
VERSION = (HERE / "VERSION").read_text()
REQUIREMENTS = (HERE / "requirements.txt").read_text().splitlines()

setup(
    name="countergen",
    version=VERSION,
    description="A counterfactual dataset generator to evaluate language model.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="SaferAI",
    author_email="saferai.audit@gmail.com",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/FabienRoger/Counterfactual-Dataset-Generator",
    python_requires=">=3.7",
    include_package_data=True,
    install_requires=REQUIREMENTS,
    entry_points={
        "console_scripts": [
            "countergen = countergen:run",
        ],
    },
)
