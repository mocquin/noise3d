# -*- coding: utf-8 -*-
import os
import pathlib
import re

from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

here = os.path.abspath(os.path.dirname(__file__))

VERSIONFILE="noise3d/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

README = (HERE / "README.md").read_text()

setup(
    name="noise3d",
    version=verstr,
    description="Noise analysis based on 3D noise model",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/mocquin/noise3d",
    author="mocquin",
    author_email="mocquin@me.com",
    license="MIT",
    keywords='noise 3Dnoise',
    packages=find_packages(exclude=("tests", )),
    # add content of MANIFEST
    include_package_data=True,
    install_requires=["numpy", 
                      "scipy", 
                      "matplotlib",
                     ]
)
