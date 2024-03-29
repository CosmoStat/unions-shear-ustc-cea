#! /usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# Set the package release version
major = 0
minor = 1
patch = 0

# Set the package details
name = 'unions_wl'
version = '.'.join(str(value) for value in (major, minor, patch))
author = 'Martin Kilbinger'
email = 'martin.kilbinger@cea.fr'
gh_user = 'martinkilbinger'
url = 'https://github.com/{0}/{1}'.format(gh_user, name)
year = '2022'
description = 'Weak-lensing library for use with UNIONS data'
license = 'MIT'

# Set the package classifiers
python_versions_supported = ['3.6', '3.7', '3.8', '3.9']
os_platforms_supported = ['Unix', 'MacOS']

lc_str = 'License :: OSI Approved :: {0} License'
ln_str = 'Programming Language :: Python'
py_str = 'Programming Language :: Python :: {0}'
os_str = 'Operating System :: {0}'

classifiers = (
    [lc_str.format(license)] + [ln_str] +
    [py_str.format(ver) for ver in python_versions_supported] +
    [os_str.format(ops) for ops in os_platforms_supported]
)

# Source package description from README.md
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Source package requirements from requirements.txt
with open('requirements.txt') as open_file:
    install_requires = open_file.read()

# Source test requirements from develop.txt
with open('develop.txt') as open_file:
    tests_require = open_file.read()

# Source doc requirements from docs/requirements.txt
with open('docs/requirements.txt') as open_file:
    docs_require = open_file.read()


# Find scripts
def find_scripts():

    sdir = 'scripts'

    res = [
        os.path.join(sdir, val) for val in os.listdir(sdir) if
            (val.endswith('.py') or val.endswith('.sh'))
            and '__init__' not in val
    ]

    return res


setup(
    name=name,
    author=author,
    author_email=email,
    version=version,
    license=license,
    url=url,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    scripts=find_scripts(),
    install_requires=install_requires,
    python_requires='>=3.6',
    setup_requires=['pytest-runner'],
    tests_require=tests_require,
    extras_require={'develop': tests_require + docs_require},
    classifiers=classifiers,
)
