import os
from setuptools import setup, find_packages

PACKAGE_NAME = 'dldb'


def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    assert False, "'{0}' not found in '{1}'".format(key, module_path)


setup(
    name=PACKAGE_NAME,
    version=read_package_variable('__version__'),
    description='Deep learning for relational datasets with a time-component',
    packages=find_packages(),
    python_requires='>=3',
    install_requires=[
        'featuretools>=0.1.20',
        'keras>=2.1.4',
        'scikit-learn>=0.19.1',
        'tensorflow>=1.6.0',
    ],
    url='https://github.com/HDI-Project/DL-DB',
)
