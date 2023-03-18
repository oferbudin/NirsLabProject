import os
from setuptools import setup


try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

PACKAGE_NAME = 'NirsLabProject'

# parse_requirements() returns generator of pip.req.InstallRequirement objects
project_absolute_path = os.path.dirname(os.path.abspath(__file__))
install_reqs = parse_requirements(os.path.join(project_absolute_path, PACKAGE_NAME, 'env', 'requirements.txt'), session=False)
reqs = [str(ir.requirement) for ir in install_reqs]

setup(
    name=PACKAGE_NAME,
    author='',
    version='0.1',
    install_requires=reqs,
    py_modules=[PACKAGE_NAME],
    packages=[PACKAGE_NAME]
)