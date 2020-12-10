from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    "pip>=9",
    "tf-nightly",
    'tensorflow>=2.3.1'
    ]

setup(name='ArtRecognition',
      version="1.0",
      description="Project Description",
      packages=find_packages(),
      test_suite = 'tests',
      install_requires=REQUIRED_PACKAGES,
      include_package_data=True,
      scripts=[],
      zip_safe=False)
