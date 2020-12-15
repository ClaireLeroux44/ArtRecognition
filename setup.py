from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f: 
  content = f.readlines() 
requirements = [x.strip() for x in content if 'git+' not in x]

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
      install_requires=requirements,
      include_package_data=True,
      scripts=[],
      zip_safe=False)


