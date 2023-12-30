import re,os,sys
from setuptools import setup
from distutils.extension import Extension

def get_requirements():
    fname = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(fname, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements

def find_from_doc(what='version'):
    f = open(os.path.join(os.path.dirname(__file__), 'fastlens/__init__.py')).read()
    match = re.search(r"^__%s__ = ['\"]([^'\"]*)['\"]"%(what), f, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find %s string."%what)

setup(
    name='fastlens',
    version=find_from_doc('version'),
    description='fft based microlensing package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url=find_from_doc('url'),
    author=find_from_doc('author'),
    author_email='sunaosugiyama@gmail.com',
    keywords=['microlensing', 'fft'],
    packages=['fastlens'],
    include_package_data=True,
    install_requires=get_requirements(),
    classifiers=['Development Status :: 5 - Production/Stable',
                 'Programming Language :: Python', 
                 'Intended Audience :: Science/Research',
                 'Programming Language :: Python :: 2',
                 'Programming Language :: Python :: 3']
)
