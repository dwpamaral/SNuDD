from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='snudd',
    url='https://github.com/dwpamaral/SNuDD.git',
    version="1.0",
    author='Dorian Amaral',
    packages=['snudd'],
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
