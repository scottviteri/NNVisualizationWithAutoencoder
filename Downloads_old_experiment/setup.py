from setuptools import setup, find_packages

setup(
    name='deep_dream_llm',
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
)
