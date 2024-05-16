from setuptools import setup

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setup(
    name='PeptideModels',
    version='1.0.0',
    packages=['peptide_models', 'peptide_models.notebooks', 'peptide_models.supporting_experiments'],
    url='https://github.com/amp91/PeptideModels',
    license='Apache License, Version 2.0',
    author='Anna M. Puszkarska',
    author_email='anpuszkarska@gmail.com',
    description='Code for peptide ligand design with machine learning models.',
    long_description=long_description,
)
