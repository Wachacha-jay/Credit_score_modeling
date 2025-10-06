"""
Setup script for Loan Approval System
"""
from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    """
    Read requirements from file
    
    Args:
        file_path: Path to requirements file
        
    Returns:
        List of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements


setup(
    name='loan_approval_system',
    version='1.0.0',
    author='James w. Ngaruiya',
    author_email='jameswachacha@gmail.com',
    description='Credit scoring ML  models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Wachacha-jay/Credit_score_modeling',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'train-model=src.pipelines.training_pipeline:main',
            'predict=src.pipelines.prediction_pipeline:main',
            'run-api=api.main:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)