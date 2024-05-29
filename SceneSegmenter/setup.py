from setuptools import setup, find_packages

# Specify the dependencies with version ranges
install_requires = [
    'matplotlib>=3.4.0,<4.0.0',
    'nltk>=3.8.0,<4.0.0',
    'numpy>=1.24.0,<2.0.0',
    'scipy>=1.10.0,<2.0.0',
    'sentence-transformers>=2.2.0,<3.0.0',
    'torch>=2.0.0,<3.0.0',
    'transformers>=4.32.0,<5.0.0',
]

setup(
    name='my_project',  # Name of your project
    version='0.1.0',  # Version of your project
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=install_requires,  # Use the flexible dependencies list
    extras_require={  # Optional dependencies
        'dev': [
            'pytest>=5.4.1,<6.0.0',
            'flake8>=3.8.3,<4.0.0',
        ],
    },
    entry_points={  # Entry points for executable scripts
        'console_scripts': [
            'my_project=my_package.module:main',  # Adjust accordingly
        ],
    },
    include_package_data=True,  # Include other files specified in MANIFEST.in
    package_data={  # Package specific data
        'my_package': ['data/*.dat'],
    },
    author='Your Name',  # Author's name
    author_email='your.email@example.com',  # Author's email
    description='A package for automatic scene segmentation and visualization.',  # Short description
    long_description=open('README.md').read(),  # Long description read from README.md
    long_description_content_type='text/markdown',  # Description content type
    url='https://github.com/yourusername/my_project',  # URL to your project
    classifiers=[  # Classifiers to categorize your project
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',  # Python version requirement
)
