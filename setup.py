from setuptools import setup, find_packages

setup(
    name='kamaji',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',  # If you need plotting
        'pytest',      # For testing (optional)
        'pyyaml'
    ],
    include_package_data=True,
    package_data={
        'kamaji': ['examples/*.py', 'docs/*.md'],
    },
    entry_points={
        'console_scripts': [
            'run_simulation=kamaji.simulation:main',  # Example if you want a CLI tool
        ],
    },
)
