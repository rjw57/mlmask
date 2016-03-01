from setuptools import setup, find_packages

setup(
    name='mlmask',
    version='0.0.1',
    author='Rich Wareham',
    packages=find_packages(),

    install_requires=[
        'numpy',
        'scipy',
        'dtcwt',
        'docopt',
        'imageio',
        'scikit-image',
        'scikit-learn',
    ],
)
