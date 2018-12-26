"""The project setup module
"""


from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), mode='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='pyautoarima',
    version='1.0.0',
    description='Enhanced auto ARIMA based time series forecasting',
    url='https://github.com/prashantnbangar/pyautoarima',
    author='Prashant Bangar',
    author_email='prashantbangar@live.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Time Series Forecasting :: ARIMA',
        'License :: GNU General Public License v3.0',
        'Programming Language :: Python :: 3.7',
    ],

    keywords='python forecasting ARIMA automatic',

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    install_requires=['statsmodel'],

    # The following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    entry_points={  # Optional
        'console_scripts': [
            'forecast=Forecaster:main',
        ],
    },

    # List additional URLs that are relevant to your project as a dict.
    #
    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    project_urls={
        'Source': 'https://github.com/prashantnbangar/pyautoarima',
    },
)
