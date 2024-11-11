from setuptools import setup, find_packages

setup(
    name='mtuq_cmaes',
    version='0.1.0',
    packages=find_packages(),
    package_data={'': ['examples/*.py', 'utils/*.py']},
    install_requires=[
        'mtuq',  # MTUQ as a dependency
    ],
    python_requires='>=3.8',
    author='Julien Thurin',
    description='CMA-ES pluggin for MTUQ',
    license='BSD-3-Clause',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
)
