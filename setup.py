from setuptools import setup, find_packages

setup(
    name='microspike',
    version='0.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'numba',
        'matplotlib',
    ],
    author='Mehran Faraji',
    author_email='mehranfaraji377@gmail.com',
    description='A library for simulating Spiking Neural Networks.',
    url='https://github.com/mehranfaraji/microspike',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
