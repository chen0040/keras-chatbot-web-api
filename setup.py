from setuptools import setup

setup(
    name='chatbot_web',
    packages=['chatbot_web'],
    include_package_data=True,
    install_requires=[
        'flask',
        'keras',
        'sklearn',
        'numpy',
        'nltk',
        'h5py'
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
)