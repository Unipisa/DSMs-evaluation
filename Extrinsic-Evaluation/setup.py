from setuptools import setup, find_packages

setup(
    name='exeval',
    version='0.0.1',
    packages=find_packages(),
    package_data={'':['*.txt.gz']},

    install_requires=['numpy', 'scikit-learn', 'keras'],

    entry_points={
        'console_scripts':[
            'exeval = exeval.__main__:main'
        ]
    },

    description='Extrinsic Evaluation Tasks',
    license='MIT',
    author='Amaru Cuba Gyllensten',
)
