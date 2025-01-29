from setuptools import setup, find_packages

setup(
    name="mirai_chile",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here, e.g.,
        # 'torch>=1.11.0',
        'numpy>=1.24.4',
        'setuptools>=73.0.1',
        'pandas>=2.0.3',
        'pillow>=10.3.0',
        'torch>=2.4.1',
        'torchvision>=0.19.1',
        'tqdm>=4.62.3',
        'matplotlib>=3.7.5',
        'scikit-learn>=1.3.2',
        'pycox>=0.3.0'
    ],
    entry_points={
        'console_scripts': [
            'mirai-chile-predict = mirai_chile.script:main',
        ]
    }
)