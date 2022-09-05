from setuptools import setup

setup(
    name='stable_diffusion_executor',
    version='0.1.0',    
    description='Fast and memory-efficient upscaling without artifacts',
    url='https://github.com/peterwilli/Diff2X',
    author="Peter Willemsen",
    author_email="peter@codebuffet.co",
    license='Apache License',
    packages=['stable_diffusion_executor'],
    install_requires=[
        'torch>=1.6',
        'diffusers',
        'jina>=3.7',
        'transformers>=4.21'
    ],
    classifiers=[
        'Development Status :: 3 - Release',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
)