import setuptools

setuptools.setup(
    name="BatchJAX", 
    version="0.0.1",
    author="O Hamelijnck",
    author_email="ohamelijnck@turing.ac.uk",
    description="Add support for Jax and OBJAX to automatically batch over lists and ModuleLists",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/defaultobject/batchjax",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

