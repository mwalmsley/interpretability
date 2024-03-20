import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="interpretability",
    version="0.0.1",
    author="Mike Walmsley, Sarah Han, Chenguang Kuang",
    author_email="walmsleymk1@gmail.com",
    description="Interpretability methods for galaxies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mwalmsley/interpretability",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Environment :: GPU :: NVIDIA CUDA"
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",  # zoobot has 3.8,
    install_requires=[
        'numpy',
        'pandas'
        # TODO
    ]
)
