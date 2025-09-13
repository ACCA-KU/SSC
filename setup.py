from setuptools import setup, find_packages

setup(
    name="SSC",
    version="1.1.0",
    author="Jinyong Park",
    author_email="phillip1998@korea.ac.kr",
    description="Graph neural network model for prediction of experimental molecular property reflecting Solvatochromic subgroup contribution approach",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/phillip1998/SGC",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
"numpy",
"pandas",
"scikit-learn",
"scipy",
"matplotlib",
"seaborn",
"torch>=2.2,<=2.4.0",
"rdkit>=2022",
"tqdm",
"joblib",
"pydantic",
"dgl",
"D4CMPP==1.26.2",
"MoleculeSculptor"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
)