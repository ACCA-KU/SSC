from setuptools import setup, find_packages

setup(
    name="SSC",
    version="1.1.0",
    author="Jinyong Park",
    author_email="phillip1998@korea.ac.kr",
    description="Graph neural network model for prediction of experimental molecular property reflecting Solvatochromic subgroup contribution approach",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ACCA-KU/SSC",
    packages=find_packages(),
    include_package_data=True,
    package_data={"SSC": ["src/network_refer.yaml"]},
    install_requires=[
"numpy",
"pandas",
"scikit-learn",
"scipy",
"matplotlib",
"seaborn",
"torch>=2.2,<3",
"torch-geometric>=2.8,<3",
"rdkit>=2022",
"tqdm",
"joblib",
"pydantic",
"D4CMPP2>=1.0.1,<2",
"MoleculeSculptor>=1.0.1,<2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.10',
)
