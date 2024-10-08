from setuptools import setup, find_packages

setup(
    name="ac-mpc",
    version="1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "scipy",
        "osqp",
        "pyconcorde",
        "segmentation-models-pytorch",
        "ruamel.yaml",
        "matplotlib",
        "simple-pid",
        "PyQt6",
    ],
)
