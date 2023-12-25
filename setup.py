from setuptools import setup, find_packages

setup(
    name="st_water_seg",
    version="0.0.1",
    install_requires=[
        "requests", "toml",
        'importlib-metadata; python_version > "3.9"', "hydra-core",
        "opencv-python", "imagecodecs", "yapf", "pytorch_lightning"
    ],
    packages=find_packages(
        # All keyword arguments below are optional:
        where=".",  # '.' by default
        include=["st_water_seg"],  # ['*'] by default
        exclude=["st_water_seg.egg-info", "dist",
                 ".vscode"],  # empty by default
    ),
)
