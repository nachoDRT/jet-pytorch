from setuptools import find_packages
from setuptools import setup


setup(
    name="jet-pytorch",
    version="0.0.2",
    packages=find_packages(),
    description='A Pytorch port of the code from "Jet: A Modern Transformer-Based Normalizing Flow"',
    author="Brandon Trude",
    author_email="brandon.trude@gmail.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "einops",
        "fire",
        "huggingface_hub",
        "matplotlib",
        "numpy",
        "opt-einsum",
        "safetensors",
        "torch",
        "torchvision",
        "tqdm",
    ],
    license="Apache 2.0 License",
    url="https://github.com/btrude/jet-pytorch",
)
