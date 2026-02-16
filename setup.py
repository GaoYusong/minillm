from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

setup(
    name="minillm",
    version="0.1.0",
    packages=find_packages(),
    rust_extensions=[
        RustExtension(
            "minillm.minillm_core",
            path="Cargo.toml",
            binding=Binding.PyO3,
        )
    ],
    install_requires=[
        "numpy>=1.24",
        "torch>=2.0",
        "safetensors>=0.4",
        "huggingface-hub>=0.19",
    ],
    python_requires=">=3.8",
    zip_safe=False,
)
