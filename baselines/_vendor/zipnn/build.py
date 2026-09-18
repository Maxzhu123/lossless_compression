"""Build the vendored ZipNN extension in place; does not install packages."""
import os
from pathlib import Path
from setuptools import Extension, setup

os.chdir(Path(__file__).resolve().parent)
fse = "include/FiniteStateEntropy/lib/"
setup(
    name="vendored-zipnn-core",
    ext_modules=[Extension(
        "zipnn.zipnn_core",
        sources=["csrc/" + name + ".c" for name in (
            "zipnn_core_module", "zipnn_core", "data_manipulation_dtype16",
            "data_manipulation_dtype32",
        )] + [fse + name + ".c" for name in (
            "fse_compress", "fse_decompress", "huf_compress", "huf_decompress",
            "entropy_common", "hist",
        )],
        include_dirs=[fse, "csrc"],
        extra_compile_args=["-O3", "-pthread"],
        extra_link_args=["-pthread"],
    )],
)
