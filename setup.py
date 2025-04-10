from setuptools import setup, Extension
import numpy

setup(
    name="galpopc",
    version="0.1.0",
    author="Lado Samushia",
    description="High-performance HOD galaxy population model using OpenMP",
    packages=["galpopc"],
    ext_modules=[
        Extension(
            "galpopc.galcore",
            ["galpopc/galcore.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3", "-fopenmp"],
            extra_link_args=["-fopenmp"]
        )
    ],
    install_requires=["numpy", "scipy"],
)
