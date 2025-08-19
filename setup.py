
import setuptools

setuptools.setup(
    name="meteolibre_model",
    version="0.0.1",
    author="Meteo-Libre",
    author_email="author@example.com",
    description="Meteo-Libre Model",
    long_description="Meteo-Libre Model",
    long_description_content_type="text/markdown",
    url="httpshttps://github.com/meteo-libre/meteolibre-model",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "h5py",
        "pyproj",
        "pandas",
        "matplotlib",
        "pyarrow",
        "datasets==4.0",
        "huggingface_hub[cli]",
        "timm",
        "imageio",
        "diffusers",
        "einops",
        "heavyball",
        "safetensors",
        "dit_ml",
        "tensorboard",
    ]
)
