from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README for long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="scid-processor",
    version="1.0.0",
    description="SCID (Screen Content Image Database) Processing Module",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/scid-processor",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            'scid-process=SCID.process_scid:main',
            'scid-convert=SCID.convert_to_jpg:main',
            'scid-attack=SCID.ImageAttacker:main',
            'scid-gnd=SCID.convert_gnd:main',
        ],
    },
)
