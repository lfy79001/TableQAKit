from pathlib import Path
from setuptools import find_packages, setup


project_root = Path(__file__).parent
install_requires = [
    "Flask>=2.2.2",
    "datasets>=2.9.0",
    "requests",
    "lxml",
    "tinyhtml",
    "xlsxwriter",
    "coloredlogs",
]

setup(
    name="ttqakit",
    version="0.0.1",
    python_requires=">=3.8",
    description="TTQAKit: A toolkit for Text-Table Hybrid Question Answering.",
    author="Fangyu Lei",
    author_email="leifangyu@qq.com",
    long_description=(project_root / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/lfy79001/TextTableQAKit",
    license="Apache-2.0 License",
    packages=find_packages(exclude=["test", "test.*"]),
    # package_dir={"": "src"},
    # package_data={
    #     "tabgenie": ["static/css/*", "static/img/*", "static/js/*", "templates/*"],
    # },
    # data_files=[("tabgenie", ["tabgenie/config.yml"])],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "tabgenie=tabgenie.cli:run",
        ],
        "flask.commands": ["export=tabgenie.cli:export", "sheet=tabgenie.cli:sheet", "info=tabgenie.cli:info"],
    },
    install_requires=install_requires,
    extras_require={
        "dev": [
            "wheel",
            "black",
        ],
        "deploy": [
            "gunicorn",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
)