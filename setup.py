import os

import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

package_dir = {
    "distributed_learning_simulation": "./simulation_lib",
}

for dirname in os.listdir("./simulation_lib"):
    if os.path.isdir(os.path.join("./simulation_lib", dirname)):
        package_dir[
            f"distributed_learning_simulation.{dirname}"
        ] = f"./simulation_lib/{dirname}"


setuptools.setup(
    name="distributed_learning_simulation",
    author="cyy",
    version="0.1",
    author_email="cyyever@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cyyever/distributed_learning_simulation_lib",
    package_dir=package_dir,
    package_data={"distributed_learning_simulator.conf": ["*/*.yaml", "*.yaml"]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
