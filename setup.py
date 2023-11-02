from setuptools import setup, find_packages

setup(
    name="manicast",        # The name of your package/project
    version="0.1",            # The version of your package
    url="https://github.com/portal-cornell/manicast",  # The URL of your project's homepage
    packages=find_packages(),  # Automatically find and include all packages in your project directory
    install_requires=open('requirements.txt').read().splitlines(),
)
