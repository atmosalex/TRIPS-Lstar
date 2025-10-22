from setuptools import setup, find_packages

setup(
    name='TRIPS',
    version='0.1.0',
    author='Alexander Lozinski',
    author_email='your_email@example.com',
    description='A short description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your_username/your_package_name',
    #packages=find_packages('src', exclude=['test']),
    #package_dir={"": "src"},
    packages=['TRIPS'],
    package_dir={'TRIPS': 'src/TRIPS'},
    package_data={'TRIPS': ['data/*']},
    python_requires='>=3.6',
    install_requires=[
        # List your package dependencies here, e.g.,
        # 'requests>=2.20.0',
        # 'numpy',
    ],
)