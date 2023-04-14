from setuptools import setup, find_packages

setup(name='pymle-diffusion',
      version='0.0.2',
      description='Maximum Likelihood Estimation (MLE) and simulation for SDE',
      long_description='Maximum Likelihood Estimation (MLE) and simulation for '
                       'Stochastic Differential Equations (SDE)',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering ',
          "Operating System :: OS Independent",
      ],
      keywords='sde mle maximum likelihood difussion estimation simulation',
      url='https://github.com/jkirkby3/pymle',
      author='Justin Lars Kirkby',
      author_email='jkirkby33@gmail.com',
      license='MIT',
      packages=find_packages(),
      python_requires=">=3.7",
      install_requires=[
          'numba~=0.53.1',
          'seaborn~=0.11.1',
          'setuptools~=56.0.0',
          'numpy~=1.20.2',
          'scipy~=1.6.2',
          'pandas~=1.2.3',
          'matplotlib~=3.4.1'
      ],
      include_package_data=True,
      zip_safe=False)
