from setuptools import setup, find_packages

setup(name='pymle-diffusion',
      version='0.0.9',
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
          'numba',
          'setuptools',
          'numpy',
          'scipy',
          'pandas',
      ],
      include_package_data=True,
      zip_safe=False)
