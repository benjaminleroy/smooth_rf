from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='smooth_rf',
      version='0.1',
      description='Smoothing Random Forests',
      long_description = readme(),
      url='http://github.com/benjaminleroy/smooth_rf',
      author='benjaminleroy',
      author_email='bpleroy@stat.cmu.edu',
      license='MIT',
      packages=['smooth_rf'],
      install_requires=[
          'numpy', 'sparse', 'scipy', 'sklearn',
          'quadprog', "Cython", # used for smooth_level (remove in the future)
          'progressbar2', 'matplotlib' # this line is less "needed"
      ],
      test_suite='nose.collector',
      test_require=['nose'],
      zip_safe=False)
