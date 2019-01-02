from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='smoothrf',
      version='0.1',
      description='Smoothing Random Forests',
      long_description = readme(),
      url='http://github.com/benjaminleroy/smoothRF',
      author='benjaminleroy',
      author_email='bpleroy@stat.cmu.edu',
      license='MIT',
      packages=['smoothRF'],
      install_requires=[
          'numpy', 'sparse', 'scipy'
      ],
      test_suite='nose.collector',
      test_require=['nose'],
      zip_safe=False)
