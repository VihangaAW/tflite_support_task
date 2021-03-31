from setuptools import find_packages, setup
setup(
    name='tflite-support-task',
    packages=find_packages(include=['tflite_support_task']),
    version='0.1.0',
    description='TensorFlow Lite Task Library for Python',
    author='Vihanga Ashinsana',
    # license='MIT',
    install_requires=[],
    include_package_data=True,
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)