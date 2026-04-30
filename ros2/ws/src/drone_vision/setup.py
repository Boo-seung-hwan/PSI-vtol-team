from setuptools import setup

package_name = 'drone_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='qntmdghkss',
    maintainer_email='qntmdghkss@todo.todo',
    description='Drone vision package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_detector = drone_vision.yolo_detector:main',
        ],
    },
)
