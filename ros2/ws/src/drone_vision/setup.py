import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'drone_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='qntmdghkss',
    maintainer_email='qntmdghkss@todo.todo',
    description='Drone vision package',
    license='TODO',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'yolo_detector = drone_vision.yolo_detector:main',
            'target_depth_estimator = drone_vision.target_depth_estimator:main',
            'target_camera_to_ned = drone_vision.target_camera_to_ned:main',
        ],
    },
)