import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'drone_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='qntmdghkss',
    maintainer_email='qntmdghkss@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
           'precision_mission_controller = drone_control.precision_mission_controller:main',
           'precision_landing_controller = drone_control.precision_landing_controller:main',
           'precision_landing_test_input = drone_control.precision_landing_test_input:main',
           'mission_manager = drone_control.supervision:main',
           'setpoint_mux = drone_control.setpoint_mux:main',
           'vertiport_tracking = drone_control.vertiport_tracking:main',
           'generator_tracking = drone_control.generator_tracking:main',
         ],
    },
)
