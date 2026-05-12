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
           'offboard_controller = drone_control.offboard_controller:main',
           'target_follow_controller = drone_control.target_follow_controller:main',
           'fake_target_publisher = drone_control.fake_target_publisher:main',
           'precision_mission_controller = drone_control.precision_mission_controller:main',
         ],
    },
)
