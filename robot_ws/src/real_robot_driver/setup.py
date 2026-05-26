from setuptools import find_packages, setup

package_name = 'real_robot_driver'

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
    maintainer='nvidia',
    maintainer_email='nvidia@todo.todo',
    description='Real robot drivers for odometry and lidar',
    license='TODO',
    entry_points={
        'console_scripts': [
            'odom_driver = real_robot_driver.odom_driver:main',
            'lidar_driver = real_robot_driver.lidar_driver:main',
        ],
    },
)
