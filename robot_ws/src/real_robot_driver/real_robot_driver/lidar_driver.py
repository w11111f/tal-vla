#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

class LidarDriver(Node):
    """
    真实激光雷达驱动
    发布 /scan 话题，坐标系为 base_scan
    """
    def __init__(self):
        super().__init__('lidar_driver')
       
        # 发布激光扫描数据
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
       
        # 定时器：10Hz 发布数据
        self.timer = self.create_timer(0.1, self.timer_callback)
       
        self.get_logger().info('真实激光雷达驱动已启动')
   
    def timer_callback(self):
        # ============================================
        # TODO: 在这里读取真实的激光雷达数据
        # 例如：通过串口或 SDK 读取 RPLidar/YDLidar 数据
        # ============================================
       
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_scan'  # 与 URDF 中的激光雷达连杆一致
       
        # 激光雷达参数 (根据你的实际雷达修改)
        msg.angle_min = -math.pi / 2      # -90度
        msg.angle_max = math.pi / 2       # 90度
        msg.angle_increment = math.pi / 180  # 1度分辨率
        msg.time_increment = 0.0
        msg.scan_time = 0.1
        msg.range_min = 0.12
        msg.range_max = 12.0
       
        # 模拟数据，实际应从硬件读取
        num_readings = int((msg.angle_max - msg.angle_min) / msg.angle_increment)
        msg.ranges = [5.0] * num_readings
        msg.intensities = [0.0] * num_readings
       
        self.scan_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = LidarDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
