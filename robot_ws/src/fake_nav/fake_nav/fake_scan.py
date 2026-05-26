import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

class FakeScan(Node):
    def __init__(self):
        super().__init__('fake_scan')
        self.publisher_ = self.create_publisher(LaserScan, 'scan', 10)
        self.timer = self.create_timer(0.1, self.publish_scan)  # 10 Hz

    def publish_scan(self):
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_scan'  # 雷达挂在 base_link 上
        msg.angle_min = -math.pi/2
        msg.angle_max = math.pi/2
        msg.angle_increment = math.pi/180  # 1° resolution
        msg.time_increment = 0.0
        msg.scan_time = 0.1
        msg.range_min = 0.12
        msg.range_max = 3.5
        # 模拟简单前方无障碍
        msg.ranges = [3.0] * int((msg.angle_max - msg.angle_min)/msg.angle_increment)
        msg.intensities = [0.0] * len(msg.ranges)
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FakeScan()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
