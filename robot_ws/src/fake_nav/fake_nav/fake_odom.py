import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class FakeOdom(Node):

    def __init__(self):
        super().__init__('fake_odom')

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(0.1, self.publish_odom)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def publish_odom(self):

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y

        self.odom_pub.publish(odom)

        t = TransformStamped()

        t.header.stamp = odom.header.stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"

        t.transform.translation.x = self.x
        t.transform.translation.y = self.y

        self.tf_broadcaster.sendTransform(t)


def main(args=None):

    rclpy.init(args=args)

    node = FakeOdom()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
