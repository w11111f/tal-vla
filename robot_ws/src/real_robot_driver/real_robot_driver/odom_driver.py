#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster
import math

class OdomDriver(Node):
    """
    真实小车里程计驱动
    订阅 /cmd_vel (Nav2 输出)，控制真实电机
    发布 /odom 和 odom -> base_link 的 TF
    """
    def __init__(self):
        super().__init__('odom_driver')
       
        # 订阅 Nav2 发出的速度指令
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_callback, 10)
       
        # 发布里程计
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
       
        # TF 广播器
        self.tf_broadcaster = TransformBroadcaster(self)
       
        # 机器人位姿 (实际应从编码器读取)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
       
        # 当前速度
        self.vx = 0.0
        self.vw = 0.0
       
        # 定时器：20Hz 更新里程计
        self.timer = self.create_timer(0.05, self.timer_callback)
       
        self.last_time = self.get_clock().now()
       
        self.get_logger().info('真实小车里程计驱动已启动')
   
    def cmd_callback(self, msg: Twist):
        """接收 Nav2 的速度指令，控制真实电机"""
        self.vx = msg.linear.x
        self.vw = msg.angular.z
       
        # ============================================
        # TODO: 在这里添加你的电机控制代码
        # 例如：通过串口发送指令给 STM32/Arduino
        # self.serial_port.write(f"{self.vx},{self.vw}\n")
        # ============================================
       
    def timer_callback(self):
        """定时更新里程计"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
       
        if dt > 0.5:
            return
       
        # ============================================
        # TODO: 在这里读取真实的编码器数据
        # 这里用速度积分模拟，实际应从硬件读取
        # ============================================
        delta_x = self.vx * math.cos(self.theta) * dt
        delta_y = self.vx * math.sin(self.theta) * dt
        delta_theta = self.vw * dt
       
        self.x += delta_x
        self.y += delta_y
        self.theta += delta_theta
       
        # 发布里程计消息
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'  # 注意：与你的 URDF 一致
       
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation = self._yaw_to_quaternion(self.theta)
       
        odom.twist.twist.linear.x = self.vx
        odom.twist.twist.angular.z = self.vw
       
        self.odom_pub.publish(odom)
       
        # 发布 TF 变换
        tf_msg = TransformStamped()
        tf_msg.header.stamp = current_time.to_msg()
        tf_msg.header.frame_id = 'odom'
        tf_msg.child_frame_id = 'base_link'
        tf_msg.transform.translation.x = self.x
        tf_msg.transform.translation.y = self.y
        tf_msg.transform.rotation = self._yaw_to_quaternion(self.theta)
       
        self.tf_broadcaster.sendTransform(tf_msg)
   
    def _yaw_to_quaternion(self, yaw):
        q = Quaternion()
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

def main(args=None):
    rclpy.init(args=args)
    node = OdomDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
