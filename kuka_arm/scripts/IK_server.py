#!/usr/bin/env python

# Copyright (C) 2017 Electric Movement Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya

# import modules
import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
from sympy import *
import math


def find_wrist_center(px, py, pz, roll, pitch, yaw):
    end_effector_length = 0.303
    wx = px - (end_effector_length) * 1.0 * cos(yaw) * cos(pitch)
    wy = py - (end_effector_length) * 1.0 * sin(yaw) * cos(pitch)
    wz = pz - (end_effector_length) * -1.0 * sin(pitch)
    return wx, wy, wz

def get_R0_6(alpha, beta, gamma):
    r11 = cos(alpha) * cos(beta)
    r12 = (cos(alpha)* sin(beta) * sin(gamma)) - (sin(alpha)*cos(gamma))
    r13 = (cos(alpha) * sin(beta) * cos(gamma)) + (sin(alpha)*sin(gamma))
    r21 = sin(alpha) * cos(beta) * 1.0
    r22 = (sin(alpha) * sin(beta) * sin(gamma)) + (cos(alpha) * cos(gamma))
    r23 = (sin(alpha)*sin(beta)*cos(gamma)) - (cos(alpha)*sin(gamma))
    r31 = -sin(beta)
    r32 = cos(beta) * sin(gamma)
    r33 = cos(beta)*cos(gamma)
    return Matrix([[r11, r12, r13],
                   [r21, r22, r23],
                   [r31, r32, r33]])

def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:
        # Initialize service response
        joint_trajectory_list = []
        for x in xrange(0, len(req.poses)):
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()

            # Define DH param symbols
            q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8') # theta_1
            d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8') # d paramater
            a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
            alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')


            # Joint angle symbols
            theta1, theta2, theta3, theta4, theta5, theta6 = symbols('theta1:7')


            # Modified DH params
            dh = {alpha0:     0, a0:      0, d1:  0.75,
                  alpha1: -pi/2, a1:   0.35, d2:     0, q2: q2 - pi/2,
                  alpha2:     0, a2:   1.25, d3:     0,
                  alpha3: -pi/2, a3: -0.054, d4:  1.50,
                  alpha4:  pi/2, a4:      0, d5:     0,
                  alpha5: -pi/2, a5:      0, d6:     0,
                  alpha6:     0, a6:      0, d7: 0.303, q7: 0}

            # General rotations around axes

            R_z = Matrix([[       cos(q1),    -sin(q1),             0, 0],
                          [       sin(q1),     cos(q1),             0, 0],
                          [             0,           0,             1, 0],
                          [             0,           0,             0, 1]])
            R_y = Matrix([[       cos(q1),           0,       sin(q1), 0],
                          [             0,           1,             0, 0],
                          [      -sin(q1),           0,       cos(q1), 0],
                          [             0,           0,             0, 1]])
            R_x = Matrix([[             1,           0,             0, 0],
                          [             0,     cos(q1),      -sin(q1), 0],
                          [             0,     sin(q1),       cos(q1), 0],
                          [             0,           0,             0, 1]])

            # Define Modified DH Transformation matrix
            R_corr = simplify(R_z.evalf(subs={q1: pi}) * R_y.evalf(subs={q1: -pi/2}))


            # Create individual transformation matrices
            T0_1 = Matrix([[            cos(q1),            -sin(q1),            0,              a0],
               [sin(q1)*cos(alpha0), cos(q1)*cos(alpha0), -sin(alpha0), -sin(alpha0)*d1],
               [sin(q1)*sin(alpha0), cos(q1)*sin(alpha0),  cos(alpha0),  cos(alpha0)*d1],
               [                  0,                   0,            0,               1]])
            T0_1 = T0_1.subs(dh)

            T1_2 = Matrix([[            cos(q2),            -sin(q2),            0,              a1],
                           [sin(q2)*cos(alpha1), cos(q2)*cos(alpha1), -sin(alpha1), -sin(alpha1)*d2],
                           [sin(q2)*sin(alpha1), cos(q2)*sin(alpha1),  cos(alpha1),  cos(alpha1)*d2],
                           [                  0,                   0,            0,               1]])
            T1_2 = T1_2.subs(dh)


            T2_3 = Matrix([[            cos(q3),            -sin(q3),            0,              a2],
                           [sin(q3)*cos(alpha2), cos(q3)*cos(alpha2), -sin(alpha2), -sin(alpha2)*d3],
                           [sin(q3)*sin(alpha2), cos(q3)*sin(alpha2),  cos(alpha2),  cos(alpha2)*d3],
                           [                  0,                   0,            0,               1]])
            T2_3 = T2_3.subs(dh)


            # Extract end-effector position and orientation from request
    	    # px,py,pz = end-effector position
    	    # roll, pitch, yaw = end-effector orientation
            px = req.poses[x].position.x
            py = req.poses[x].position.y
            pz = req.poses[x].position.z

            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                    req.poses[x].orientation.z, req.poses[x].orientation.w])

            # Calculate joint angles using Geometric IK method

            wx, wy, wz = find_wrist_center(px, py, pz, roll, pitch, yaw)

            theta1 = atan2(wy, wx)

            s1 = sqrt(wx*wx + wy*wy) - dh[a1]
            s2 = wz - dh[d1]
            s3 = sqrt(s1*s1 + s2*s2)
            s4 = sqrt(dh[a3]*dh[a3] + dh[d4]*dh[d4])

            wx0, wy0, wz0 = 1.85, 0, 1.947
            s1_initial = sqrt(wx0*wx0 + wy0*wy0) - dh[a1]
            s2_initial = wz0 - dh[d1]
            s3_initial = sqrt(s1_initial*s1_initial + s2_initial*s2_initial)
            s4_initial = s4

            beta1_initial = atan2(s2_initial, s1_initial)
            beta2_d_initial = (dh[a2]*dh[a2] + s3_initial*s3_initial - s4_initial*s4_initial) / (2.0*dh[a2]*s3_initial)
            beta2_initial = atan2(sqrt(1 - beta2_d_initial*beta2_d_initial), beta2_d_initial)

            beta1 = atan2(s2, s1)
            beta2_d = (dh[a2]*dh[a2] + s3*s3 - s4*s4) / (2*dh[a2]*s3)
            beta2 = atan2(sqrt(1 - beta2_d*beta2_d), beta2_d)

            print s1, s2, s3, s4
            print(beta1)
            print(beta2)

            # theta2
            print(beta1_initial, beta2_initial)
            print(beta1, beta2)
            theta2 = (beta1_initial + beta2_initial) - (beta1 + beta2)

            # Finding theta3
            # beta3_intiial
            beta4 = pi/2 - atan2(dh[a3], dh[d4])
            beta3_initial_d = (dh[a2]*dh[a2] + s4_initial*s4_initial - s3_initial*s3_initial) / (2.0*dh[a2]*s4_initial)
            beta3_initial = atan2(sqrt(1 - beta3_initial_d*beta3_initial_d), beta3_initial_d)
            # beta3_initial = 1.5348118667

            # beta3
            beta3_d = (dh[a2]*dh[a2] + s4*s4 - s3*s3) / (2.0*dh[a2]*s4)
            beta3 = atan2(sqrt(1 - beta3_d*beta3_d), beta3_d)

            # theta3
            theta3 = beta3_initial - beta3

            # Get R0_6/Rrpy
            R0_6 = get_R0_6(roll, pitch, yaw)
            R0_6 = R0_6.row_join(Matrix([[px],
                                         [py],
                                         [pz]]))
            R0_6 = R0_6.col_join(Matrix([[0, 0, 0, 1]]))
            R0_6 = R0_6

            # Get R0_3
            R0_3 = (T0_1 * T1_2 * T2_3).evalf(subs={q1: theta1, q2: theta2, q3: theta3})

            # Get R3_6
            R3_6 = Transpose(R0_3) * R0_6
            R3_6 = R3_6 * R_corr.inv() * R_x.evalf(subs={q1: pi/2})

            R3_6_converted = matrix2numpy(R3_6)

            theta4, theta5, theta6 = tf.transformations.euler_from_matrix(R3_6_converted, 'rzyz')
            theta4 += pi
            if theta4 > 2*pi:
                theta4 -= 2*pi
            elif theta4 < -2*pi:
                theta4 += 2*pi

            theta5 += pi/2
            if theta5 > 2*pi:
                theta5 -= 2*pi
            elif theta5 < -2*pi:
                theta5 += 2*pi
            # print theta4, theta5, theta6

            # r23 = R3_6[1, 2]
            # r33 = R3_6[2, 2]
            # theta4 = -atan2(r33, r23)
            #
            # theta5 = acos(r23)
            #
            # r21 = R3_6[1, 0]
            # r22 = R3_6[1, 1]
            # theta6 = atan2(r22, r21)
            # if math.isnan(theta4):
            #     theta4 = 0
            # if math.isnan(theta5):
            #     theta5 = 0
            # if math.isnan(theta6):
            #     theta6 = 0
            # print theta4, theta5, theta6

            # Populate response for the IK request
            # In the next line replace theta1,theta2...,theta6 by your joint angle variables
    	    joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
    	    joint_trajectory_list.append(joint_trajectory_point)

        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()

if __name__ == "__main__":
    IK_server()
