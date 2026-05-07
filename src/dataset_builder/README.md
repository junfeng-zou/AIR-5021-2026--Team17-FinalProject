# DOBOT CR5 Pouring Dataset

DOBOT CR5 robot arm performing water pouring demonstrations.
95 filtered episodes from real robot teleoperation data.

## Actions
- 7-DoF end-effector delta actions: [dx, dy, dz, droll, dpitch, dyaw, gripper]
- Gripper: 1 = open, 0 = close

## Observations
- 224x224 RGB images from fixed camera
- 7-dim state: [EE_x, EE_y, EE_z, EE_rx, EE_ry, EE_rz, gripper_state]
