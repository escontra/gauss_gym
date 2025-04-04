from booster_robotics_sdk_python import LowCmd, LowCmdType, MotorCmd, B1JointCnt


def init_Cmd_T1(low_cmd: LowCmd):
    low_cmd.cmd_type = LowCmdType.SERIAL
    motorCmds = [MotorCmd() for _ in range(B1JointCnt)]
    low_cmd.motor_cmd = motorCmds

    for i in range(B1JointCnt):
        low_cmd.motor_cmd[i].q = 0.0
        low_cmd.motor_cmd[i].dq = 0.0
        low_cmd.motor_cmd[i].tau = 0.0
        low_cmd.motor_cmd[i].kp = 0.0
        low_cmd.motor_cmd[i].kd = 0.0
        # weight is not effective in custom mode
        low_cmd.motor_cmd[i].weight = 0.0


def create_prepare_cmd(low_cmd: LowCmd, cfg):
    init_Cmd_T1(low_cmd)
    for i in range(B1JointCnt):
        low_cmd.motor_cmd[i].kp = cfg["prepare"]["stiffness"][i]
        low_cmd.motor_cmd[i].kd = cfg["prepare"]["damping"][i]
        low_cmd.motor_cmd[i].q = cfg["prepare"]["default_qpos"][i]
    return low_cmd


def create_first_frame_rl_cmd(low_cmd: LowCmd, cfg):
    init_Cmd_T1(low_cmd)
    for i in range(B1JointCnt):
        low_cmd.motor_cmd[i].kp = cfg["common"]["stiffness"][i]
        low_cmd.motor_cmd[i].kd = cfg["common"]["damping"][i]
        low_cmd.motor_cmd[i].q = cfg["common"]["default_qpos"][i]

    return low_cmd
