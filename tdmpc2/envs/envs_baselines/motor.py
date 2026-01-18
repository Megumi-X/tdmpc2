import os
import sys
sys.path.append(os.getcwd())

import numpy as np

class MotorModel2208:
    def __init__(self):
        self.R = 14.4
        self.KV = 100
        self.Kt = 9.55 / self.KV
        self.Ke = self.Kt
        
        self.V_max = 12.0
        self.Torque_limit = 0.03

        self.damping_coeff = self.Ke * self.Kt / self.R

    def apply_limit(self, desired_torque, current_omega):
        desired_current = desired_torque / self.Kt
        back_emf = current_omega * self.Ke
        required_voltage = (desired_current * self.R) + back_emf
        actual_voltage = np.clip(required_voltage, -self.V_max, self.V_max)
        actual_current = (actual_voltage - back_emf) / self.R
        actual_torque = actual_current * self.Kt
        final_torque = np.clip(actual_torque, -self.Torque_limit, self.Torque_limit)
        
        return final_torque