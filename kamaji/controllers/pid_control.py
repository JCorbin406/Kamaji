"""
pid_control
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import sys
import numpy as np
from abc import ABC, abstractmethod
sys.path.append('..')

class PID_Parent(ABC):
    
    @abstractmethod
    def __init__(self, upper_limit, lower_limit):
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
            
    def saturate(self, input):
        if input <= self.lower_limit:
            output = self.lower_limit
        elif input >= self.upper_limit:
            output = self.upper_limit
        else:
            output = input
        return np.array(output)
        
class PControl(PID_Parent):
    # PD control with rate information
    # u = kp*(yref-y) - kd*ydot
    def __init__(self, kp, upper_limit, lower_limit):
        self.kp = kp
        super().__init__(upper_limit, lower_limit)
        

    def update(self, y_ref, y):
        u = self.kp * (y_ref - y)  
        # saturate PID control at limit
        u_sat = self.saturate(u)
        return u_sat

class PDControlWithRate(PID_Parent):
    # PD control with rate information
    # u = kp*(yref-y) - kd*ydot
    def __init__(self, kp, kd, upper_limit, lower_limit):
        self.kp = kp
        self.kd = kd
        super().__init__(upper_limit, lower_limit)

    def update(self, y_ref, y, ydot):
        u = self.kp * (y_ref - y)  - self.kd * ydot
        # saturate PID control at limit
        u_sat = self.saturate(u)
        return u_sat

class PIControl(PID_Parent):
    def __init__(self, kp, ki, Ts, sigma, upper_limit, lower_limit, anti_windup, anti_windup_limit):
        self.kp = kp
        self.ki = ki
        self.Ts = Ts
        self.integrator = 0.0
        self.error_delay_1 = 0.0
        self.anti_windup = anti_windup
        self.anti_windup_limit = anti_windup_limit
        self.a1 = (2.0 * sigma - Ts) / (2.0 * sigma + Ts)
        self.a2 = 2.0 / (2.0 * sigma + Ts)
        self.error_dot_delay_1 = 0.0
        super().__init__(upper_limit, lower_limit)
        
        if not (anti_windup == "none" or anti_windup == "error_dot" or anti_windup == "error"):
            raise Exception("invalid anti-windup type specified in vehicle params file")

    def update(self, y_ref, y):

        # compute the error
        error = y_ref - y
        
        # update the differentiator
        error_dot = self.a1 * self.error_dot_delay_1 \
                         + self.a2 * (error - self.error_delay_1)
        
        # update the integrator using trapazoidal rule
        if (self.anti_windup  == "none") \
                or (self.anti_windup == "error_dot" and abs(error_dot) < self.anti_windup_limit) \
                or (self.anti_windup == "error" and abs(error) < self.anti_windup_limit):
            self.integrator = self.integrator \
                                + (self.Ts/2) * (error + self.error_delay_1)
        # PI control
        u = self.kp * error \
            + self.ki * self.integrator
        # saturate PI control at limit
        u_sat = self.saturate(u)
        # integral anti-windup
        #   adjust integrator to keep u out of saturation
        if np.abs(self.ki) > 0.0001 and abs(self.integrator) > 0.0001:
            self.integrator = self.integrator \
                              + (self.Ts / self.ki) * (u_sat - u)
        # update the delayed variables
        self.error_delay_1 = error
        self.error_dot_delay_1 = error_dot
        return u_sat

class PIDControl(PID_Parent):
    def __init__(self, kp, ki, kd, Ts, sigma, upper_limit, lower_limit, anti_windup, anti_windup_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.Ts = Ts
        self.integrator = 0.0
        self.error_delay_1 = 0.0
        self.error_dot_delay_1 = 0.0
        self.y_dot = 0.0
        self.y_delay_1 = 0.0
        self.y_dot_delay_1 = 0.0
        # gains for differentiator
        self.a1 = (2.0 * sigma - Ts) / (2.0 * sigma + Ts)
        self.a2 = 2.0 / (2.0 * sigma + Ts)
        self.anti_windup = anti_windup # 0 = no AW, 1 = AW based on error_dot, 2 = AW based on error
        self.anti_windup_limit = anti_windup_limit
        super().__init__(upper_limit, lower_limit)
        
        if not (anti_windup == "none" or anti_windup == "error_dot" or anti_windup == "error"):
            raise Exception("invalid anti-windup type specified in vehicle params file")

    def update(self, y_ref, y, reset_flag=False):
        if reset_flag is True:
            self.integrator = 0.0
            self.error_delay_1 = 0.0
            self.y_dot = 0.0
            self.y_delay_1 = 0.0
            self.y_dot_delay_1 = 0.0
        # compute the error
        error = y_ref - y

        # error = np.linalg.norm(y_ref - y)
        # print(error)
        
        # update the differentiator
        error_dot = self.a1 * self.error_dot_delay_1 \
                         + self.a2 * (error - self.error_delay_1)
        
        # update the integrator using trapazoidal rule
        if (self.anti_windup  == "none") \
                or (self.anti_windup == "error_dot" and abs(error_dot) < self.anti_windup_limit) \
                or (self.anti_windup == "error" and abs(error) < self.anti_windup_limit):
            self.integrator = self.integrator \
                            + (self.Ts/2) * (error + self.error_delay_1)
        
        # PID control
        u = self.kp * error \
            + self.ki * self.integrator \
            + self.kd * error_dot
        # saturate PID control at limit
        u_sat = self.saturate(u)
        # print(f"Error: {error}")
        # print(f"Control: {u}")

        # integral anti-windup
        #   adjust integrator to keep u out of saturation
        if np.abs(self.ki) > 0.0001 and abs(self.integrator) > 0.0001:
            self.integrator = self.integrator \
                              + (self.Ts / self.ki) * (u_sat - u)
        # update the delayed variables
        self.error_delay_1 = error
        self.error_dot_delay_1 = error_dot
        return u_sat

    def update_with_rate(self, y_ref, y, ydot, reset_flag=False):
        if reset_flag is True:
            self.integrator = 0.0
            self.error_delay_1 = 0.0
            self.y_dot = 0.0
            self.y_delay_1 = 0.0
            self.y_dot_delay_1 = 0.0
        # compute the error
        error = y_ref - y
        
        error_dot = self.a1 * self.error_dot_delay_1 \
                         + self.a2 * (error - self.error_delay_1)
        
        # update the integrator using trapazoidal rule
        if (self.anti_windup  == "none") \
                or (self.anti_windup == "error_dot" and abs(error_dot) < self.anti_windup_limit) \
                or (self.anti_windup == "error" and abs(error) < self.anti_windup_limit):
            self.integrator = self.integrator \
                            + (self.Ts/2) * (error + self.error_delay_1)
        # PID control
        u = self.kp * error \
            + self.ki * self.integrator \
            - self.kd * ydot
        # saturate PID control at limit
        u_sat = self.saturate(u)
        # integral anti-windup
        #   adjust integrator to keep u out of saturation
        if np.abs(self.ki) > 0.0001 and abs(self.integrator) > 0.0001:
            self.integrator = self.integrator \
                              + (self.Ts / self.ki) * (u_sat - u)
        self.error_delay_1 = error
        self.error_dot_delay_1 = error_dot
        return u_sat
    