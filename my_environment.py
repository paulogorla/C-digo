from math import sqrt
import numpy as np
import pandas as pd

TIME_STEP = 3

# Define car parameters to estimate the consumption as reward
b0 = 0.1569
b1 = 2.450*pow(10,-2)
b2 = -7.415*pow(10,-4)
b3 = 5.975*pow(10,-5)
c0 = 0.07224
c1 = 9.681*pow(10,-2)
c2 = 1.075*pow(10,-3)

# Define your custom environment
class RouteEnvironment:
    def __init__(self, data):
        self.original_data = data
        self.data = data
        self.current_step = 0
        self.total_steps = len(data)
        self.states_parameters = 7
        self.actions_qty = 5
        self.penalty = 0
        
    def reset(self):
        #print(self.data)
        self.current_step = 0
        self.data = self.original_data
        self.total_steps = len(self.data)
        self.penalty = 0
        initial_state = self.data.iloc[self.current_step][['total_distance_traveled_normalized','distance_remaining_normalized','altitude_normalized','diff_to_next_100m_altitude_normalized','current_speed_normalized','max_speed_normalized','recomm_speed_normalized']]
        return initial_state
    
    def calculate_acceleration_to_slope(self,altitude_diff,distance=100):
        return 9.80665*altitude_diff/sqrt(pow(distance,2)+pow(altitude_diff,2)) if pow(distance,2)+pow(altitude_diff,2) > 0 else 0

    def calculate_distance_traveled(self,speed,acceleration,time=TIME_STEP):
        return speed*time+acceleration*pow(time,2)/2
    
    def drive_distance(self,distance_traveled,df,current_step,next_step):
        aux_df = pd.DataFrame()
        aux_df = df[(df['total_distance_traveled']>=distance_traveled) | (df.index == len(df)-1)]
        df = df.iloc[:next_step]
        df = pd.concat([df,aux_df])
        df = df.reset_index(drop=True)
        return df
    
    def step(self, action):
        #set next step index
        next_step = self.current_step + 1
        
        #check if actual step is the last or it finished the route
        if next_step < self.total_steps:
            #gets state parameters: total_distance_traveled, distance_remaining, altitude, diff_to_next_100m_altitude, current_speed, max_speed, recomm_speed
            total_distance_traveled = self.data.iloc[self.current_step]['total_distance_traveled']
            distance_remaining = self.data.iloc[self.current_step]['distance_remaining']
            altitude = self.data.iloc[self.current_step]['altitude']
            diff_to_next_100m_altitude = self.data.iloc[self.current_step]['diff_to_next_100m_altitude']
            speed = self.data.iloc[self.current_step]['current_speed']
            max_speed = self.data.iloc[self.current_step]['max_speed']
            min_speed = self.data.iloc[self.current_step]['recomm_speed']

            #define acceleration based on action chosen
            acceleration = (
                -3 if action == 0
                else -1 if action == 1
                else 0 if action == 2
                else 1 if action == 3
                else 3
                )
            
            #calculates acceleration needed to surpass the slope based on the altitude difference
            acceleration_to_slope = self.calculate_acceleration_to_slope(diff_to_next_100m_altitude)
            #define real acceleration
            real_acceleration = acceleration-acceleration_to_slope
    
            if speed == 0 and acceleration < 0:
                #if it is idleing and deaccelerate stop the environment and give negative reward
                reward = -10
                state = self.data.iloc[self.current_step][['total_distance_traveled_normalized','distance_remaining_normalized','altitude_normalized','diff_to_next_100m_altitude_normalized','current_speed_normalized','max_speed_normalized','recomm_speed_normalized']]
                next_state = None
                done = True
            elif speed == 0 and acceleration == 0:
                #if idleing and keep, penalize and set the next state to the current state
                reward = -20
                state = self.data.iloc[self.current_step][['total_distance_traveled_normalized','distance_remaining_normalized','altitude_normalized','diff_to_next_100m_altitude_normalized','current_speed_normalized','max_speed_normalized','recomm_speed_normalized']]
                next_state = state
                done = False
            else:
                #if have speed or accelerate, or both then
                #calculates step_distance_traveled, driven_dataframe, update total steps and next speed
                distance_traveled_m = self.calculate_distance_traveled(speed,real_acceleration)
                distance_traveled_km = distance_traveled_m/1000
                total_traveled_distance = self.data.loc[self.current_step]['total_distance_traveled'] + distance_traveled_m
                self.data = self.drive_distance(total_traveled_distance,self.data,self.current_step,next_step)
                self.total_steps = len(self.data)
                next_speed = speed+real_acceleration*TIME_STEP

                if next_speed <= 0:
                    #if the chosen acceleration stops the car or gives negative speed then stop environment and give negative reward based on the speed difference
                    reward = next_speed - speed
                    state = self.data.iloc[self.current_step][['total_distance_traveled_normalized','distance_remaining_normalized','altitude_normalized','diff_to_next_100m_altitude_normalized','current_speed_normalized','max_speed_normalized','recomm_speed_normalized']]
                    next_state = None
                    done = True
                else:
                    #if the chosen acceleration keeps the car moving then
                    #if the current speed is lower than minimal speed and deaccelerate or keep then set penalty to the difference
                    if speed < min_speed and acceleration < 0:
                      self.penalty = min_speed - speed
                    #if the current speed is higher than maximum speed and accelerates or keep then set penalty to the difference
                    if speed > max_speed and acceleration > 0:
                        self.penalty = pow((speed - max_speed),2)
                    
                    #estimate consumption in mL/s with the formula and calculates in liters mL/s * s /1000
                    if acceleration >=0:
                        consumption_mls = (b0+b1*speed+b2*pow(speed,2)+b3*pow(speed,3)+acceleration*(c0+c1*speed+c2*pow(speed,2)))
                        consumption_l = consumption_mls*TIME_STEP/1000
                    else:
                        consumption_mls = (b0+b1*speed+b2*pow(speed,2)+b3*pow(speed,3))
                        consumption_l = consumption_mls*TIME_STEP/1000

                    #set reward as efficiency in km/l and deduct the penalties
                    reward = distance_traveled_km/consumption_l
                    reward -= self.penalty

                    #reset the penalties
                    self.penalty = 0

                    #set next speed based on the calculated speed and normalize it based on max speed permited
                    self.data.at[next_step,'current_speed'] = next_speed
                    self.data.at[next_step,'current_speed_normalized'] = next_speed/max_speed

                    #next total_distance_traveled, remaining_distance, altitude, diff_to_next_100m_altitude
                    #(norm and not) are already on the data after the drive_distance method

                    next_state = self.data.iloc[next_step][['total_distance_traveled_normalized','distance_remaining_normalized','altitude_normalized','diff_to_next_100m_altitude_normalized','current_speed_normalized','max_speed_normalized','recomm_speed_normalized']]
                    
                    #set state to pass
                    state = self.data.iloc[self.current_step][['total_distance_traveled_normalized','distance_remaining_normalized','altitude_normalized','diff_to_next_100m_altitude_normalized','current_speed_normalized','max_speed_normalized','recomm_speed_normalized']]
                    done = False
                    self.current_step = next_step
        else:
            state = self.data.iloc[self.current_step][['total_distance_traveled_normalized','distance_remaining_normalized','altitude_normalized','diff_to_next_100m_altitude_normalized','current_speed_normalized','max_speed_normalized','recomm_speed_normalized']]
            reward = 10000
            next_state = None
            done = True        
        return state, next_state, reward, done
