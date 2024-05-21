import gym
#from gym import spaces
from gym.utils import seeding
import numpy as np
import random
#import pygame
import math
from collections import deque
from gym.spaces import Dict, Discrete, Box
random.seed(10)
#I do not comment the self parameter because it's just an instance of the class (classic parameter)

class pedestrian:
    ped_rand, seed = seeding.np_random(7)
    def __init__(self, ped_b, car_b, cross, lines, limit_speed, dt,
                 simulation="unif", is_crossing=True, exist=True, number_name=-1):
        #self.seed(seed)
        # Env parameters
        self.worst_dl=0.0
        self.number_name=number_name
        self.cross=cross
        self.cross_lines=lines*cross
        self.max_lines=lines
        self.limit_speed=limit_speed
        self.dt=dt
        self.simulation=simulation
        
        self.ped_b=ped_b
        self.car_b=car_b
        self.decision=False
        self.at_crossing=False
        
        self.ped_left=False
        self.ped_in_cross=False
        
        self.accident=False
        self.possible_accident=False
        self.possible_accident2= False
        self.error_scenario=False
        self.is_crossing=is_crossing
        self.remove=not self.is_crossing
        self.time_to_remove = random.randint(0,20)
        self.time_before_crossing=0.
        self.waiting_time=0.
        self.crossing_time=0.
        self.worst_scenario_accident=0
        self.follow_rule = random.randint(0,9)<3 #30% de chance que le pieton suive les indications du vehicule (help converging)
        self.exist=exist
        
        # Initial pedestrian speed
        self.Vm=2.5
        self.tau = 1.0
        self.direction =  2*random.randint(0,1)-1
        self.line_pos=self.max_lines*(self.direction<0) -1*(self.direction>0)
        self.initial_speed = [random.uniform(ped_b[0,0], ped_b[1,0]),
                              random.uniform(ped_b[0,1], ped_b[1,1]) * self.direction]

        # Initial pedestrian position
        self.initial_pos = [random.uniform(ped_b[0,2], ped_b[1,2]),
                            (random.uniform(ped_b[0,3], ped_b[1,3])-self.cross_lines/2.)* self.direction]
        self.ratio=0.0
        if(not self.exist):
            self.initial_speed = [0.,0.]
            self.initial_pos = [ped_b[0,2],ped_b[0,3]* self.direction]
        elif(not self.is_crossing):
            self.initial_speed = [0.,0.]
            self.direction=0
        else:
            self.ratio=self.initial_speed[0]/self.initial_speed[1]
        self.Vp_x, self.Vp_y = self.initial_speed
        self.Sp_x, self.Sp_y = self.initial_pos
        
        self.worst_pos_p = self.Sp_y
        if(self.Sp_y*self.direction <= (-self.cross_lines/2)):
            self.worst_pos_p = self.worst_pos_p + self.dt*self.ped_b[1,1]
            
        self.time_stop = 0
        self.stop=False
        self.t0 = 0.0
        self.gender = random.randint(0,1)
        self.age = random.randint(0,2)
        self.CG = self.CG_score(self.cross) 
        self.change_line=False
        self.delta=0.0
        
        if self.simulation == "sin" and self.is_crossing:
            self.t_init = 0.0
            abs_speed=abs(self.initial_speed[1])
            self.T = self.cross_lines / (abs_speed+10e-3)
            check = ((abs_speed * math.pi) / 2.0 <= self.Vm)
            self.A = check*math.pi*abs_speed/2.0 + (not check)*(self.Vm-abs_speed) / (1.0-(2.0/math.pi))
            self.B = (not check)*(self.Vm - self.A)
            self.w = math.pi / self.T
            self.function_step=self.new_pedestrian_sin_y
        else:
            self.function_step=self.new_pedestrian_unif_y

    def reset_ped(self, speed_x, speed_y, pos_x, pos_y, dl, leave, CZ, exist, direction):
        self.initial_speed = [speed_x,speed_y]
        self.initial_pos =[ pos_x, pos_y]
        self.Vp_x, self.Vp_y = self.initial_speed
        self.Sp_x, self.Sp_y = self.initial_pos
        self.ratio=self.initial_speed[0]/(self.initial_speed[1]+1e-3)
        self.worst_pos_p = self.Sp_y
        if(self.Sp_y*self.direction <= (-self.cross_lines/2)):
            self.worst_pos_p = self.worst_pos_p + self.dt*self.ped_b[1,1]
        self.direction = direction
        self.line_pos=self.max_lines*(self.direction<0) -1*(self.direction>0)
        self.delta=dl
        self.exist=exist
        self.worst_pos_p = self.Sp_y
        if(self.Sp_y*self.direction <= (-self.cross_lines/2)):
            self.worst_pos_p = self.worst_pos_p + self.dt*self.ped_b[1,1]
        self.ped_left=leave
        self.ped_in_cross=CZ
        self.is_crossing=True
        self.CG = self.CG_score(self.cross_lines)  
        if self.simulation == "sin" and self.is_crossing:
            self.t_init = 0.0
            abs_speed=abs(self.initial_speed[1])
            self.T = self.cross_lines / (abs_speed+10e-3)
            check = ((abs_speed * math.pi) / 2.0 <= self.Vm)
            self.A = check*math.pi*abs_speed/2.0 + (not check)*(self.Vm-abs_speed) / (1.0-(2.0/math.pi))
            self.B = (not check)*(self.Vm - self.A)
            self.w = math.pi / self.T
            self.function_step=self.new_pedestrian_sin_y
        else:
            self.function_step=self.new_pedestrian_unif_y
            
            
    def choix_pedestrian(self, cars_speed, cars_pos, cars_line, cars_light, car_size=4):
        car_time=0
        if(self.follow_rule):
            cars=[i for i in range(len(cars_speed))]
            if(len(cars_speed)>1):
                random.shuffle(cars)
            for i in cars: # test if cars in crosswalk in the first lane
                if(self.is_crossing_in_front(cars_line[i],prev_line=0.5)*(self.is_in_front(cars_line[i],next_line=1.0))):
                    if (cars_pos[i] < car_size+self.Sp_x) * (cars_pos[i] > self.Sp_x):
                        return False
            for i in cars: # then decide according to a car in the road (this time not only the first lane)
                if (cars_pos[i] < self.Sp_x and cars_light[i]!=0):
                    return (cars_light[i]>0.)
        for i in range(len(cars_speed)):
            if(self.is_in_front(cars_line[i],next_line=1.0)):
                if (cars_pos[i] < car_size+self.Sp_x) * (cars_pos[i] > self.Sp_x):
                    return False
                if (cars_pos[i] < self.Sp_x):
                    car_time = abs((cars_pos[i]-self.Sp_x) / (cars_speed[i]+10e-3)) 
                    # ne prend pas en compte le fait d aller vers le vehicule
                    CG=self.CG_score(abs(self.line_pos-cars_line[i])*self.cross) 
                    if (car_time+cars_light[i] < CG):
                        return False
        return True
    
    def detection(self, cars,prev_cars_pos):
        car_dangers=[]
        car_Ts=[]
        for i in range(len(prev_cars_pos)):
            if(self.is_in_front(cars[i].line)):
                ped_accident=(not self.accident)*(self.worst_scenario_accident)
                self.worst_scenario_accident = 1 if self.worst_delta_l(cars[i].Sc, cars[i].Vc,cars[i].line)<0 else 0
                if ped_accident*(self.is_crossing_in_front(cars[i].line)*(prev_cars_pos[i]<self.Sp_x)*(cars[i].Sc>self.Sp_x)):
                    self.accident=True
                    print("Accident! : ", self.Sp_x)
                if (self.is_crossing_in_front(cars[i].line)):
                    if((cars[i].Vc)<0.05):
                        dl=0.
                    else:
                        dl=self.worst_delta_l(cars[i].Sc, cars[i].Vc,cars[i].line)/(cars[i].Vc)
                    possible_accident=0
                    if(dl>0):
                        possible_accident = -1.*math.exp(-4.*(dl))
                    else:
                        possible_accident = -1.*dl-1
                    if(possible_accident<-1. and cars[i].possible_accident>=-1.):
                        print("Possible accident! ")
                    cars[i].possible_accident=min(cars[i].possible_accident,possible_accident)

                if(cars[i].Sc<self.Sp_x):
                    if(cars[i].Ts==-10 and self.waiting_time+2.*self.crossing_time-cars[i].time_braking+1.>-10):
                        print(cars[i].time_braking)
                    cars[i].Ts=max(self.waiting_time+2.*self.crossing_time-cars[i].time_braking+1., cars[i].Ts)
                
                if(cars[i].light<0.0):
                    if(cars[i].Ts<0):
                        new_error=-1.*math.exp(4.*(cars[i].Ts))
                    else:
                        new_error=-1.*(1+cars[i].Ts)
                    if(cars[i].error_scenario>=-1. and new_error<-1.):
                        print("Small mistake - priority ? ")
                    if(self.is_crossing_in_front(cars[i].line)*(cars[i].Sc<self.Sp_x)):
                        print("Pedestrian is not waiting ")
                    cars[i].error_scenario = min(new_error, cars[i].error_scenario)
                
                if(cars[i].light>0.0):
                    if(self.Sp_x-cars[i].Sc>0):
                        new_error=-1.*math.exp(-4.*(self.Sp_x-cars[i].Sc))
                    else:
                        new_error=-1.*(1+cars[i].Sc-self.Sp_x)
                    if(cars[i].error_scenario>=-1. and new_error<-1.):
                        print("Mauvais signal vert ")
                    cars[i].error_scenario = min(new_error, cars[i].error_scenario)

        cars_light_green=sum([1.0 for car in cars if car.light>0.])
        cars_possible_accident=sum([car.possible_accident for car in cars])
        cars_possible_accident2=sum([car.possible_accident2 for car in cars])
        for i in range(len(prev_cars_pos)):
            cars_error_scenario=sum([car.error_scenario for car in cars if car.light<0. and car.Ts>0 and car.line!=cars[i].line])
            res_accident = cars[i].possible_accident 
            res_consistent = cars[i].error_scenario 
            res = res_accident+res_consistent
            car_dangers.append(res)
        return np.array(car_dangers)
    
    def boolean_ped_position(self):
        if(self.direction*self.Sp_y>= self.cross_lines/2):
            self.ped_in_cross=False
            self.ped_left=True
        elif(self.direction*self.Sp_y > -self.cross_lines/2):
            self.ped_in_cross=True
            self.ped_left=False
        else:
            self.ped_in_cross=False
            self.ped_left=False
        if(self.ped_left):
            self.time_to_remove=self.time_to_remove-1
            
    def will_change_line(self, pos, new_pos):
        if(abs(new_pos) < self.cross_lines/2):
            new_line = (new_pos + self.cross_lines/2)//self.cross
            if(new_line!=self.line_pos):
                if(abs(pos) < self.cross_lines/2):
                    self.change_line=True

    def apply_change_line(self, pos, new_pos):
        if(abs(new_pos) >= self.cross_lines/2):
            self.line_pos = self.max_lines*(self.direction<0) -1*(self.direction>0)
        else:
            new_line = (new_pos + self.cross_lines/2)//self.cross
            if(new_line!=self.line_pos):
                self.line_pos = new_line
        
            
    def step(self, time, cars_speed, cars_pos, cars_line, car_light):
        pp_y, _ = self.new_pedestrian_unif_y(time)
        self.boolean_ped_position()

        #worst pedestrian position
        self.worst_pos_p = self.Sp_y
        if(self.Sp_y*self.direction <= (-self.cross_lines/2)):
            self.worst_pos_p = self.worst_pos_p + self.dt*self.ped_b[1,1]*self.direction
            self.time_before_crossing=self.time_before_crossing+self.dt
        if(self.is_crossing):
            self.choose = True
            # Pedestrian choice
            if (not self.decision)*(self.at_crossing): 
                self.choose = self.choix_pedestrian(cars_speed, cars_pos, cars_line, car_light)
                if(self.choose):
                    self.line_pos = (self.max_lines-1)*(self.direction<0) 
                    self.at_crossing=False
                self.decision=True 
                self.t0 = time
            # Pedestrian arrives at the crosswalk
            if (self.Sp_y*self.direction < -self.cross_lines/2.) * (pp_y*self.direction > -self.cross_lines/2.) * (not self.decision):
                #Thales theorem
                pos_p_x = (self.Vp_x*self.dt)*(abs(-self.cross_lines/2.-self.Sp_y*self.direction)/abs(self.Vp_y*self.dt +10e-3))
                self.Vp_x = pos_p_x / self.dt
                self.Sp_x = self.Sp_x + pos_p_x
                self.Vp_y = self.direction*abs(-self.Sp_y*self.direction - (self.cross_lines/2.)) / self.dt
                self.Sp_y = - self.direction*self.cross_lines/2.
                self.time_stop = 0
                self.at_crossing=True

            # Pedestrian in crosswalk
            elif ((abs(self.Sp_y) <= self.cross_lines/2) + self.decision):
                # The pedestrain waits
                if (self.time_stop != 0):
                    self.Sp_x, self.Vp_x = self.Sp_x, 0.0
                    self.Sp_y, self.Vp_y = self.Sp_y, 0.0
                    self.time_stop = self.time_stop - 1
                    self.t0 = self.t0 + self.dt
                elif (random.uniform(0, 1) < 0.98) * self.choose :
                    self.decision=False 
                    new_spy, new_vpy = self.function_step(time)
                    self.will_change_line(self.Sp_y, new_spy)
                    distance_to_cross=(self.max_lines-self.line_pos-1)*self.cross*(self.direction>0) 
                    distance_to_cross += (self.line_pos)*self.cross* (self.direction<0)
                    if(self.change_line)and(distance_to_cross > 0. and distance_to_cross<self.cross_lines):
                        new_choice = self.choix_pedestrian(cars_speed, cars_pos, cars_line,car_light) #self.cross
                        if(new_choice and self.stop):
                            self.stop=False
                    else:
                        new_choice=False
                    
                    if(self.stop):
                        self.Sp_x, self.Vp_x = self.Sp_x, 0.0
                        self.Sp_y, self.Vp_y = self.Sp_y, 0.0
                        self.t0 = self.t0 + self.dt
                        if(self.change_line):
                            self.time_before_crossing=self.time_before_crossing+ self.dt
                            self.waiting_time=self.waiting_time+ self.dt # a voir
                        
                    elif((not self.change_line) or (self.change_line and new_choice)):
                        self.crossing_time
                        self.Sp_y, self.Vp_y = new_spy, new_vpy
                        self.Sp_x, self.Vp_x = self.Sp_x + self.Vp_y*self.ratio*self.dt, self.Vp_y*self.ratio
                        self.crossing_time=self.crossing_time+self.dt
                        if self.change_line and new_choice:
                            self.apply_change_line(self.Sp_y, new_spy)
                    elif(self.change_line and not new_choice):
                        self.stop=True
                        distance = abs(self.direction*(self.cross_lines-distance_to_cross) - self.direction*self.cross_lines/2. -self.Sp_y)
                        pos_p_x = (self.Vp_x)*(distance)/abs(self.Vp_y +10e-3)
                        self.Vp_x = pos_p_x / self.dt
                        self.Sp_x = self.Sp_x + pos_p_x
                        self.Vp_y = self.direction*distance / self.dt
                        self.Sp_y = self.direction*((self.cross_lines-distance_to_cross) - self.cross_lines/2.)
                    else:
                        print("error")
                    self.change_line=False
                    
                # The pedestrian stops
                else:
                    self.time_stop = random.randint(2, 5)
                    if not self.choose: # si on est toujours en train d attendre avant de traverser
                        self.decision=False
                        self.time_stop = 0
                        self.waiting_time=self.waiting_time+ self.dt
                    self.Sp_x, self.Vp_x = self.Sp_x, 0.0
                    self.Sp_y, self.Vp_y = self.Sp_y, 0.0
                    self.t0 = self.t0 + self.dt
            # Before or after the crosswalk
            else:
                self.Sp_x, self.Vp_x = self.new_pedestrian_unif_x(time)
                self.Sp_y, self.Vp_y = self.new_pedestrian_unif_y(time)

    def CG_score(self, crossing_size):
        if(self.is_crossing):
            fem, child, midage, old = 0.0369, -0.0355, -0.0221, -0.1810
            alpha, sigma = 0.09, 0.09 
            gamma = math.log10(crossing_size/abs(self.initial_speed[1]+10e-3))
            log_val = alpha+gamma+fem*(self.gender==1)+child*(self.age==0)+midage*(self.age==1)+old*(self.age==2)
            log_val = log_val+random.normalvariate(0.0, sigma)
            return math.pow(10, log_val)
        else:
            return 0.
    
    def new_pedestrian_unif_x(self, time):
        return self.Sp_x + self.initial_speed[0] * self.dt, self.initial_speed[0]
    
    def new_pedestrian_unif_y(self, time):
        return self.Sp_y + self.initial_speed[1] * self.dt, self.initial_speed[1]
    
    def new_pedestrian_sin_y(self, time):
        t = time + self.dt
        speed_p = (self.A*math.sin(self.w*(t-self.t0))+self.B)
        pos_p = ((-self.cross_lines/2.) + (self.A*(-math.cos(self.w*(t-self.t0))+math.cos(self.w*self.t_init))/self.w))
        if(pos_p>=0.0 and speed_p<abs(self.initial_speed[1])):
            pos_p, speed_p=self.new_pedestrian_unif_y(time)
            return pos_p, speed_p
        return self.direction*pos_p, self.direction*speed_p
    
    def seed(self, seed=None):
        self.np_rand, seed = seeding.np_random(seed)
        return [seed]
    
    def get_data(self, cars_pos, cars_speed, cars_line, cars_light):
        if(not self.exist):
            return [0., 0., 0., 0., 0., 0., 0, 0, 0]
        self.delta=min(self.delta_l_all(cars_pos, cars_speed, cars_line, cars_light)* (self.is_crossing)* ( not self.ped_left),self.delta)
        return [self.Vp_x, self.Vp_y, self.Sp_x, self.Sp_y, self.delta, self.ped_left, self.ped_in_cross, self.exist, self.direction] 

    def is_in_front(self, car_line, next_line=0):
        line_1, line_2 = (-self.cross_lines/2)+self.cross*(car_line-0.5*next_line+1), (self.cross_lines/2)-self.cross*(self.max_lines-0.5*next_line-car_line)
        if(self.direction==-1):
            return self.Sp_y>=line_2-0.001
        else:
            return self.Sp_y<=line_1+0.001
        
    def is_crossing_in_front(self, car_line, prev_line=0):#pieton a commence a traverser
        line_1 = (-self.cross_lines/2)+self.cross*(car_line-prev_line)
        line_2=(self.cross_lines/2)-self.cross*(self.max_lines-car_line-1-prev_line)
        if(self.direction==-1):
            return self.Sp_y<line_2
        else:
            return self.Sp_y>line_1
        
    def new_reward_wait_safety(self, car_speed, car_pos, car_line):
        rew1=0.
        if ((not self.ped_left) * (self.is_crossing) * (car_pos < self.Sp_x) * self.is_in_front(car_line)):
            if(car_speed<0.05):
                exp_dl=0.
            else:
                dl = self.delta_l(car_pos, car_speed, car_line)/(car_speed)#+0.5)
                if(dl>=-1.):
                    exp_dl=max(-20.*math.exp(-4.*(dl)-4.),-20.0)
                else:
                    exp_dl=20.*dl
            exp_dl=exp_dl -(self.accident)*20
            if(exp_dl<self.worst_dl):
                self.worst_dl=exp_dl
        rew1=self.worst_dl
        rew2 = 0.0
        return rew1 + rew2
    
    def delta_l_all(self, car_pos, car_speed, car_line, car_light):
        delta_l=0.0
        for i in range(len(car_speed)):
            if(car_pos[i]<=self.Sp_x) and (self.is_in_front(car_line[i])) and (not self.ped_left) and (car_light[i]>=0):
                new_delta=abs(car_pos[i]-self.Sp_x)-(car_speed[i]*car_speed[i]/(-2.0*self.car_b[0,0])) - self.tau * (car_speed[i])#+0.5)
                delta_l= min(delta_l,new_delta)
        return delta_l
                       
    def delta_l(self, car_pos, car_speed, car_line):
        if(car_pos>self.Sp_x or self.ped_left or ( not self.is_in_front(car_line))):
            return 0.0
        else:
            return abs(car_pos - self.Sp_x) - (car_speed * car_speed / (-2.0 * self.car_b[0,0]))- self.tau * (car_speed)#+0.5)
                       
    def worst_delta_l(self, car_pos, car_speed,car_line):
        # car is after the ped or ped left or is not in front
        if(car_pos>self.Sp_x  or self.ped_left or (not self.is_in_front(car_line))):
            return 0.0
        else:
            return abs(car_pos - self.Sp_x) - (car_speed * car_speed / (-2.0 * self.car_b[0,0]))
    
class car:
    #car_rand, seed = seeding.np_random(42)
    def __init__(self, car_b, ped_b, cross, lines, dt, speed_limit, line):
        #self.seed(seed)
        self.car_b = car_b
        self.dt=dt
        self.cross=cross
        self.cross_lines=lines*cross
        self.line=line
        self.light=0.
        # Initial car acceleration
        self.initial_acc =0.
        self.Ac = self.initial_acc
        # Initial car speed
        self.initial_speed = speed_limit
        self.Vc = self.initial_speed
        # Initial car position
        #self.initial_pos = 0.0
        #self.Sc = self.initial_pos
        # Other parameters
        self.previous_acc = deque([0,0,0])
        self.discount_array = [1.0,0.,0.]
        self.acc_param = [(self.car_b[1,0]-self.car_b[0,0])/2.0, (self.car_b[1,0]+self.car_b[0,0])/2.0]
        self.possible_accident=0.0
        self.possible_accident2= 0.0
        self.error_scenario=0.0
        self.Ts=-10.
        mean_speed_ped=ped_b[0,1] + ped_b[1,1]/2
        min_speed_ped=ped_b[0,1]
        max_speed_ped=ped_b[1,1]
        min_pos_ped=ped_b[0,3]
        max_pos_ped=ped_b[1,3]
        finish_crosslines_time = (self.cross_lines*self.initial_speed)/(mean_speed_ped)
        mid_cross_time = 0.5*(self.cross*self.initial_speed)/(mean_speed_ped)
        low_car_range = (ped_b[0,3]*self.initial_speed)/min_speed_ped#slow time for ped * speed of the car  min_pos_ped+
        high_car_range = (ped_b[1,3]*self.initial_speed)/max_speed_ped #fast time for ped*speed of the car max_pos_ped+
        car_brake_dist = (self.initial_speed * self.initial_speed / (-2.0 * self.car_b[0,0]))
        self.initial_pos = random.uniform(low_car_range - finish_crosslines_time, high_car_range )#- 0.5*mid_cross_time)
        self.Sc = self.initial_pos #car_brake_dist
        self.finish_crossing=False
        self.tau=1.
        self.time_braking= - (self.Vc/ (2.0 * self.car_b[0,0]))+ self.tau 
        
    def reset_car(self, speed_x, pos_x, light, line):
            self.Sc = pos_x
            self.Vc = speed_x
            self.light = light
            self.line = line
        
    def acceleration(self, value_acc): # if we want to pre-transform the acceleration
        return min(max(value_acc,self.car_b[0,0]),self.car_b[1,0])
    
    def sigma(self, a, e=1):
        if(self.Vc==0.): # no limitation if the car accelerates
            return max(0.,a/abs(a))
        if(a>0):
            return 1
        else:
            sg=max(min(-self.Vc/(self.dt*a),1.),0.)
            return sg
        
    def step(self, action_acc, action_light):
        self.previous_acc.popleft()
        acc=self.acceleration(action_acc)
        sg=self.sigma(acc)
        self.previous_acc.append(acc)
        
        final_acc_array=np.array(self.previous_acc)
        final_acc=0.0
        for i in range(3):
                final_acc=final_acc+self.discount_array[i]*final_acc_array[-1-i]
        final_acc=final_acc*sg
        speed = self.Vc + self.dt * final_acc
        pos = (final_acc * math.pow(self.dt, 2.0)/2.0) + (self.Vc * self.dt) + (self.Sc)
        if(self.Sc>0. and pos>0.):
            self.finish_crossing=True
        self.Ac, self.Vc, self.Sc, self.light = final_acc, speed, pos, action_light
        
    def get_data(self):
        return [self.Ac, self.Vc, self.initial_speed-self.Vc, self.Sc, self.light, self.line]

    def new_reward_cross(self):
        # Speed_reward
        rew = - 10. * (self.Vc - self.initial_speed)**2 / self.initial_speed**2 
        return rew
    
    def new_reward_wait_speed(self):
        # Speed_reward
        rew = - 10. * (self.Vc - self.initial_speed)**2 / self.initial_speed**2 
        return rew                       
                       
                       
class Crosswalk_hybrid_multi_opti7_3(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    """
        Crosswalk_hybrid_multi_opti7_3: creation of the environment
        :param dt: step time
        :param cross_b: parameters for initial state env ([lower bound,upper bound])
        :param car_b: parameters for initial state car ([lower bound,upper bound])
        :param ped_b: parameters for initial state ped ([lower bound,upper bound])
        :param nb_ped: number of pedestrian
        :param simulation: pedestrian speed model
    """
    def __init__(self, car_b, ped_b, cross_b, nb_car, nb_ped, nb_lines, dt, max_episode, simulation="unif"):
        super(Crosswalk_hybrid_multi_opti7_3, self).__init__()
        #self.seed()
        self.viewer = False
        self.window = None
        # Fixed data for intersection & condition simulation
        self.dt = dt  # interval step
        self.car_b = car_b 
        self.ped_b = ped_b 
        self.cross_b = cross_b 
        self.Vm = 2.5  # max pedestrian speed
        self.tau = 1.0 # reaction time = 1 sec
        self.car_size = 4.0
        # Pedestrian model
        self.simulation = simulation    
        self.nb_car = nb_car
        self.nb_ped = nb_ped
        self.nb_lines = nb_lines
        # Max steps
        self.max_episode = max_episode
        self.average_ped_crossing=min(max_episode, 60)
        # Define action space: car acceleration
        self.action_space = Box(low = np.repeat(np.array([-1.0,-1.0]),nb_car),
                                high = np.repeat(np.array([1.0,1.0]),nb_car),
                                shape = (2*nb_car,), dtype=np.float32) 
        # respectively car acceleration (0), car_speed(1), car speed diff (2), car position (3), light(4), line (5)
        car_low = np.array([self.car_b[0,0], np.finfo(np.float32).min, np.finfo(np.float32).min,
                            np.finfo(np.float32).min,-1,0])
        car_high = np.array([self.car_b[1,0], np.finfo(np.float32).max, np.finfo(np.float32).max,
                             np.finfo(np.float32).max,1,nb_lines])
        # respectively  pedestrian speed x (0), pedestrian speed y (1), pedestrian position x(2)
        #               position y (3), dl(4), leave CZ(5), in CZ (6), exist(7), direction (8)
        ped_low = np.expand_dims(np.array([-self.Vm, -self.Vm, np.finfo(np.float32).min,np.finfo(np.float32).min,
                                           np.finfo(np.float32).min,0,0,0,-1]),axis=0)
        ped_high = np.expand_dims(np.array([self.Vm, self.Vm, np.finfo(np.float32).max, np.finfo(np.float32).max,
                                            np.finfo(np.float32).max,1,1,1,1]),axis=0)
        # respectively sizealllines (0), nb_ped (1), lines (1-4 par exemple)
        #any pedestrian in crosswalk , all pedestrian leave crosswalk, 
        env_low = np.array([ 0.1, 0, 0])
        env_high = np.array([ self.cross_b[1]*nb_lines, self.nb_ped, nb_lines])
        # we could add pedestrian presence in crosswalk and pedestrian finish crossing
        self.observation_space = Dict(
            {"car": Box(low=np.repeat(car_low,self.nb_car,axis=0).flatten(),#TODO: SEQUENCE FOR CAR RANDOM NUMBER
                        high=np.repeat(car_high,self.nb_car,axis=0).flatten(),
                        shape=(self.nb_car*6,), dtype=np.float32),
             # acc, speed, pos of car
             "ped": Box(low=np.repeat(ped_low,self.nb_ped,axis=0).flatten(),#TODO: SEQUENCE FOR PEDESTRIAN RANDOM NUMBER
                        high=np.repeat(ped_high,self.nb_ped,axis=0).flatten(),
                        shape=(self.nb_ped*9,), dtype=np.float32),
             # speed, pos in 2D of pedestrian, 
             "env": Box(low=env_low,
                        high=env_high,
                        shape=(3,), dtype=np.float32)
             #limit speed, CZ size, any ped in CZ, all ped leave, new ped
            })
        self.acc_param = [(self.car_b[1,0]-self.car_b[0,0])/2.0, (self.car_b[1,0]+self.car_b[0,0])/2.0]
    
    """
        Setting the seed: we need with the same seed for PPO and DDPG
        :param seed: seed value
        :return: seed value
    """
    def seed(self, seed=1):
        self.np_rand, seed = seeding.np_random(seed)
        return [seed]

    """
        Compute safety factor
        :param x: distance between the car and the crosswalk (negative value) 
        :param v: vehicle speed
        :return: safety factor
    """
    def delta_l(self, x, v):
        return -x - (v * v / (-2.0 * self.car_b[0,0]) + self.tau * v)

    """
        Compute reward
        :param acc: vehicle acceleration
        :param pos: vehicle position
        :param speed: vehicle speed
        :param pos_p: pedestrian position
        :param speed_p: pedestrian speed
        :return: immediate reward with respect to the new state
    """
    def new_reward_cross(self, speed):
        # Speed_reward
        rew = - 10. * (speed - self.speed_limit)**2 / self.speed_limit**2 
        return rew
    
    def new_reward_wait_speed(self, speed):
        # Speed_reward
        rew = - 10. * (speed - self.speed_limit)**2 / self.speed_limit**2 
        return rew
    
    """
        Compute the new step: with respect to the previous state and the taken action
        :param action_ar: action
        :return: new state, immediate reward, boolean (is the new state terminal?)
    """
    def step(self, actions):
        self.ped_possible= sum([(ped.remove>=1) for ped in self.pedestrian])>0 and self.time<(self.max_episode-self.average_ped_crossing)
        prev_cars_Sc = [car.Sc for car in self.cars]
        for i in range(self.nb_car):
            self.cars[i].step(actions[i],actions[i+self.nb_car])
        cars_Sc = [car.Sc for car in self.cars]
        cars_line = [car.line for car in self.cars]
        cars_Vc = [car.Vc for car in self.cars]
        cars_light = [car.light for car in self.cars]
        cars_time_braking = [car.time_braking for car in self.cars]
        for ped in self.pedestrian:
            ped.step(self.time, cars_Vc, cars_Sc, cars_line, cars_light)
        
        # Change pedestrian list
        new_ped_state=-1
        # enleve les pietons au bout d'un moment
        #for ped in self.pedestrian:
        #    if(ped.Sp_y*ped.direction>self.cross):
        #        number_name=ped.number_name
        #        if(self.pedestrian[number_name].is_crossing):
        #            self.ped_traffic-=1
        #        ped = pedestrian(self.ped_b, self.car_b, self.cross, self.nb_lines,
        #                         self.speed_limit, self.dt, simulation=self.simulation, 
        #                         is_crossing=False, exist=False, number_name=number_name)
                
        #if (self.ped_possible and random.uniform(0,1)<0.05 and self.ped_traffic<self.nb_ped):
        #    self.worst_dl = 0.0
        #    is_crossing = random.uniform(0,1)<0.5
        #    select_ped_id = random.choices([ped.number_name for ped in self.pedestrian if (ped.remove)])[0]
        #    select_car_Sc = random.choices([car.Sc for car in self.cars])[0]
        #    if(is_crossing):
        #        new_ped_state=select_ped_id
        #        self.ped_traffic+=1
        #    if(self.pedestrian[select_ped_id].is_crossing):
        #        self.ped_traffic-=1
        #    
        #    self.pedestrian[select_ped_id] = pedestrian(self.ped_b, self.car_b, self.cross, self.nb_lines, self.speed_limit, self.dt, simulation=self.simulation, is_crossing=is_crossing, exist=is_crossing, number_name=select_ped_id)
        
        #Boolean variables
        peds_left = sum([ped.ped_left for ped in self.pedestrian]) == self.ped_traffic
        peds_in_CZ = sum([ped.ped_in_cross for ped in self.pedestrian]) == self.ped_traffic
        
        # Detect dangerous situation
        #car_acc, car_speed, car_pos, car_speed_diff = self.car.get_data()
        car_dangers=np.array([0.]*self.nb_car)
        for ped in self.pedestrian:
            detection=ped.detection(self.cars,prev_cars_Sc)
            if ped.is_crossing and len(detection):
                car_dangers+=detection
        self.reward_light=car_dangers
        #print(car_dangers)
        # Compute reward for decision-making
        rewards=[]
        for car in self.cars:
            if(car.light<=0.0):
                reward=car.new_reward_cross()
            else:
                reward=car.new_reward_wait_speed()
                wait_reward=[ped.new_reward_wait_safety(car.Vc, car.Sc, car.line) for ped in self.pedestrian if ped.exist]
                if(len(wait_reward)>0):
                    reward+=min(wait_reward)
            rewards.append(reward)
        rewards = np.array(rewards)
        # Change state variable
        self.state["car"] = np.float32(np.array([car.get_data() for car in self.cars]).flatten())
        self.state["ped"] = np.float32(np.array([ped.get_data(cars_Sc,cars_Vc,cars_line, cars_light) for ped in self.pedestrian]).flatten()) 
        # need to choice the pedestrian
        self.state["env"] = np.float32(np.array([self.cross*self.nb_lines/2., self.ped_traffic, self.nb_lines]))
        
        # End of scenario?
        done = (self.time >= self.episode_length) or (self.ped_traffic<=0)
        self.time = self.time + self.dt
        self.prev_cars_Sc =cars_Sc 
        
        return self.state, rewards, bool(done), False, {}

    """
        Initialize a new episode
        :return: initial state of the episode
    """
    def reset(self, seed=None, options=None):
        
        if(seed is not None):
            self.seed(seed)
            
        # Environment
        self.cross = random.uniform(self.cross_b[0], self.cross_b[1]) 
        self.speed_limit = 10 
        # Initial pedestrian and car class
        self.pedestrian = [pedestrian(self.ped_b, self.car_b, self.cross, self.nb_lines, self.speed_limit, self.dt, simulation=self.simulation, is_crossing=False, exist=False, number_name=i) for i in range(self.nb_ped)]
        self.cars = [car(self.car_b, self.ped_b, self.cross, self.nb_lines, self.dt,
                         self.speed_limit, i%self.nb_lines) for i in range(self.nb_car)]
        #New pedestrian
        self.ped_traffic=random.randint(1,self.nb_ped) # we must use choices to prevent selection on the same element
        for i in range(self.ped_traffic):
            self.pedestrian[i]=pedestrian(self.ped_b, self.car_b, self.cross, self.nb_lines, self.speed_limit,
                                          self.dt, simulation=self.simulation, is_crossing=True, exist=True, number_name=i)
        # Initial car position
        #self.car = car(self.car_b, self.ped_b, self.cross, self.dt, self.speed_limit)
        
        # State
        cars_Sc = [car.Sc for car in self.cars]
        cars_Vc = [car.Vc for car in self.cars]
        cars_line = [car.line for car in self.cars]
        cars_light = [car.light for car in self.cars]
        self.state = self.observation_space.sample()
        self.state["car"] = np.float32(np.array([car.get_data() for car in self.cars]).flatten())
        self.state["ped"] = np.float32(np.array([ped.get_data(cars_Sc, cars_Vc,cars_line,cars_light) for ped in self.pedestrian]).flatten()) 
        self.state["env"] = np.float32(np.array([self.cross*self.nb_lines/2., self.ped_traffic, self.nb_lines])) 
        
        # Initial time
        self.time=0.0
        # Accident and Safe state
        self.error_scenario = False
        self.accident = False
        self.possible_accident = False
        self.possible_accident2= False
        self.worst_dl=0.0
        self.reward_light=0.0
        # Time after pedestrian crossing (let the vehicle accelerate)
        self.temps = self.speed_limit/self.car_b[1,0]
        self.episode_length=(self.max_episode-1)*self.dt
        return self.state,{}  # reward, done, info can't be included
    
    def reset_pedestrian(self,num_ped, speed_x, speed_y, pos_x, pos_y, dl, leave, CZ, exist, direction):
        self.pedestrian = [pedestrian(self.ped_b, self.car_b, self.cross, self.nb_lines,
                                      self.speed_limit, self.dt, simulation=self.simulation,
                                      is_crossing=False, exist=False, number_name=i) for i in range(self.nb_ped)]
        self.pedestrian[num_ped].reset_ped(speed_x, speed_y, pos_x, pos_y, dl, leave, CZ, exist, direction)
        
    def reset_cars(self, num_car, speed_x, pos_x, light, line):
        self.cars[num_car].reset_car(speed_x, pos_x, light, line)
        
    def reset_distrib(self,state_distrib):
        state_distrib=state_distrib.detach().numpy()
        input_p, input_c, input_e = 5, 1, 1
        self.cross = state_distrib[self.nb_car*input_c+input_p*self.nb_ped]
        for i in range(self.nb_ped):
            state_distrib[input_p*i+4] = 1.*(state_distrib[input_p*i+4]>=0)-1.*(state_distrib[input_p*i+4]<0)
            state_distrib[input_p*i+3] = state_distrib[input_p*i+4]*(state_distrib[input_p*i+3]-self.cross*self.nb_lines/2.)
            self.pedestrian[i].reset_ped(state_distrib[input_p*i], state_distrib[input_p*i+1], state_distrib[input_p*i+2],
                                         state_distrib[input_p*i+3], 0, state_distrib[input_p*i+4], self.cross)
        for i in range(self.nb_car):
            self.cars[i].reset_car(0, state_distrib[i+input_p*self.nb_ped], 0, i)
        
    def get_state(self):
        cars_Sc = [car.Sc for car in self.cars]
        cars_Vc = [car.Vc for car in self.cars]
        cars_line = [car.line for car in self.cars]
        cars_light = [car.light for car in self.cars]
        self.state = self.observation_space.sample()
        self.state["car"] = np.float32(np.array([car.get_data() for car in self.cars]).flatten())
        self.state["ped"] = np.float32(np.array([ped.get_data(cars_Sc,cars_Vc,cars_line,cars_light) for ped in self.pedestrian]).flatten())
        self.state["env"] = np.float32(np.array([self.cross*self.nb_lines/2., self.ped_traffic, self.nb_lines])) 
        return self.state

    """
        Quick render
        :param mode: only human render using pygame
    """
    def render(self, mode='human'):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((640, 480))
        # Discard events:
        for event in pygame.event.get():
            pass
        # Positions:
        half_width = 5.0
        half_height = 5.0
        x_ref = 640 / 2.0 + half_width
        y_ref = 480 / 2.0 + half_height - 25.0
        state_pos_p = self.state[3]
        state_pos = self.state[2]
        self.window.fill((0, 0, 0))
        pygame.draw.rect(self.window, pygame.Color((255, 0, 0)),
                         pygame.Rect(state_pos + x_ref - half_width, 0.0 + y_ref - half_height, 2.0 * half_width,
                                     2.0 * half_height), 2)
        
        pygame.draw.rect(self.window, pygame.Color((0, 255, 0)),
                         pygame.Rect(0.0 + x_ref - half_width, state_pos_p + y_ref - half_height, 2.0 * half_width,
                                     2.0 * half_height), 2)
        pygame.display.flip()
        #return None
    """
        Close the render
    """
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
