#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 15:32:46 2024

@author: leo
"""

from neuron import h
import matplotlib.pyplot as plt
import numpy as np

def rect(T):
    """create a centered rectangular pulse of width $T"""
    return lambda t: (-T/2 <= t) & (t < T/2)

def trapezoid(T, rise, fall):
    """
    Create a centered trapezoidal pulse with specified rise and fall times
    
    Parameters:
    - T: Total width of the trapezoidal pulse (T = rise + plateau + fall)
    - rise: Time duration for the rising edge
    - fall: Time duration for the falling edge
    
    Returns:
    - A function that evaluates the pulse at any time t.
    """

    return lambda t: np.clip((t + (T/2)) / rise, 0, 1) * np.clip(((T/2) - t) / fall, 0, 1)


def pulse_train(t, at, shape):
    """create a train of pulses over $t at times $at and shape $shape"""
    return np.sum(shape(t - at[:,np.newaxis]), axis=0)


#sufix "np" means intended use for Numpy. Note difference with Vector class of Neuron
tstop = 300
dt_np = 0.0001
cell_pos = 0.5

train_period = 30 #ms
theta1 = 30
theta2 = 30
alpha = .3
pulse_width = train_period * (1 - alpha)
delay = train_period * alpha

C_base = 1


h.load_file("stdrun.hoc")
h.tstop = tstop

cell = h.Section()
#hh = Hodgkin Huxley channels
cell.insert("hh")
#pas mechanism will add term g_pas*(v-e_pas) to equation
cell.insert("pas")
cell.e_pas = 0 

v = h.Vector()
v.record(cell(cell_pos)._ref_v)

t = h.Vector()
t.record(h._ref_t)

t_np = np.arange(0, tstop, dt_np)
tt = h.Vector(t_np)


for C_0 in (.8,):
    
    #with sign switching, "rise" will be "fall"
    rise = (C_base - C_0) * train_period / theta1
    fall = (C_base - C_0) * train_period / theta2
    cm_np = pulse_train(
        t = t_np,              # time domain
        at = np.arange(delay, tstop, train_period),  # times of pulses
        shape = trapezoid(pulse_width, rise, fall)   # shape of pulse
    )
    cm_np = cm_np * (C_base - C_0) + C_0
    cm = h.Vector(cm_np)    
    cm.play(cell(cell_pos)._ref_cm, tt)
    
    dc_dt_np = np.diff(cm_np, prepend=cm_np[0]) / dt_np
    dc_dt = h.Vector(- dc_dt_np)
    dc_dt.play(cell(cell_pos)._ref_g_pas, tt)
    
    cmr = h.Vector()
    cmr.record(cell(cell_pos)._ref_cm)
    
    h.run()
    plt.subplot(211)
    plt.plot(t, v)
    plt.subplot(212)
    plt.plot(t, cmr)
