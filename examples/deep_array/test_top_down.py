# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import matplotlib.pyplot as plt

import floris.tools as wfct
import numpy as np

import sys
from top_down import TopDown
import scipy.optimize as opt
import copy

# ==================================================
# COMPUTE FLORIS WITHOUT DEEP ARRAY EFFECTS:
# ==================================================
# Initialize the FLORIS interface fi
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes
fi = wfct.floris_interface.FlorisInterface("../example_input.json")

# reinitialize flow field with large farm
layout_x = [2436.8,3082.4,2759.6,2114,1791.2,2114,2759.6,3728,3555.012001,3082.4,2436.8,1791.2,1318.587999,1145.6,
            1318.587999,1791.2,2436.8,3082.4,3555.012001,4373.6,4268.658834,3965.207339,3496.129193,2912.256291,
            2276.860324,1658.796302,1125.041052,733.4352387,526.4154276,526.4154276,733.4352387,1125.041052,1658.796302,
            2276.860324,2912.256291,3496.129193,3965.207339,4268.658834]
layout_y = [2436.8,2436.8,2995.906001,2995.906001,2436.8,1877.693999,1877.693999,2436.8,3082.4,3555.012001,3728,
            3555.012001,3082.4,2436.8,1791.2,1318.587999,1145.6,1318.587999,1791.2,2436.8,3065.677932,3626.407182,
            4058.224035,4314.334435,4366.984846,4210.469779,3861.75007,3358.614911,2755.586802,2118.013198,1514.985089,
            1011.84993,663.1302209,506.6151539,559.2655649,815.3759649,1247.192818,1807.922068]
fi.reinitialize_flow_field(layout_array=(layout_x,layout_y))

# Calculate wake
fi.floris.farm.flow_field.solver = 'basic'
fi.calculate_wake()
u_floris = [fi.floris.farm.turbines[i].average_velocity for i in range(len(layout_x))]
power_init = fi.get_turbine_power()

# ==================================================
# COMPUTE FLORIS WITH DEEP ARRAY EFFECTS:
# ==================================================

# initialize the top down model
fi.floris.farm.flow_field.solver='deep_array'
zh = fi.floris.farm.turbines[0].hub_height
z0 = 0.15   # Bottom roughness height [m]
D = fi.floris.farm.turbines[0].rotor_diameter
H = 750     # minimum boundary layer height
N = len(layout_x)
U_infty = 8.0
td = TopDown(zh, z0, D, U_infty, H, N, layout_x, layout_y, fi)
td.calculate_area()

# recalculate farm thrust coefficient for top down model
td.fi = copy.deepcopy(fi)

cft = td.farm_thrust_coefficient()
print('Wind Farm Thrust Coefficient: ', cft)
# plt.figure()
# plt.plot(cft, label='cft')
# # plt.plot(fbar, label='fbar')
# plt.plot(Ctp, label='Ctp')
# plt.plot(Ct, label='Ct')
# plt.legend()

td.calc(cft,layout_x)
u_td = td.uh

sowfa_base_power = 1000*np.array([782.2,879.8,606.1,1550,626.7,1594.6,622.3,826.8,1033.8,1052.1,1936.4,1971.7,2023.2,1952.7,1813.4,
                    1954.8,1818.4,1078.3,845.9,896.9,685.6,924.1,1907.6,822.2,1971,1871.3,2016.4,2089.6,2046.3,1966.7,
                    1779.9,1939.4,1988.3,1756.3,782.8,1789.6,1000.9,636.9])

rho = 1.225
Cp = 0.45
power_conv = 0.5 * rho * ( np.pi * (D/2)**2 ) * Cp
plt.figure()
plt.plot(u_td**3 * power_conv, 'ko-', label='Top Down')
plt.plot(np.array(u_floris)**3 * power_conv, 'bo-', label='Initial FLORIS')
plt.plot(sowfa_base_power,'ro-',label='SOWFA')
plt.legend()

plt.figure()
plt.plot(u_td**3 * power_conv, sowfa_base_power, 'ko', label='Top Down')
plt.plot(np.array(u_floris)**3 * power_conv, sowfa_base_power, 'bo', label='Initial FLORIS')
plt.plot([np.min(sowfa_base_power), np.max(sowfa_base_power)], [np.min(sowfa_base_power),np.max(sowfa_base_power)])
plt.grid()
plt.legend()
plt.show()