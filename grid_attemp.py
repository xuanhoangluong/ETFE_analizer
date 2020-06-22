# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 00:08:57 2020

@author: xuanh
"""

import numpy as np
import math

R = float(5)
##
def condition1(point, neighbor, ix, data):
    k = data['bound'] - data['box']
    if point[ix] - R < 0:
        A = ((np.all(neighbor[ix] >= 0) and np.all(point[ix] + R >= neighbor[ix])) or \
            (np.all(point[ix] - R + k[ix] <= neighbor[ix]) and np.all(neighbor[ix] < k[ix])))
        return A
    elif point[ix] + R - k[ix] >= 0:
        A = ((np.all(point[ix] - R <= neighbor[ix]) and np.all(neighbor[ix] < k[ix])) or \
            (np.all(neighbor[ix] >= 0) and np.all(point[ix] + R - k[ix] >= neighbor[ix])))
        return A
    else:
        A = (np.all(point[ix] - R <= neighbor[ix]) and np.all(point[ix] + R >= neighbor[ix]))
        return A

##
def condition2(point, i, ix, data):
    k1 = data['bound'] - data['box']
    if point[ix] - R < 0 and point[ix] - R + k1[ix] - i[ix] <= 0 < k1[ix] - i[ix]:
        return -1
    elif point[ix] + R - k1[ix] >= 0 and -i[ix] <= 0 <= point[ix] + R - k1[ix] - i[ix]:
        return 1
    else:
        return 0

##
def search(v_index, ix, data, vector1):
    # Sorting
    vector1.sort(key=lambda x: x[0][ix])
    k = []
    for i in vector1:
        k.append(i[2])
    k1 = k.index(v_index)
    k2 = round(len(vector1)*0.5)
    if k2 > k1:
        k3 = vector1[(k1-k2):].copy()
        del vector1[(k1-k2):]
        for i in range(len(k3)):
            vector1.insert(0, k3[-(i+1)])
        k1 += len(k3)
    elif k2 < k1:
        k3 = vector1[:(k1-k2)].copy()
        del vector1[:(k1-k2)]
        vector1.append(k3)
        k1 -= len(k3)
    # Searching
    vector2 = []
    for i in [1, -1]:
        k2 = 1
        j = condition1(vector1[k1][0], vector1[k1 + k2*i][0], ix, data)
        while j:
            vector2.append(vector1[k1 + k2*i])
            k2 += 1
            j = condition1(vector1[k1][0], vector1[k1 + k2*i][0], ix, data)
    if ix != 2:
        vector2.append(vector1[k1])
    return vector2

##Griding
dimension = (round((data['bound'][i]-data['box'][i])/R)+1 for i in range(3))
grid = []
for i1 in range(dimension[0]):
    grid.append([])
    for i2 in range(dimension[1]):
        grid[i1].append([])
        for i3 in range(dimension[2]):
            grid[i1][i2].append([])
for i1 in range(dimension[0]):
    for i2 in range(dimension[1]):
        for i3 in range(dimension[2]):
            for i in vector:

##
def density(v_index, data, vector):
    dump = []
    vector1 = vector.copy()
    for i in range(len(vector1)):
        vector1[i].append(i)
    # x-dimension
    vector_x = search(v_index, 0, data, vector1)
    # y-dimension
    vector_y = search(v_index, 1, data, vector_x)
    # z-dimension
    vector_z = search(v_index, 2, data, vector_y)
    for i in vector_z:
        k = [condition2(vector[v_index][0], i[0], ix, data) for ix in range(3)]
        k = vector[v_index][0] - i[0] - k*(data['bound']-data['box'])
        k = np.linalg.norm(k)
        if k <= R:
            dump.append(i)
    D = 0
    if len(dump) != 0:
        for i in dump:
            unit_point = vector[v_index][1]/np.linalg.norm(vector[v_index][1])
            unit_i = i[1]/np.linalg.norm(i[1])
            angle = np.arccos(np.dot(unit_point, unit_i))
            angle = angle*180/math.pi
            if angle >= 90:
                angle -= 90
            D += 1/np.linalg.norm(vector[v_index][0]-i[0])*angle/90
    return D

##
with open('dump.comp_0.cfg', 'r') as dump:
    data = dict()
    data['box'] = []
    data['bound'] = []
    data['data'] = []
    x2, x3 = 0, 3
    for x in dump:
        x = x.strip("\n")
        x = x.split()
        if x[0] == "ITEM:":
            if x[1] == "TIMESTEP":
                x2 = 1
            elif x[1:] == ["NUMBER","OF","ATOMS"]:
                x2 = 2
            elif x[1] == "BOX":
                x2 = 3
            elif x[1] == "ATOMS":
                data['name'] = x[2:]
                x2 = 4
        else:
            if x2 == 1:
                data['timestep'] = int(x[0])
                x2 = 0
            elif x2 == 2:
                data['nums_of_atoms'] = int(x[0])
                x2 = 0
            elif x2 == 3:
                data['box'].append(float(x[0]))
                data['bound'].append(float(x[1]))
                x3 -= 1
                if x3 == 0:
                    x2 = 0
                    data['box'] = np.asarray(data['box'], dtype='f4')
                    data['bound'] = np.asarray(data['bound'], dtype='f4')
            elif x2 == 4:
                data['data'].append(np.asarray([float(i) for i in x], dtype='f4'))

post_data = data['data'].copy()
for i in range(data['nums_of_atoms']):
    post_data[i][3:] -= data['box']
post_data.sort(key=lambda x: x[0])
molecule, vector_z = [], []
x = 1
for i in post_data:
    if i[1] == x:
        x += 1
        molecule.append([i])
    else:
        molecule[x-2].append(i)

for i in molecule:
    for j in range(len(i)):
        if j != 0 and j != len(i)-1:
            k = np.zeros((3,3), dtype=int)
            k1 = data['bound'] - data['box']
            for x in range(3):
                if abs(i[j+1][x+3]-i[j-1][x+3]) > k1[x]*0.5:
                    if i[j-1][x+3] < k1[x]*0.5:
                        k[0][x] = 1
                    if i[j][x+3] < k1[x]*0.5:
                        k[1][x] = 1
                    if i[j+1][x+3] < k1[x]*0.5:
                        k[2][x] = 1
            k2 = (i[j-1][3:] + i[j][3:] + i[j+1][3:] + (k[0]+k[1]+k[2])*k1[x])*0.33333
            for x in range(3):
                if k2[x] >= k1[x]:
                    k2[x] -= k1[x]
            vector.append([k2, i[j+1][3:]-i[j-1][3:]])

density_map = np.zeros(len(vector), 4)
for j in range(len(vector)):
    print(j)
    density_map[j][:3] = vector[j][0].copy()
    density_map[j][3] = density(j, data, vector)
