# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 13:11:33 2020

@author: xuanh
"""
import numpy as np
import copy
import math

R = 5
##
def condition1(point, neighbor, ix):
    k = [float((data['bound']-data['box'])[i]) for i in range(3)]
    poi = [float(point[i]) for i in range(3)]
    nei = [float(neighbor[i]) for i in range(3)]
    if poi[ix] - R < 0:
        A = ((0 <= nei[ix] <= poi[ix] + R) or (poi[ix] - R + k[ix] <= nei[ix] < k[ix]))
        return A
    elif poi[ix] + R - k[ix] >= 0:
        A = ((poi[ix] - R <= nei[ix] < k[ix]) or (0 <= nei[ix] <= poi[ix] + R - k[ix]))
        return A
    else:
        A = (poi[ix] - R <= nei[ix] <= poi[ix] + R)
        return A

##
def condition2(point, neighbor):
    k = [float((data['bound']-data['box'])[i]) for i in range(3)]
    poi = [float(point[i]) for i in range(3)]
    j = 0
    while True:
        nei = [float(neighbor[j][0][i]) for i in range(3)]
        add = [0,0,0]
        for ix in range(3):
            if (poi[ix] - R < 0) and (poi[ix] - R + k[ix] <= nei[ix] < k[ix]):
                add[ix] = -1
            elif (poi[ix] + R >= k[ix]) and (0 <= nei[ix] <= poi[ix] + R - k[ix]):
                add[ix] = 1
        yield np.array(add)

##
def search(v_index, ix, vector1):
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
        for i in range(len(k3)):
            vector1.append(k3[i])
        k1 -= len(k3)
    # Searching
    for i in [1, -1]:
        k2 = 1
        j = condition1(vector1[k1][0], vector1[k1 + k2*i][0], ix)
        while j:
            yield vector1[k1 + k2*i]
            k2 += 1
            j = condition1(vector1[k1][0], vector1[k1 + k2*i][0], ix)
    if ix != 2:
        yield vector1[k1]

## Density calculator
def density(vector):    
    v_index = 0
    vector1 = vector.copy()
    for i in range(len(vector1)):
        vector1[i].append(i)
    vector_x, vector_y, vector_z = [], [], []
    while True:
        # x-dimention
        while True:
            vector_x.append(next(search(v_index, 0, vector1)))
        # y-dimention
        while True:
            vector_y.append(next(search(v_index, 1, vector_x)))
        # z-dimention
        while True:
            vector_z.append(next(search(v_index, 2, vector_y)))
        D = 0
        unit_point = vector[v_index][1]/np.linalg.norm(vector[v_index][1])
        for neighbor in vector_z:
            k = next(condition2(vector[v_index][0], vector_z))
            k = vector[v_index][0] - neighbor[0] - k*(data['bound']-data['box'])
            k = np.linalg.norm(k)
            if k <= R:
                unit_neighbor = neighbor[1]/np.linalg.norm(neighbor[1])
                angle = np.arccos(np.dot(unit_point, unit_neighbor))
                angle = angle*180/math.pi
                norm = np.linalg.norm(vector[v_index][0]-neighbor[0])
                if angle >= 90:
                    angle -= 90
                D += 1/norm*angle/90
        yield D
        if v_index + 1 < len(vector):
            v_index += 1

##
def main():
    # Made dump.data sorting in order of atom number.
    post_data = data['data'].copy()
    for i in range(data['nums_of_atoms']):
        post_data[i][3:] -= data['box']
    post_data.sort(key=lambda x: x[0])
    # Organize datas to set of molecule
    molecule, vector, density_map = [], [], []
    x = 1
    for i in post_data:
        if i[1] == x:
            x += 1
            molecule.append([i])
        else:
            molecule[x-2].append(i)
    # Create a medium set of point and its vector from 3-continuos atoms to for
    # calculate crystallization density (cheng2017).
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
    # Calculate crystallization density for each point.
    j = 0
    while True:
        print(j)
        D = next(density(vector))
        yield [vector[j][0], D]
        j += 1

if __name__== "__main__":
    # Read dump.file
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
    # Mapping crystallizaton density of the structure
    density_map = []
    density_map.append(next(main()))