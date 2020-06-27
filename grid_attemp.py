# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 00:08:57 2020

@author: xuanh
"""
import numpy as np
import math, copy
from grispy import GriSPy
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

R = 7.5
fn = 0
##
with open('dump.comp_%d.cfg' % fn, 'r') as dump:
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

post_data = copy.copy(data['data'])
for i in range(data['nums_of_atoms']):
    post_data[i][3:] -= data['box']
post_data.sort(key=lambda x: x[0])
molecule, vector, vector_np = [], [], []
x = 1
for i in post_data:
    if i[1] == x:
        x += 1
        molecule.append([i])
    else:
        molecule[x-2].append(i)
k1 = data['bound'] - data['box']
for i in molecule:
    for j in range(1, len(i)-1):
        k = np.zeros((3,3), dtype=int)
        for x in range(3):
            if abs(i[j+1][x+3]-i[j-1][x+3]) > k1[x]*0.5:
                if i[j-1][x+3] < k1[x]*0.5:
                    k[0][x] = 1
                elif i[j+1][x+3] < k1[x]*0.5:
                    k[2][x] = 1
                if i[j][x+3] < k1[x]*0.5:
                    k[1][x] = 1
        k2 = (i[j-1][3:] + i[j][3:] + i[j+1][3:] + (k[0,:]+k[1,:]+k[2,:])*k1)*0.33333
        for x in range(3):
            if k2[x] >= k1[x]:
                k2[x] -= k1[x]
        vector.append([k2, (i[j+1][3:]-i[j-1][3:]+k[0,:]*k1[0]-k[2,:]*k1[2])])
        vector_np.append(k2)

vector_np = np.array(vector_np)
periodic = {0: (0, float(k1[0])), 1: (0, float(k1[1])), 2: (0, float(k1[2]))}
gsp = GriSPy(vector_np)
gsp.set_periodicity(periodic)
bubble_dist, bubble_ind = gsp.bubble_neighbors(vector_np, distance_upper_bound=R)
density_map = np.zeros((len(vector), 4), dtype=float)
for i in range(len(vector)):
    D = 0
    print(i)
    for j in range(len(bubble_ind[i])):
        if bubble_ind[i][j] != i:
            unit_point = vector[i][1]/np.linalg.norm(vector[i][1])
            unit_neighbor = vector[bubble_ind[i][j]][1]/np.linalg.norm(vector[bubble_ind[i][j]][1])
            dot = np.dot(unit_point, unit_neighbor)
            if dot > 1:
                dot = 1
            elif dot < -1:
                dot = -1
            factor = 1
            if 0 <= abs(j-i) < 3:
                fator = (abs(j-i)/3)**2
            angle = np.arccos(dot)
            angle = angle*180/math.pi
            if angle >= 90:
                angle -= 90
            D += (5.01/bubble_dist[i][j])*(1-angle/90)*factor
    density_map[i,:3] = copy.copy(vector_np[i])
    density_map[i,3] = D
d_min = np.amin(density_map[:,3])
d_max = np.amax(density_map[:,3])
density_map[:,3] = ((density_map[:,3]-d_min)/(d_max-d_min))

## x-z plane
xz_map_dump = []
xz = list(range(len(density_map)))
i = 0
if len(density_map[:,0]) == len(set(density_map[:,0])) and \
   len(density_map[:,2]) == len(set(density_map[:,2])):
        while True:
            xz_duplicate = []
            print(xz[0])
            for j in xz:
                k = 0
                if j != i and \
                   density_map[i,0] == density_map[j,0] and \
                   density_map[i,2] == density_map[j,2]:
                       if k == 0:
                           xz_duplicate.append([i])
                           k += 1
                       xz_duplicate[-1].append(j)
            xz_map_dump.append(density_map[i])
            if len(xz_duplicate) != 0:
                for j in range(1, len(xz_duplicate)):
                    xz_map_dump[-1][3] += density_map[j,3]
                for j in xz_duplicate:
                    del xz[xz.index(j)]
            else:
                del xz[xz.index(i)]
            if len(xz) != 0:
                i = xz[0]
            else:
                break
        xz_map = copy.copy(np.array(xz_map_dump))
        del xz_map_dump, xz_duplicate
else:
    xz_map = copy.copy(density_map)
xz_map[:,1] = 0

## y-z plane
yz_map_dump = []
yz = list(range(len(density_map)))
i = 0
if len(density_map[:,1]) == len(set(density_map[:,1])) and \
   len(density_map[:,2]) == len(set(density_map[:,2])):
        while True:
            yz_duplicate = []
            print(yz[0])
            for j in yz:
                k = 0
                if j != i and \
                   density_map[i,1] == density_map[j,1] and \
                   density_map[i,2] == density_map[j,2]:
                       if k == 0:
                           yz_duplicate.append([i])
                           k += 1
                       yz_duplicate[-1].append(j)
            yz_map_dump.append(density_map[i])
            if len(yz_duplicate) != 0:
                for j in range(1, len(yz_duplicate)):
                    yz_map_dump[-1][3] += density_map[j,3]
                for j in yz_duplicate:
                    del yz[yz.index(j)]
            else:
                del yz[yz.index(i)]
            if len(yz) != 0:
                i = yz[0]
            else:
                break
        yz_map = copy.copy(np.array(yz_map_dump))
        del yz_map_dump, yz_duplicate
else:
    yz_map = copy.copy(density_map)
yz_map[:,0] = 0

####################################### Plot
fig, (ax1, ax2) = plt.subplots(ncols=2, dpi=300)

# ----------
# Tricontour
# ----------
# Directly supply the unordered, irregularly spaced coordinates
# to tricontour.

# XZ POV
ax1.tricontour(xz_map[:,0], xz_map[:,2], xz_map[:,3], levels=14, linewidths=0.5, colors='k')
cntr1 = ax1.tricontourf(xz_map[:,0], xz_map[:,2], xz_map[:,3], levels=14, cmap="RdBu_r")
fig.colorbar(cntr1, ax=ax1)
#ax1.plot(xz_map[:,0], xz_map[:,2], 'ko', ms=3)          % reference points
ax1.set(xlim=(0, float(k1[0])), ylim=(0, float(k1[2])))
ax1.set_title('xz POV')

#YZ POV
ax2.tricontour(yz_map[:,1], yz_map[:,2], yz_map[:,3], levels=14, linewidths=0.5, colors='k')
cntr2 = ax2.tricontourf(yz_map[:,1], yz_map[:,2], yz_map[:,3], levels=14, cmap="RdBu_r")
fig.colorbar(cntr2, ax=ax2)
#ax2.plot(yz_map[:,0], yz_map[:,2], 'ko', ms=3)          % reference points
ax2.set(xlim=(0, float(k1[1])), ylim=(0, float(k1[2])))
ax2.set_title('yz POV')

#
plt.subplots_adjust(hspace=0.5)
plt.savefig('%d.png' % fn)
plt.show()
