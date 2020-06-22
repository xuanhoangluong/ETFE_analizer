# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 12:41:51 2020

@author: xuanh
"""
""

import copy, time, math
import random as rd
import numpy as np
from numpy import linalg as LA
from scipy.stats import expon

# In this code I am generating a semicrystalline copolymer (ETFE). Base on suggetion 
# about lattic in https://icme.hpc.msstate.edu/mediawiki/index.php/Amorphous_Polymer_Generator
# and using SRW(self random walk) for amorphous region and parameter for crystal region form 
# balijepalli2000 and Flourinated Polymer(book) and TDT et al 2020.
# Implement form NguyenVietBac source code.
# Structure set up was confirmed by Lee2011 and reuse by kim2014.
# Simulation will employed coarse grain simulation for CH2 and CF2
start_time = time.time()


## Initial data
file_out_name = 'EFTE_LXH_upgmodel'
#density = 0.081 # density was cacualted from density of ETFE (1.7g/cm3) 
b_length = 1.55                                 # avarage bond length between CH and CF
massCH2 = 14
massCF2 = 50
cube_a = b_length*np.sqrt(2)                    # length of the cube fcc

crystallinity = 0.35                            # percentage of unit in crystals region

n_oc_sites = 20000                              # total nuber of unit in sample
n_lame = int(n_oc_sites*crystallinity)          # number of unit in crystal region

height_poly = int(162/b_length*np.sqrt(2))		# L1D = 162-190 Angstrom from Tap et al 2020
if height_poly%2 == 0:
    height_poly += 1
height_lame = int(49/(b_length*np.sqrt(2))) + 1
height_amophous = height_poly - 2*height_lame
restrict_under = height_lame
restrict_above = height_poly - height_lame

## Calculate size of lattice in x and y axis
n_size_ground = int(n_lame/(2*height_lame+2))   # number of atoms in a layer in lamellar regions
j = 5                                       	# j = 20 cho n_size_ground = 116
while True:
	atoms = 0
	for x in range(j+1):
		i = 0
		while True:
			if x % 6 == 0:
				y = 10*i
			elif x % 6 == 3:
				y = 10*i + 5
			if y > j:
				break
			i += 1
			atoms += 1
	if atoms >= n_size_ground:
		break
	j += 1
#size_ground = 2*(j+4-(j%4))
size_ground = j - 1
n_size_ground = atoms

net_crys = []
for x in range(int(size_ground/1)):
	i = 0
	while True:
		if x % 6 == 0:
			y = 10*i
			a = 0
		elif x % 6 == 3:
			y = 10*i + 5
			a = 1
		if y > int(size_ground/1) - 1:
			break
		i += 1
		net_crys.append([1*x,1*y])  # temporary for base layer

#def rotation_matrix(A,B):
## a and b are in the form of numpy array
#   ax, ay, az = A[0], A[1], A[2]
#   bx, by, bz = B[0], B[1], B[2]
#   au = A/(np.sqrt(ax*ax + ay*ay + az*az))
#   bu = B/(np.sqrt(bx*bx + by*by + bz*bz))
#   scale = (np.sqrt(bx*bx + by*by + bz*bz))/(np.sqrt(ax*ax + ay*ay + az*az))
#
#   R=np.array([[bu[0]*au[0], bu[0]*au[1], bu[0]*au[2]],
#               [bu[1]*au[0], bu[1]*au[1], bu[1]*au[2]],
#               [bu[2]*au[0], bu[2]*au[1], bu[2]*au[2]] ])
#   return R
#R = []
#ref = [[0,1,1],[1,1,0],[-1,1,0],[1,0,1],[-1,0,1]]
#for x in range(1,len(ref)):
#    R.append(rotation_matrix(ref[0],ref[x]))

    
# This function take two atom (a dict in python) and return an vector from atomb to atoma
def create_vector(atoma, atomb):
	atom1 = dict(atoma)
	atom2 = dict(atomb)
	# create first vector
	aa = np.zeros(3)
	aa[0] += aa[0] + atom2['x0'] - atom1['x0']
	aa[1] += aa[1] + atom2['x1'] - atom1['x1']
	aa[2] += aa[2] + atom2['x2'] - atom1['x2']
	return aa

# This function take an atom(a dict in python) and find and return all empty site around that site in the net
def find_empty_sites_around(curr_site, sites):
	empty_sites_around = []
	m = [[1,0,1],[0,1,1],[-1,0,1],[0,-1,1],
		 [1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0],
		[1,0,-1],[0,1,-1],[-1,0,-1],[0,-1,-1]]
	for x in range(len(m)):
		dx = curr_site['x0'] + m[x][0]
		dy = curr_site['x1'] + m[x][1]
		dz = curr_site['x2'] + m[x][2]
		if dx == size_ground:
			dx = 0
		if dy == size_ground:
			dy = 0
		if dx > size_ground:
			dx = 1
		if dy > size_ground:
			dy = 1
		if dx < 0:
			dx = size_ground - 1
		if dy < 0:
			dy = size_ground - 1
		if sites[dx][dy][dz] == 0:
			temp = {}
			temp['x0'] = dx
			temp['x1'] = dy
			temp['x2'] = dz
			temp['vec'] = np.array(m[x])
			empty_sites_around.append(temp)
	return empty_sites_around

# This function take an atom then from that site find an random new empty site which satify condition
# of possible fo find a new site which again satify the condition of possible find a a new empty site
# (there is another condition needed to be satify. It is the angle between 2 vector of three recent atom
# have to be 60 degree)
def find_next_site(curr_site, sites, restrict_under, restrict_above, relative_site):
	empty_sites_around = find_empty_sites_around(curr_site, sites)
	found = 0
	distance_check_curr = LA.norm(create_vector(curr_site, relative_site))
	active_check = 1
	while found != 1:
		if empty_sites_around == []:
			break
		temp1 = rd.randint(0, len(empty_sites_around) - 1)
		next_site = empty_sites_around[temp1]
		del empty_sites_around[temp1]
		va1 = next_site['vec']
		empty_sites_around2 = find_empty_sites_around(next_site, sites)
		for check_site in empty_sites_around2:
			vb1 = check_site['vec']
			angle = math.acos((np.dot(va1, vb1))/(LA.norm(va1)*LA.norm(vb1)))
			angle = round(angle*180/math.pi)
			if (45<angle<75) and (restrict_above>next_site['x2']>restrict_under):
				distance_check_next = LA.norm(create_vector(next_site, relative_site))
				if active_check == 1:
					if distance_check_next >= distance_check_curr * 1.0:
						found = 1
						return next_site
				elif active_check == 0:
					found = 1
					return next_site
				if empty_sites_around == []:
					empty_sites_around = find_empty_sites_around(curr_site, sites)
					active_check = 0
	return 0

# As the find_next_sites above but this time instead of empty site we will find and head( a begin atom
# of a chain in crystall region)
def find_exist_head_around(curr_site, sites):
	empty_head_aroud = []

	for z in [1, -1]:
		for x in [-1, 0, 1]:
			for y in [-1, 0, 1]:
				dx = curr_site['x0'] + x
				dy = curr_site['x1'] + y
				dz = curr_site['x2'] + z
				if dx == size_ground:
					dx = 0
				if dy == size_ground:
					dy = 0
				if dx > size_ground:
					dx = 1
				if dy > size_ground:
					dy = 1
				if dx < 0:
					dx = size_ground - 1
				if dy < 0:
					dy = size_ground - 1
				if sites[dx][dy][dz] == 2:
					temp = {}
					temp['x0'] = dx
					temp['x1'] = dy
					temp['x2'] = dz
					temp['vec'] = np.array([x,y,z])
					empty_head_aroud.append(temp)
	return empty_head_aroud

## Contruction of chain in crystal region
def chain_in_crystal(head, seed_index, temp_net, sites, temp_chain):
	temp = {}	
	if head == restrict_above:
		i = net_crys.index([temp_net[1][seed_index][0]-temp_net[3], temp_net[1][seed_index][1]])
		for z in range(head, height_poly+1, 1):
			if z % 2 == 1:
				temp['x0'] = net_crys[i][0] + 1
			else:
				temp['x0'] = net_crys[i][0]
			temp['x1'] = net_crys[i][1]
			temp['x2'] = z
			temp['vec'] = []
			sites[temp['x0']][temp['x1']][temp['x2']] = 1
			if z == head:
				sites[temp['x0']][temp['x1']][temp['x2']] = 3
			temp_chain.append(dict(temp))
		for z in range(0, height_lame+1, 1):
			if z % 2 == 1:
				temp['x0'] = net_crys[i][0] + 1
			else:
				temp['x0'] = net_crys[i][0]
			temp['x1'] = net_crys[i][1]
			temp['x2'] = z
			temp['vec'] = []
			sites[temp['x0']][temp['x1']][temp['x2']] = 1
			temp_chain.append(dict(temp))
#		sites[temp['x0']][temp['x1']][temp['x2']] = 2
		del temp_net[0][temp_net[0].index([net_crys[i][0]+temp_net[2], net_crys[i][1]])]
	elif head == restrict_under:
		i = net_crys.index([temp_net[0][seed_index][0]-temp_net[2], temp_net[0][seed_index][1]])
		for z in range(head, -1, -1):
			if z % 2 == 1:
				temp['x0'] = net_crys[i][0] + 1
			else:
				temp['x0'] = net_crys[i][0]
			temp['x1'] = net_crys[i][1]
			temp['x2'] = z
			temp['vec'] = []
			sites[temp['x0']][temp['x1']][temp['x2']] = 1
			if z == head:
				sites[temp['x0']][temp['x1']][temp['x2']] = 3
			temp_chain.append(dict(temp))
		for z in range(height_poly, restrict_above-1, -1):
			if z % 2 == 1:
				temp['x0'] = net_crys[i][0] + 1
			else:
				temp['x0'] = net_crys[i][0]
			temp['x1'] = net_crys[i][1]
			temp['x2'] = z
			temp['vec'] = []
			sites[temp['x0']][temp['x1']][temp['x2']] = 1
			temp_chain.append(dict(temp))
		del temp_net[1][temp_net[1].index([net_crys[i][0]+temp_net[3], net_crys[i][1]])]
#		sites[temp['x0']][temp['x1']][temp['x2']] = 2
	return

# Decise max lengh of a chain in amophous region
tank = [expon.rvs(loc=int((height_amophous-1)/2), scale=15, size=5000),
		expon.rvs(loc=int((height_amophous-1)/2), scale=30, size=5000)]
def chain_lengh():
	while True:
		temp_lengh = [int(tank[0][rd.randint(0, len(tank[0])-1)]),
					  int(tank[1][rd.randint(0, len(tank[1])-1)])]
		if (int((height_amophous-1)/2) <= temp_lengh[0] <= int((height_amophous-1)*2.5)) or \
		   (int((height_amophous-1)/1) <= temp_lengh[1] <= int((height_amophous-1)*2.5)):
			break
	# Maximum lengh of a tail and loop, respectively
	print('max lengh ' + str(temp_lengh[0]) + ' ' + str(temp_lengh[1]))
	return temp_lengh

def main():
	# In here we will create a layer of atoms (imitate the subtructe saw in xy plane)
	# then duplicate with a small shift so we can get zig-zag like structure (although C-C angle = 90 degree)
	# define sites as 4 dimention matrix of lacttic and atoms_list to contain all created atoms
	sites = np.zeros((size_ground+1,size_ground+1,height_poly+2), dtype=np.int)
	# sites hold all unit on lattic in 3 dimention matrix
	# Define where to start construct lamellae this first point, we build from here outward
	temp = {}
	temp['x0'] = 0				# index x axis
	temp['x1'] = 0				# index y axis
	temp['x2'] = 0				# index z axis
	temp['vec'] = []

#		if a == 0:
#			if ((y - 4)//8)*16 + 8 > 2*(y - 4):
#				tfloop_bound = 2*(y - 8)
#			else:
#				tfloop_bound = 2*(y - 4)
	# Initial tigh-fold loop trajectory
#	tfloop_net = []
#	for i in range(tfloop_bound//8 + 1):
#		for j in range(tfloop_bound//16 + 1):
#			tfloop_net.append([8*i, 16*j, 8*i, 16*j+8])
#	tfloop = [[0,0,0],[0,8,0],
#			  [-1,0,1],[-1,1,2],[0,1,3],[0,2,4],[1,2,5],[1,3,6],[0,4,6],
#			  [1,5,6],[1,6,5],[0,6,4],[0,7,3],[-1,7,2],[-1,8,1]]

	# Section 2: Growing amouphous region from surface
	temp_net = [copy.deepcopy(net_crys), copy.deepcopy(net_crys), [0], [0]]
	xn = 0
	for x in [restrict_under, restrict_above]:
		for n in range(len(net_crys)):
			if x % 2 == 1:
				temp_net[xn][n][0] += 1
				temp_net[xn+2] = 1
			else:
				temp_net[xn][n][0] += 0
				temp_net[xn+2] = 0
			sites[temp_net[xn][n][0]][temp_net[xn][n][1]][x] = 2
		xn += 1

	# Growing amouphous
	amo_sites = int(n_size_ground*2*(height_lame+1)/crystallinity*(1-crystallinity)) + 1
	chain_num, switch, switch1, n, lamellar_chain = 0, 0, 0, 0, 0
	loop, tail, bridge = [], [], []
	list_chains_length = []
	list_atoms_in_chains = []

	# Create series of tight-fold loops
#	for i in tfloop_net:
#		temp_chain = []
#		ix = temp_net[0].index([i[0]+temp_net[2], i[1]])
#		tv = [temp_net[0][ix][0] - tfloop[0][0],
#			  temp_net[0][ix][1] - tfloop[0][1]]
#		chain_in_crystal(0, restrict_under, ix, 1, 0, temp_net, sites, temp_chain)
#		for x in tfloop[2:]:
#			temp['x0'] = x[0] + tv[0]
#			temp['x1'] = x[1] + tv[1]
#			temp['x2'] = x[2] + restrict_under
#			temp['vec'] = []
#			if temp['x0'] == size_ground:
#				temp['x0'] = 0
#			if temp['x1'] == size_ground:
#				temp['x1'] = 0
#			if temp['x0'] > size_ground:
#				temp['x0'] = 1
#			if temp['x1'] > size_ground:
#				temp['x1'] = 1
#			if temp['x0'] < 0:
#				temp['x0'] = size_ground - 1
#			if temp['x1'] < 0:
#				temp['x1'] = size_ground - 1
#			sites[temp['x0']][temp['x1']][temp['x2']] = 1
#			temp_chain.append(dict(temp))
#			n += 1
#		ix = temp_net[0].index([i[2]+temp_net[2], i[3]])
#		chain_in_crystal(restrict_under, 0, ix, -1, 0, temp_net, sites, temp_chain)
#		## Create comonomer order
#		chain_num += 1
#		for x in range(len(temp_chain)):
#			temp_chain[x]['chain'] = chain_num
#			if (x%4 == 0 or x%4 == 1):
#				temp_chain[x]['type'] = 1
#			else:
#				temp_chain[x]['type'] = 2
#		loop[0].append(temp_chain)
#		loop[1].append(chain_num)
#		temp_chain[0]['type'] = 3
#		if len(temp_chain)%4 == 0 or len(temp_chain)%4 == 1:
#			temp_chain[-1]['type'] = 3
#		else:
#			temp_chain[-1]['type'] = 4
#		list_chains_length.append(len(temp_chain))
#		list_atoms_in_chains.extend(temp_chain)
		
#		temp_chain = []
#		ix = temp_net[1].index([i[0]+temp_net[3], i[1]])
#		tv = [temp_net[1][ix][0] - tfloop[0][0],
#			  temp_net[1][ix][1] - tfloop[0][1]]
#		chain_in_crystal(height_poly, restrict_above, ix, -1, 1, temp_net, sites, temp_chain)
#		for x in tfloop[2:]:
#			temp['x0'] = x[0] + tv[0]
#			temp['x1'] = x[1] + tv[1]
#			temp['x2'] = restrict_above - x[2] 
#			temp['vec'] = []
#			if temp['x0'] == size_ground:
#				temp['x0'] = 0
#			if temp['x1'] == size_ground:
#				temp['x1'] = 0
#			if temp['x0'] > size_ground:
#				temp['x0'] = 1
#			if temp['x1'] > size_ground:
#				temp['x1'] = 1
#			if temp['x0'] < 0:
#				temp['x0'] = size_ground - 1
#			if temp['x1'] < 0:
#				temp['x1'] = size_ground - 1
#			sites[temp['x0']][temp['x1']][temp['x2']] = 1
#			temp_chain.append(dict(temp))
#			n += 1
#		ix = temp_net[1].index([i[2]+temp_net[3], i[3]])
#		chain_in_crystal(restrict_above, height_poly, ix, 1, 1, temp_net, sites, temp_chain)
#		## Create comonomer order
#		chain_num += 1
#		for x in range(len(temp_chain)):
#			temp_chain[x]['chain'] = chain_num
#			if (x%4 == 0 or x%4 == 1):
#				temp_chain[x]['type'] = 1
#			else:
#				temp_chain[x]['type'] = 2
#		temp_chain[0]['type'] = 3
#		if len(temp_chain)%4 == 0 or len(temp_chain)%4 == 1:
#			temp_chain[-1]['type'] = 3
#		else:
#			temp_chain[-1]['type'] = 4
#		loop[0].append(temp_chain)
#		loop[1].append(chain_num)
#		list_chains_length.append(len(temp_chain))
#		list_atoms_in_chains.extend(temp_chain)
		
	# We begin to grow amouphous from surface of lamellae 
	standby, temp_chain = [], []
	while n <= amo_sites or standby != []:
		# First we randomly choose a chain in list chain in crystal region then begin to
		# generate more atom in the amophous region
		# In here, in order to generate desity equally in difference part of the 
		# polymer we switch between generate above and generate below
		temp_seed = {}
		if standby != []:
			temp_chain.reverse()
			chain_num += 1
			del temp_net[switch1][temp_net[switch1].index(standby[0])]
			del standby[0]
			if switch1 == 1:
				switch = 0
			else:
				switch = 1
		else:
			temp_chain = []
			if len(temp_net[0]) == 0 and len(temp_net[1]) == 0:
				print('out of initial sites')
				break
			if switch == 0:
				l_rand = rd.randint(0, len(temp_net[1])-1)
				standby.append(temp_net[1][l_rand])
				chain_in_crystal(restrict_above, l_rand, temp_net, sites, temp_chain)
				switch, switch1 = 1, 1
			elif switch == 1:
				l_rand = rd.randint(0, len(temp_net[0])-1)
				standby.append(temp_net[0][l_rand])
				chain_in_crystal(restrict_under, l_rand, temp_net, sites, temp_chain)
				switch, switch1 = 0, 0
		temp_seed = temp_chain[-1]
		print('switch ' + str(switch))
		
		count = 0
		while count <= int((height_amophous-1)*3.5):
			relative_site = temp_chain[-7]
			next_site = find_next_site(temp_seed, sites, restrict_under, restrict_above, relative_site)
			if next_site == 0:
				break
			temp_chain.append(next_site)
			count += 1
			temp_seed = temp_chain[-1]
			sites[next_site['x0']][next_site['x1']][next_site['x2']] = 1	
			
			# Check for intercept with crystal region under
			if next_site['x2'] == restrict_under + 1 and count > (height_amophous - 1)/8:
				next_head = find_exist_head_around(next_site, sites)
				if next_head == []:
					continue
				for x in range(len(next_head)):
					check_head = temp_net[0]
					for i in range(len(temp_net[0])):
						if next_head[x]['x0'] == check_head[i][0] and \
						   next_head[x]['x1'] == check_head[i][1]:
							temp_lengh = chain_lengh()
							if switch == 1 and count > temp_lengh[1]:
								for temp in temp_chain[(count*-1):]:
									sites[temp['x0']][temp['x1']][temp['x2']] = 0                                
								del temp_chain[(count*-1):]
								count = -2
								break
							if switch == 1:
								loop.append(temp_chain[(count*-1):])
								print('loop')
								switch = 0
							else:
								bridge.append(temp_chain[(count*-1):])
								print('bridge')
	#							lame_chain_index = copy.deepcopy(temp_net[1][i])
							chain_in_crystal(restrict_under, i, temp_net, sites, temp_chain)
							del temp_net[0][i]
							na = copy.deepcopy(count)
							count = -1
							break
					if count == -1 or count == -2:
						break
			# Check for intercept with crystal region above
			elif next_site['x2'] == restrict_above - 1 and count > (height_amophous - 1)/8:
				next_head = find_exist_head_around(next_site, sites)
				if next_head == []:
					continue
				for x in range(len(next_head)):
					check_head = temp_net[1]
					for i in range(len(temp_net[1])):
						if next_head[x]['x0'] == check_head[i][0] and \
						   next_head[x]['x1'] == check_head[i][1]:
							temp_lengh = chain_lengh()
							if switch == 0 and count > temp_lengh[1]:
								for temp in temp_chain[(count*-1):]:
									sites[temp['x0']][temp['x1']][temp['x2']] = 0                                
								del temp_chain[(count*-1):]
								count = -2
								break
							if switch == 0:
								loop.append(temp_chain[(count*-1):])
								print('loop')
								switch = 1
							else:
								bridge.append(temp_chain[(count*-1):])
								print('bridge')
	#							lame_chain_index = copy.deepcopy(temp_net[1][i])
							chain_in_crystal(restrict_above, i, temp_net, sites, temp_chain)
							del temp_net[1][i]
							na = copy.deepcopy(count)
							count = -1
							break
					if count == -1 or count == -2:
						break
			if count == -1:
				temp_seed = temp_chain[-1]
				count = 0
				n += na
				continue
			elif count == -2:
				count = 0
				continue
		temp_lengh = chain_lengh()
		if count > 0:
			if temp_lengh[0] < count:
				n += temp_lengh[0]
				del temp_chain[((count-temp_lengh[0])*-1):]
				tail.append(temp_chain[(temp_lengh[0]*-1):])
			else:
				n += count
				tail.append(temp_chain[(count*-1):])
			print('tail')
		elif count == 0:
			print('lamellar chain')
			lamellar_chain += 1
		## Create comonomer order
		if standby == []:
			print(chain_num,  len(temp_chain))
			for x in range(len(temp_chain)):
				temp_chain[x]['chain'] = chain_num
				if (x%4 == 0 or x%4 == 1):
					temp_chain[x]['type'] = 1
				else:
					temp_chain[x]['type'] = 2
			list_chains_length.append(len(temp_chain))
			list_atoms_in_chains.extend(temp_chain)

	if len(temp_net[0]) != 0:
			for i in range(len(temp_net[1])):
				temp_chain = []
				chain_in_crystal(restrict_above , i, temp_net, sites, temp_chain)
				chain_num += 1
				lamellar_chain += 1
				## Create comonomer order
				for x in range(len(temp_chain)):
					temp_chain[x]['chain'] = chain_num
					if (x%4 == 0 or x%4 == 1):
						temp_chain[x]['type'] = 1
					else:
						temp_chain[x]['type'] = 2
				list_chains_length.append(len(temp_chain))
				list_atoms_in_chains.extend(temp_chain)
			del temp_net[0][:], temp_net[1][:]

	print(n,   len(list_atoms_in_chains))
	print(len(loop),   len(loop)/(len(loop)+len(tail)+len(bridge)))
	print(len(tail),   len(tail)/(len(loop)+len(tail)+len(bridge)))
	print(len(bridge), len(bridge)/(len(loop)+len(tail)+len(bridge)))
	print(chain_num)
	print(len(list_chains_length))

	# Check atoms which have same position on lacttic
	count = 0
	temp_list1 = []
	for x in range(len(list_atoms_in_chains)):
		for y in range(x + 1, len(list_atoms_in_chains)):
			if x!=y and \
			   list_atoms_in_chains[x]['x0']==list_atoms_in_chains[y]['x0'] and \
			   list_atoms_in_chains[x]['x1']==list_atoms_in_chains[y]['x1'] and \
			   list_atoms_in_chains[x]['x2']==list_atoms_in_chains[y]['x2']:
					count+=1
					temp_list1.append([x,y])
					continue

## Analytic data
	file = open("Analytic.dat", "w")
	file.write("## Generation values\n")
	file.write("\n")
	file.write("switch 0%10i\n" % len(temp_net[0]))
	file.write("switch 1%10i\n" % len(temp_net[1]))
	file.write("%10i chains\n\n" % len(list_chains_length))
	file.write("%10i loops %10.2f\n" % (len(loop), len(loop)/(len(loop)+len(tail)+len(bridge))*100))
	for x in range(len(loop)):
		file.write("%6i" % (len(loop[x])))
	file.write("\n\n")
	file.write("%10i tails %10.2f\n" % (len(tail), len(tail)/(len(loop)+len(tail)+len(bridge))*100))
	for x in range(len(tail)):
		file.write("%6i" % (len(tail[x])))
	file.write("\n\n")
	file.write("%10i bridges %10.2f\n" % (len(bridge), len(bridge)/(len(loop)+len(tail)+len(bridge))*100))
	for x in range(len(bridge)):
		file.write("%6i" % (len(bridge[x])))
	file.write("\n\n")
	file.write("%10i lamellar\n\n" % (lamellar_chain))
	for x in range(len(temp_list1)):
		file.write("%10i%10i\n" % (temp_list1[x][0], temp_list1[x][1]))

## Print to file
	list_atoms_in_chains = np.copy(list_atoms_in_chains)
	real_num_atoms = len(list_atoms_in_chains)

	file = open("ETFE.dat", "w")
	file.write("# Model of semicrystalline ETFE Co-polymer\n")
	file.write("\n")
	file.write("%10i     atoms\n" % (real_num_atoms))
	file.write("%10i     bonds\n" % (real_num_atoms - chain_num))
	file.write("%10i     angles\n" % (real_num_atoms - 2*chain_num))
	file.write("%10i     dihedrals\n" % (real_num_atoms - 3*chain_num))
	file.write("\n")
	file.write("%10i     atom types\n" % 2)
	file.write("%10i     bond types\n" % 3)
	file.write("%10i     angle types\n" % 2)
	file.write("%10i     dihedral types\n" % 3)
	file.write("\n")
	file.write("%10.4f%10.4f xlo xhi\n" % (0.0, size_ground*b_length/np.sqrt(2)))
	file.write("%10.4f%10.4f ylo yhi\n" % (0.0, size_ground*b_length/np.sqrt(2)))
	file.write("%10.4f%10.4f zlo zhi\n" % (0.0, (height_poly+1)*b_length/np.sqrt(2)))
	file.write("\n")
	file.write("Masses\n")
	file.write("\n")
	file.write("%10i %14.2f\n" % (1, massCH2))
	file.write("%10i %14.2f\n" % (2, massCF2))


	file.write("\n")
	file.write("Atoms\n")
	file.write("\n")
	count = 0
	for x in list_atoms_in_chains:
		xx = dict(x)
		xx['x0'] = b_length/np.sqrt(2)*xx['x0']
		xx['x1'] = b_length/np.sqrt(2)*xx['x1']
		xx['x2'] = b_length/np.sqrt(2)*xx['x2']
		count += 1 
		file.write("%10i%10i%10i%10.4f%10.4f%10.4f\n" % (count, int(xx['chain']), int(xx['type']), xx['x0'], xx['x1'], xx['x2']))


	file.write("\n")
	file.write("Bonds \n")
	file.write("\n")
	count, atom_1, atom_2 = 1, 1, 2
	for x in range(chain_num):
		c_length = list_chains_length[x]
		for y in range(c_length - 1):
			if y%4 == 0:
				file.write("%10i%10i%10i%10i\n" % (count, 1, atom_1, atom_2))
			elif (y%4 == 1) or (y%4 == 3):
				file.write("%10i%10i%10i%10i\n" % (count, 2, atom_1, atom_2))
			else:
				file.write("%10i%10i%10i%10i\n" % (count, 3, atom_1, atom_2))
			count += 1
			atom_1 += 1
			atom_2 += 1 
		atom_1 += 1
		atom_2 += 1


	file.write("\n")
	file.write("Angles \n")
	file.write("\n")
	count, atom_1, atom_2, atom_3 = 1, 1, 2, 3
	for x in range(chain_num):
		c_length = list_chains_length[x]
		for y in range(c_length - 2):
			file.write("%10i%10i%10i%10i%10i\n" % (count, y%2+1, atom_1, atom_2, atom_3))
			count += 1
			atom_1 += 1
			atom_2 += 1 
			atom_3 += 1
		atom_1 += 2
		atom_2 += 2
		atom_3 += 2


	file.write("\n")
	file.write("Dihedrals \n")
	file.write("\n")
	count, atom_1, atom_2, atom_3, atom_4 = 1, 1, 2, 3, 4
	for x in range(chain_num):
		c_length = list_chains_length[x]
		for y in range(c_length - 3):
			if (y%4 == 0) or (y%4 == 2):
				file.write("%10i%10i%10i%10i%10i%10i\n" % (count, 1, atom_1, atom_2, atom_3, atom_4))
			elif (y%4 == 1):
				file.write("%10i%10i%10i%10i%10i%10i\n" % (count, 2, atom_1, atom_2, atom_3, atom_4))
			else:
				file.write("%10i%10i%10i%10i%10i%10i\n" % (count, 3, atom_1, atom_2, atom_3, atom_4))
			count += 1
			atom_1 += 1
			atom_2 += 1
			atom_3 += 1
			atom_4 += 1
		atom_1 += 3
		atom_2 += 3
		atom_3 += 3
		atom_4 += 3

if __name__== "__main__":
	main()

end_time = time.time()

print(end_time - start_time)


