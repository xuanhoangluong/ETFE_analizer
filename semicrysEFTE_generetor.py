import time
import math
import random as rd
import numpy as np
from numpy import linalg as LA


# In this code I am generating a semicrystalline copolymer (ETFE). Base on suggetion 
# about lattic in https://icme.hpc.msstate.edu/mediawiki/index.php/Amorphous_Polymer_Generator
# and using SRW(self random walk) for amorphous region and parameter for crystal region form 
# balijepalli1998 and Flourinated Polymer(book) (citation would be include in my thesis). 
# Structure set up was confirmed by Lee2011 and reuse by kim2014.
# simulation will employed coarse grain simulation for CH2 and CF2
start_time = time.time()
# initial data
file_out_name = 'EFTE_NgVB_upgmodel'
density = 0.081 # density was cacualted from density of ETFE (1.7g/cm3) 
b_length = 1.53 # bond length was also take from balijepalli1998
massCH2 = 14
massCF2 = 50
cube_a = b_length*np.sqrt(2) # length of the cube fcc
# print(cube_a)

size_ground = 36 # choose size for x and y axis 
size_lamex = int(size_ground/3) # distance between chains in lamellae along x axis
size_lamey = int(size_ground/2)
print(size_lamex)
print(size_lamey)
crystallinity = 0.35 # percentage of unit in crystals region

n_oc_sites = 20000 # total nuber of unit in sample
n_lame = int(n_oc_sites*crystallinity) # number of unit in crystal region
height_lame = int(n_lame/(size_lamex*size_lamey)) # length of lamellae along z axis
# print(n_lame)
height_lame = (int(height_lame/2))*2
print(height_lame)
t_sites = n_oc_sites/density 
height_poly = int((t_sites/4)/(size_ground*size_ground)) + 1 # from data above we obtain the height of sample
# print(height_poly)


# This function take two atom (a dict in python) and return an vector from atomb to atoma
def create_vector(atoma, atomb):
	atom1 = dict(atoma)
	atom2 = dict(atomb)
	# create first vector
	a = np.zeros(3)
	aa = np.zeros(3)
	va = np.zeros(3)
	if (atom1['x0']<5 and atom2['x0']>10):
		atom2['x0'] = atom2['x0'] - size_ground
	if (atom2['x0']<5 and atom1['x0']>10):
		atom2['x0'] = atom2['x0'] + size_ground
	if (atom1['x1']<5 and atom2['x1']>10):
		atom2['x1'] = atom2['x1'] - size_ground
	if (atom2['x1']<5 and atom1['x1']>10):
		atom2['x1'] = atom2['x1'] + size_ground

	if atom1['x3']==0 :
		a[0] = 0
		a[1] = 0
		a[2] = 0
	elif atom1['x3']==1:
		a[0] = 0.5
		a[1] = 0.5
		a[2] = 0
	elif atom1['x3']==2:
		a[0] = 0.5
		a[1] = 0
		a[2] = 0.5
	elif atom1['x3']==3:
		a[0] = 0
		a[1] = 0.5
		a[2] = 0.5

	if atom2['x3']==0 :
		aa[0] = 0
		aa[1] = 0
		aa[2] = 0
	elif atom2['x3']==1:
		aa[0] = 0.5
		aa[1] = 0.5
		aa[2] = 0
	elif atom2['x3']==2:
		aa[0] = 0.5
		aa[1] = 0
		aa[2] = 0.5
	elif atom2['x3']==3:
		aa[0] = 0
		aa[1] = 0.5
		aa[2] = 0.5

	aa[0] += aa[0] + atom2['x0'] - atom1['x0']
	aa[1] += aa[1] + atom2['x1'] - atom1['x1']
	aa[2] += aa[2] + atom2['x2'] - atom1['x2']

	va[0] = a[0] - aa[0]
	va[1] = a[1] - aa[1]
	va[2] = a[2] - aa[2]

	return va

# This function take an atom(a dict in python) and find and return all empty site around that site in the net
def find_empty_sites_around(curr_site, sites):
	# d = np.zeros((3, 3, 3, 4), dtype=np.int)
	dx = np.zeros(3, dtype=np.int)
	dy = np.zeros(3, dtype=np.int)
	dz = np.zeros(3, dtype=np.int)
	# m = np.zeros(4, dtype=np.int)
	a = np.zeros(3)
	aa = np.zeros(3)
	va = np.zeros(3)
	empty_sites_around = []

	m = curr_site['x3']
	dx[1] = curr_site['x0']
	dy[1] = curr_site['x1']
	dz[1] = curr_site['x2']
	dx[2] = curr_site['x0']+1
	dy[2] = curr_site['x1']+1
	dz[2] = curr_site['x2']+1
	dx[0] = curr_site['x0']-1
	dy[0] = curr_site['x1']-1
	dz[0] = curr_site['x2']-1

	if dx[2]==size_ground:
		dx[2] = 0
	if dy[2]==size_ground:
		dy[2] = 0
	if dz[2]==height_poly:
		dz[2] = 0
	if dx[0] < 0:
		dx[0] = size_ground - 1 
	if dy[0] < 0:
		dy[0] = size_ground - 1
	if dz[0] < 0:
		dz[0] = height_poly - 1

	if m==0:
		a[0] = 0
		a[1] = 0
		a[2] = 0
	elif m==1:
		a[0] = 0.5
		a[1] = 0.5
		a[2] = 0
	elif m==2:
		a[0] = 0.5
		a[1] = 0
		a[2] = 0.5
	elif m==3:
		a[0] = 0
		a[1] = 0.5
		a[2] = 0.5

	for mm in range(4):
		for x in range(3):
			for y in range(3):
				for z in range(3):
					if mm==0:
						aa[0] = 0
						aa[1] = 0
						aa[2] = 0
					elif mm==1:
						aa[0] = 0.5
						aa[1] = 0.5
						aa[2] = 0
					elif mm==2:
						aa[0] = 0.5
						aa[1] = 0
						aa[2] = 0.5
					elif mm==3:
						aa[0] = 0
						aa[1] = 0.5
						aa[2] = 0.5
					aa[0] += x - 1	
					aa[1] += y - 1
					aa[2] += z - 1
					va[0] = aa[0] - a[0]
					va[1] = aa[1] - a[1]
					va[2] = aa[2] - a[2]
					# print(aa)
					va_mag = LA.norm(va)
					if 0.5<va_mag<=np.sqrt(2)/2+0.01:
						if sites[dx[x]][dy[y]][dz[z]][mm] == 0:
							temp = {}
							temp['x0'] = dx[x]
							temp['x1'] = dy[y]
							temp['x2'] = dz[z]
							temp['x3'] = mm
							temp['head'] = 0
							temp['type'] = curr_site['type']
							temp['vec'] = np.copy(va)
							empty_sites_around.append(temp)

	return empty_sites_around

# This function take an atom then from that site find an random new empty site which satify condition
# of possible fo find a new site which again satify the condition of possible find a a new empty site
# (there is another condition needed to be satify. It is the angle between 2 vector of three recent atom
# have to be 60 degree)
def find_next_site(curr_site, sites, restrict_under, restrict_above, relative_site):
	empty_sites_around = find_empty_sites_around(curr_site, sites)
	found = 0
	next_chain = 0

	distance_check_curr = LA.norm(create_vector(curr_site, relative_site))

	# print(restrict_under)
	# print(restrict_above)
	active_check = 1
	while found != 1:
		if empty_sites_around == []:
			next_chain = 1
			# return next_chain
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
			# print(angle)
			if (40<angle<70) and (restrict_above>next_site['x2']>restrict_under):
				distance_check_next = LA.norm(create_vector(next_site, relative_site))
				if active_check == 1:
					if distance_check_next >= distance_check_curr:
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
	# d = np.zeros((3, 3, 3, 4), dtype=np.int)
	dx = np.zeros(3, dtype=np.int)
	dy = np.zeros(3, dtype=np.int)
	dz = np.zeros(3, dtype=np.int)
	# m = np.zeros(4, dtype=np.int)
	a = np.zeros(3)
	aa = np.zeros(3)
	va = np.zeros(3)
	empty_head_aroud = []

	m = curr_site['x3']
	dx[1] = curr_site['x0']
	dy[1] = curr_site['x1']
	dz[1] = curr_site['x2']
	dx[2] = curr_site['x0']+1
	dy[2] = curr_site['x1']+1
	dz[2] = curr_site['x2']+1
	dx[0] = curr_site['x0']-1
	dy[0] = curr_site['x1']-1
	dz[0] = curr_site['x2']-1

	if dx[2]==size_ground:
		dx[2] = 0
	if dy[2]==size_ground:
		dy[2] = 0
	if dz[2]==height_poly:
		dz[2] = 0
	if dx[0] < 0:
		dx[0] = size_ground - 1 
	if dy[0] < 0:
		dy[0] = size_ground - 1
	if dz[0] < 0:
		dz[0] = height_poly - 1

	if m==0:
		a[0] = 0
		a[1] = 0
		a[2] = 0
	elif m==1:
		a[0] = 0.5
		a[1] = 0.5
		a[2] = 0
	elif m==2:
		a[0] = 0.5
		a[1] = 0
		a[2] = 0.5
	elif m==3:
		a[0] = 0
		a[1] = 0.5
		a[2] = 0.5

	for mm in range(4):
		for x in range(3):
			for y in range(3):
				for z in range(3):
					if mm==0:
						aa[0] = 0
						aa[1] = 0
						aa[2] = 0
					elif mm==1:
						aa[0] = 0.5
						aa[1] = 0.5
						aa[2] = 0
					elif mm==2:
						aa[0] = 0.5
						aa[1] = 0
						aa[2] = 0.5
					elif mm==3:
						aa[0] = 0
						aa[1] = 0.5
						aa[2] = 0.5
					aa[0] += x - 1	
					aa[1] += y - 1
					aa[2] += z - 1
					va[0] = aa[0] - a[0]
					va[1] = aa[1] - a[1]
					va[2] = aa[2] - a[2]
					# print(aa)
					va_mag = LA.norm(va)
					if 0.5<va_mag<=np.sqrt(2)/2+0.01:
						if sites[dx[x]][dy[y]][dz[z]][mm] == 2:
							temp = {}
							temp['x0'] = dx[x]
							temp['x1'] = dy[y]
							temp['x2'] = dz[z]
							temp['x3'] = mm
							# temp['chain'] = curr_site['chain']
							temp['type'] = curr_site['type']
							temp['vec'] = np.copy(va)
							empty_head_aroud.append(temp)

	return empty_head_aroud

# Take all surface above atom of crystall region
def extract_surface_above(atoms_list, height_lame):
	surf_above = []
	num_st = int(len(atoms_list)/height_lame)
	temp = 0
	for x in range(0, num_st):
		surf_above.append(temp)
		temp += height_lame
	return surf_above
 
# Take all surface under atom af crystal region
def extract_surface_under(atoms_list, height_lame):
	surf_under = []
	num_st = int(len(atoms_list)/height_lame)
	temp = height_lame - 1
	for x in range(0, num_st):
		surf_under.append(temp)
		temp += height_lame
	return surf_under	

def main():
	# In here we will create a straight chain then duplicate it along x and y axis
	# define sites as 4 dimention matrix of lacttic and atoms_list to contain all created atoms
	sites = np.zeros((size_ground,size_ground,height_poly,4), dtype=np.int)
	# sites hold all unit on lattic in 4 dimention matrix
	# Define where to start construct lamellae this first point, we build from here upward
	start_lame = int(height_poly - height_lame/4)
	temp = {}
	temp['x0'] = 0
	temp['x1'] = 0
	temp['x2'] = start_lame
	temp['x3'] = 0
	# temp['lame'] = 1
	temp['type'] = 1
	temp['vec'] = []
	temp['head'] = 1

	# restrict ward that lamellae ocupying later when we initial random amouphous, it is gonna stay out of this region
	# restrict_above = start_lame
	# restrict_under = height_lame - (height_poly - start_lame)


	sites[temp['x0']][temp['x1']][temp['x2']][temp['x3']] = 1
	# print(temp)
	atoms_list = []
	atoms_list.append(dict(temp))
	temp['head'] = 0
	type_arr = 1
	for x in range(height_lame - 1):
		if x == height_lame - 2:
			temp['head'] = 1
		last_atom = atoms_list[-1]
		if type_arr == 2:
			temp['type'] = last_atom['type'] + 1
			type_arr = 1
		elif type_arr <= 1:
			temp['type'] = last_atom['type']
			type_arr += 1
		if temp['type'] > 2:
			temp['type'] = 1

		if last_atom['x3'] == 0:
			temp['x0'] = last_atom['x0'] - 1
			temp['x1'] = last_atom['x1']
			temp['x2'] = last_atom['x2']
			temp['x3'] = 2
			# temp['lame'] = 1
			temp['vec'] = []
		elif last_atom['x3'] == 2:
			temp['x0'] = last_atom['x0']
			temp['x1'] = last_atom['x1']
			temp['x2'] = last_atom['x2'] + 1
			temp['x3'] = 1
			# temp['lame'] = 1
			temp['vec'] = []
		elif last_atom['x3'] == 1:
			temp['x0'] = last_atom['x0']
			temp['x1'] = last_atom['x1']
			temp['x2'] = last_atom['x2']
			temp['x3'] = 3
			# temp['lame'] = 1
			temp['vec'] = []
		elif last_atom['x3'] == 3:
			temp['x0'] = last_atom['x0']
			temp['x1'] = last_atom['x1'] + 1
			temp['x2'] = last_atom['x2'] + 1
			temp['x3'] = 0
			# temp['lame'] = 1
			temp['vec'] = []
		
		if temp['x2'] == height_poly:
			temp['x2'] = 0
		elif temp['x2'] < 0:
			temp['x2'] = height_poly - 1
		if temp['x1'] == size_ground:
			temp['x1'] = 0
		elif temp['x1'] < 0:
			temp['x1'] = size_ground - 1
		if temp['x0'] == size_ground:
			temp['x0'] = 0
		elif temp['x0'] < 0:
			temp['x0'] = size_ground - 1
		# print(temp)
		sites[temp['x0']][temp['x1']][temp['x2']][temp['x3']] = 1

		atoms_list.append(dict(temp))
	temp_list1 = np.copy(atoms_list)
	# print(temp_list1[0])

	# duplicate lame
	for x in range(size_lamex):
		temp_list2 = np.copy(temp_list1)
		for y in range(size_lamey - 1):
			temp_list3 = []
			for z in range(height_lame):
				temp = dict(temp_list2[z])
				# temp['x0'] += 3
				temp['x1'] += 2

				if temp['x1'] >= size_ground:
					temp['x1'] = temp['x1'] - size_ground
				elif temp['x1'] < 0:
					temp['x1'] = temp['x1'] + size_ground 
				if temp['x0'] >= size_ground:
					temp['x0'] = temp['x0'] - size_ground
				elif temp['x0'] < 0:
					temp['x0'] = temp['x0'] + size_ground
				temp_list3.append(dict(temp))

				sites[temp['x0']][temp['x1']][temp['x2']][temp['x3']] = 1
				atoms_list.append(dict(temp))
			temp_list2 = np.copy(temp_list3)

		temp_list2 = []
		for z in range(height_lame):
			temp = dict(temp_list1[z])
			temp['x0'] += 3
			if temp['x1'] >= size_ground:
				temp['x1'] = temp['x1'] - size_ground
			elif temp['x1'] < 0:
				temp['x1'] = temp['x1'] + size_ground 
			if temp['x0'] >= size_ground:
				temp['x0'] = temp['x0'] - size_ground
			elif temp['x0'] < 0:
				temp['x0'] = temp['x0'] + size_ground


			temp_list2.append(dict(temp))
			sites[temp['x0']][temp['x1']][temp['x2']][temp['x3']] = 1
			atoms_list.append(dict(temp))
		temp_list1 = np.copy(temp_list2)
		
	# Delete atoms which have same position on lacttic
	count = 0
	temp_list1 = []
	sig_continue = 0
	for x in range(len(atoms_list)):
		for y in range(x + 1, len(atoms_list)):
			if x!=y:
				# check
				if atoms_list[x]['x0']==atoms_list[y]['x0']:
					if atoms_list[x]['x1']==atoms_list[y]['x1']:
						if atoms_list[x]['x2']==atoms_list[y]['x2']:
							if atoms_list[x]['x3']==atoms_list[y]['x3']:
								sig_continue = 1
								count+=1
								break
		if sig_continue == 1:
			sig_continue = 0
			continue
		temp_list1.append(atoms_list[x])		

	print(count)
	atoms_list = list(temp_list1)
	# temp_list1 = list(atoms_list)
	temp_list2 = []
	print(len(temp_list1))

	# Section 2: Growing amouphous region from surface
	amo_sites = n_oc_sites - len(atoms_list)
	# Growing amouphous
	# We begin to grow amouphous from surface of lamellae 
	# extract surface of lamellae
	num_st_line_in_lame = size_lamex*size_lamey
	
	surf_above = extract_surface_above(atoms_list, height_lame)
	# for x in surf_above:
	# 	print(atoms_list[x])
	p_rand = 0
	for x in range(num_st_line_in_lame):
		surf_above.append(p_rand)
		temp = atoms_list[p_rand]
		sites[temp['x0']][temp['x1']][temp['x2']][temp['x3']] = 2
		# print(temp_list1[p_rand])
		p_rand += height_lame

	surf_under = extract_surface_under(atoms_list, height_lame)
	# for x in surf_under:
	# 	print(atoms_list[x])
	p_rand = height_lame - 1
	for x in range(num_st_line_in_lame):
		surf_under.append(p_rand)
		temp = atoms_list[p_rand]
		sites[temp['x0']][temp['x1']][temp['x2']][temp['x3']] = 2
		# print(temp_list1[p_rand])
		p_rand += height_lame


	# generate initial sites in amorphous to ensure the desity is constant during 
	# sampling process
	temp_list1 = list(atoms_list)
	# print(temp_list1)
	temp_list2 = []
	# initial units on lacttic
	# rand_atoms = []
	restrict_under = atoms_list[-1]['x2']
	restrict_above = atoms_list[1]['x2']
	# print(restrict_above)
	# print(restrict_under)
	x = 0
	while x < amo_sites:
		temp = {}
		temp['x0'] = rd.randint(0,size_ground - 1)
		temp['x1'] = rd.randint(0,size_ground - 1)
		temp['x2'] = rd.randint(restrict_under,restrict_above)
		temp['x3'] = rd.randint(0,3)
		# temp['lame'] = 0
		temp['type'] = 0
		temp['vec'] = []
		temp['head'] = 0
		if sites[temp['x0']][temp['x1']][temp['x2']][temp['x3']] == 0:
			sites[temp['x0']][temp['x1']][temp['x2']][temp['x3']] = 1
			# temp_list1.append(dict(temp))
			temp_list2.append(dict(temp))
			x += 1


	# generate "bridge", "tail" type amorphous
	list_atoms_in_chains = []
	restrict_under = atoms_list[-1]['x2']
	restrict_above = atoms_list[1]['x2']
	num_bridge = int(num_st_line_in_lame/4)
	st_count = 0
	tail_count = 0
	chain_num = 0
	list_chains_length = []
	bridge_count = 0
	switch = 0
	# Generate more atom till break the condition
	while st_count<num_bridge and bridge_count < int(amo_sites):
		chain_num += 1
		temp_chain = []
		l_rand = rd.randint(0,num_st_line_in_lame)
		# First we random a chain in list chain in crystal region then begin to
		# generate more atom from above or below
		# In here, in order to generate desity equally in difference part of the 
		# polymer we switch between generate above and generate below
		if switch == 0:
			atoms_list[l_rand*height_lame]['head'] = 0
			atoms_list[l_rand*height_lame + height_lame-1]['head'] = 0
			temp_chain.extend(list(atoms_list[l_rand*height_lame:l_rand*height_lame+height_lame]))
			del atoms_list[l_rand*height_lame:l_rand*height_lame+height_lame]
			surf_under = extract_surface_under(atoms_list, height_lame)
			surf_above = extract_surface_above(atoms_list, height_lame)
			num_st_line_in_lame -= 1
			# print(temp_chain)
			atom1 = temp_chain[-1]
			atom2 = temp_chain[-2]
			atom1['vec'] = create_vector(atom1, atom2)
			switch = 1
			# print(atom1['vec'])
		elif switch == 1:
			atoms_list[l_rand*height_lame]['head'] = 0
			atoms_list[l_rand*height_lame + height_lame-1]['head'] = 0
			# temp_chain.extend(atoms_list[l_rand*height_lame:l_rand*height_lame+height_lame])
			# for x in range(0,height_lame):
			# 	temp_chain.append(atoms_list[l_rand*height_lame+height_lame-x-1])
			temp_chain.extend(list(reversed(atoms_list[l_rand*height_lame:l_rand*height_lame+height_lame])))
			del atoms_list[l_rand*height_lame:l_rand*height_lame+height_lame]
			surf_under = extract_surface_under(atoms_list, height_lame)
			surf_above = extract_surface_above(atoms_list, height_lame)
			num_st_line_in_lame -= 1
			# print(temp_chain)
			atom1 = temp_chain[-1]
			atom2 = temp_chain[-2]
			atom1['vec'] = create_vector(atom1, atom2)
			switch = 0

		count = 0
		sig_break = 0
		# Once we choose inital chain in crystal region. We begin to create more atoms
		# follow the proceduce to create amouphous with restriction by surface of crystal
		# region. We the intercept the surface of crystal region, the code will check for
		# any possible link to a chain in crystal region, once found, we adopt it in to our chain
		# and repeat the proceduce. The chain will end when running from the surface for 500 step
		# and could'nt find any intercept with a new chain in crystal region. 
		while count<500:
			if bridge_count >= int(amo_sites):
				break
			# We use relative_site as a reference which every atom newly create much be futher
			# from that site in compare with previous atom.
			relative_site = temp_chain[-5]
			next_site = find_next_site(atom1, sites, restrict_under, restrict_above, relative_site)
			if next_site == 0:
				print('terminated before end')
				# st_count += 0
				break
			
			temp_chain.append(next_site)
			count += 1
			temp_list1.append(next_site)
			bridge_count += 1
			atom1 = temp_chain[-1]
			sites[next_site['x0']][next_site['x1']][next_site['x2']][next_site['x3']] = 1	
			
			# rand_atoms = rd.randint(0, len(temp_list2))
			if temp_list2 != []:
				sites[temp_list2[0]['x0']][temp_list2[0]['x1']][temp_list2[0]['x2']][temp_list2[0]['x3']] = 0
				del temp_list2[0]
			# Check for intercept with crystal region under
			if next_site['x2']==restrict_under + 1 and count>100:
				next_head = []
				next_head = find_exist_head_around(next_site, sites)
				if next_head == []:
					continue
				next_head = next_head[0]
				# print(next_head)

				for x in surf_under:
					check_head = atoms_list[x]
					# print((check_head))
					if next_head['x0'] == check_head['x0']:
						if next_head['x1'] == check_head['x1']:
							if next_head['x2'] == check_head['x2']:
								if next_head['x3'] == check_head['x3']:
									va1 = create_vector(atoms_list[x-1], atoms_list[x])
									vb1 = next_head['vec']
									angle = math.acos((np.dot(va1, vb1))/(LA.norm(va1)*LA.norm(vb1)))
									angle = round(angle*180/math.pi)
									print(angle)
									# for y in range(0,height_lame):
									# 	temp_chain.append(atoms_list[x-y])
									temp_chain.extend(list(reversed(atoms_list[x - height_lame+1:x+1])))
									del atoms_list[x - height_lame+1:x+1]
									num_st_line_in_lame -= 1
									surf_under = extract_surface_under(atoms_list, height_lame)
									surf_above = extract_surface_above(atoms_list, height_lame)
									temp_chain[-1]['vec'] = create_vector(temp_chain[-1], temp_chain[-2])
									# print(temp_chain[-1])
									atom1 = temp_chain[-1]
									count = 0
									st_count +=1
									sig_break = 0
									break
			# Check for intercept with crystal region under
			if next_site['x2']==restrict_above - 1 and count>100:
				next_head = []
				next_head = find_exist_head_around(next_site, sites)
				if next_head == []:
					continue
				next_head = next_head[0]
				# print(next_head)

				for x in surf_above:
					check_head = atoms_list[x]
					# print((check_head))
					if next_head['x0'] == check_head['x0']:
						if next_head['x1'] == check_head['x1']:
							if next_head['x2'] == check_head['x2']:
								if next_head['x3'] == check_head['x3']:
									va1 = create_vector(atoms_list[x+1], atoms_list[x])
									vb1 = next_head['vec']
									angle = math.acos((np.dot(va1, vb1))/(LA.norm(va1)*LA.norm(vb1)))
									angle = round(angle*180/math.pi)
									print(angle)
									# for y in range(0,height_lame):
									# 	temp_chain.append(atoms_list[x+y])
									temp_chain.extend(list(atoms_list[x:x+height_lame]))
									del atoms_list[x:x+height_lame]
									num_st_line_in_lame -= 1
									surf_under = extract_surface_under(atoms_list, height_lame)
									surf_above = extract_surface_above(atoms_list, height_lame)
									temp_chain[-1]['vec'] = create_vector(temp_chain[-1], temp_chain[-2])
									# print(temp_chain[-1])
									atom1 = temp_chain[-1]
									count = 0
									st_count += 1
									sig_break = 0
									break
			if count == 500:
				tail_count +=1
			if sig_break == 1:
				break
		for x in range(len(temp_chain)):
			temp_chain[x]['chain'] = chain_num
		list_chains_length.append(len(temp_chain))
		list_atoms_in_chains.extend(temp_chain)
						
	print(st_count)
	print(chain_num)
	print(tail_count)
	print(atoms_list[-1])
	# print(np.copy(list_atoms_in_chains))
	print(len(list_atoms_in_chains))
	chain_num += 1
	count = 0
	for x in range(len(atoms_list)):
		atoms_list[x]['chain'] = chain_num
		count += 1
		if count == height_lame:
			count = 0
			chain_num += 1
	chain_num -= 1

	list_atoms_in_chains.extend(atoms_list)
	st_left = int(len(atoms_list)/height_lame)
	if (len(atoms_list)%height_lame) == 0:
		for x in range(st_left):
			list_chains_length.append(height_lame)
	else:
		print('string length are not correct')

	type_arr = 1

## create dual monome order
	count, a, modul = 1, 0, 0
	for x in range(len(list_atoms_in_chains)):
		if list_atoms_in_chains[x]['chain'] > count:
			a += list_chains_length[count-1]
			count += 1
			modul = a%4
		if ((x-modul)%4 == 0) or ((x-modul)%4 == 1):
			list_atoms_in_chains[x]['type'] = 1
		else:
			list_atoms_in_chains[x]['type'] = 2

	print(chain_num)
	print(len(list_chains_length))
	# print to file

	# file = open("lame.dat", "w")

	# list_atoms_in_chains = np.copy(temp_list1)
	list_atoms_in_chains = np.copy(list_atoms_in_chains)
	# for x in range(0, len(list_atoms_in_chains)):
	# 	print(list_atoms_in_chains[x])
	# print(list_atoms_in_chains)
	real_num_atoms = len(list_atoms_in_chains)

	file = open("ETFE.dat", "w")
	file.write("# Model of semicrystalline Polymer copyright NguyenVietBac\n")
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
	file.write("%10.4f%10.4f xlo xhi\n" % (0.0, size_ground*b_length*np.sqrt(2)))
	file.write("%10.4f%10.4f ylo yhi\n" % (0.0, size_ground*b_length*np.sqrt(2)))
	file.write("%10.4f%10.4f zlo zhi\n" % (0.0, height_poly*b_length*np.sqrt(2)))
	file.write("\n")
	file.write("Masses\n")
	file.write("\n")
	file.write("%10i %14.2f\n" % (1, massCH2))
	file.write("%10i %14.2f\n" % (2, massCF2))


	file.write("\n")
	file.write("Atoms\n")
	file.write("\n")
	a = np.zeros(3)
	count = 0
	for x in list_atoms_in_chains:
		xx = dict(x)
		# print(type(x))
		if type(x) == "numpy.ndarray":
			print(numpy.ndarray)
		if xx['x3'] == 0:
			a[0] = 0
			a[1] = 0
			a[2] = 0
		elif xx['x3'] == 1:
			a[0] = 0.5
			a[1] = 0.5
			a[2] = 0
		elif xx['x3'] == 2:
			a[0] = 0.5
			a[1] = 0
			a[2] = 0.5
		elif xx['x3'] == 3:
			a[0] = 0
			a[1] = 0.5
			a[2] = 0.5

		xx['x0'] = b_length*np.sqrt(2)*(xx['x0'] + a[0])
		xx['x1'] = b_length*np.sqrt(2)*(xx['x1'] + a[1])
		xx['x2'] = b_length*np.sqrt(2)*(xx['x2'] + a[2])
		count += 1 
		file.write("%10i%10i%10i%10.4f%10.4f%10.4f\n" % (count, int(xx['chain']), int(xx['type']), xx['x0'], xx['x1'], xx['x2']))
		print(count, int(xx['chain']), int(xx['type']))


	file.write("\n")
	file.write("Bonds \n")
	file.write("\n")
	count = 1
	atom_1 = 1
	atom_2 = 2
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
	count = 1
	atom_1 = 1
	atom_2 = 2
	atom_3 = 3
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
	count = 1
	atom_1 = 1
	atom_2 = 2
	atom_3 = 3
	atom_4 = 4

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


