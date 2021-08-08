"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 26 May 2021
Description :
Input :
Output :
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


# ================================ Imports ================================ #
from libc.math cimport sqrt

import numpy as np
cimport numpy as np


NP_DTYPE_I = np.int32
NP_DTYPE_F = np.float64

ctypedef np.int32_t DTYPE_i
ctypedef np.float64_t DTYPE_f


cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function


def cy_cosine(np.ndarray[np.float64_t] x, np.ndarray[np.float64_t] y):
	cdef double xx=0.0
	cdef double yy=0.0
	cdef double xy=0.0
	# cdef Py_ssize_t i	
	cdef int arr_size = x.shape[0]
	cdef int i

	for i in range(arr_size):
		xx+=x[i]*x[i]
		yy+=y[i]*y[i]
		xy+=x[i]*y[i]
	return 1.0-xy/sqrt(xx*yy)

def value_propagation(np.ndarray[DTYPE_f, ndim=2] time_index, np.ndarray[DTYPE_f, ndim=2] nodes, np.ndarray[DTYPE_f, ndim=2] nbr, np.ndarray[DTYPE_f, ndim=2] fg_soa, np.ndarray[DTYPE_f, ndim=2] rg_tab, np.ndarray[DTYPE_f, ndim=1] sorted_time, DTYPE_i MAX_ITER, np.ndarray[DTYPE_f, ndim=1] fg_index, DTYPE_f Discount, np.ndarray[DTYPE_f, ndim=1] conv_itr):
	
	# -------- Typecast
	cdef int[:] sorted_time_int = np.array(sorted_time, dtype=NP_DTYPE_I)
	cdef int[:] fg_index_int = np.array(fg_index, dtype=NP_DTYPE_I)
	cdef int[:,:] nodes_int = np.array(nodes, dtype=NP_DTYPE_I)
	cdef int[:,:] nbr_int = np.array(nbr, dtype=NP_DTYPE_I)
	cdef double[:,:] fg_tab_c = np.array(fg_soa, dtype=NP_DTYPE_F)
	cdef double[:,:] fg_tab_old = np.array(fg_soa, dtype=NP_DTYPE_F)
	cdef double[:,:] rg_tab_c = np.array(rg_tab, dtype=NP_DTYPE_F)
	cdef double[:] conv_itr_c = np.array(conv_itr, dtype=NP_DTYPE_F)

	# --------- indexes
	cdef int node_size = nodes.shape[0]
	cdef int s_max = rg_tab.shape[0]
	cdef int a_max = rg_tab.shape[1]
	cdef int fg_index_size = fg_index_int.shape[0] 
	cdef int fg_tab_size = fg_soa.shape[0] 

	# ---- Value Prop
	cdef int i, j, k, l, m
	cdef int sorted_time_size = sorted_time.shape[0]
	cdef int nbr_size = nbr.shape[1]
	cdef int node_id, s, a, sp, nb_node_id, nbr_count
	cdef float next_state_sum, next_action_max, mx

	# ---- Convergence
	cdef float avg_diff, diff


	# --- Convergence Loop
	for i in range(MAX_ITER):
		for j in range(sorted_time_size):
			node_id = sorted_time_int[j]
			if node_id != -1:			
				s = nodes_int[node_id, 0]
				# a = nodes_int[node_id, 1]
				for a in range(a_max):
					# ------ next state
					nbr_count = 0
					next_state_sum = 0
					for k in range(nbr_size):
						nb_node_id = nbr_int[node_id][k]
						if nb_node_id != -1:
							sp = nodes_int[nb_node_id][0]
							# --- find max
							next_action_max = -10000.0
							for ap in range(a_max):
								if next_action_max < fg_tab_c[nb_node_id][ap]:
									next_action_max = fg_tab_c[nb_node_id][ap]
							next_state_sum += next_action_max
							nbr_count += 1
					if nbr_count != 0:
						next_state_sum = next_state_sum / nbr_count				
						# fg_tab_c[node_id][a] = rg_tab_c[s][a] + Discount *next_state_sum
						fg_tab_c[node_id][a] = rg_tab_c[node_id][a] + Discount *next_state_sum

		# ---- check convergence
		avg_diff = 0.0
		for m in range(fg_tab_size):
			for a in range(a_max):
				diff = fg_tab_c[m][a] - fg_tab_old[m][a]
				if diff < 0:
					diff = -1 * diff
				avg_diff += diff
		avg_diff = avg_diff / (fg_tab_size*a_max)
		conv_itr_c[i] = avg_diff		

		# -- update the old
		for m in range(fg_tab_size):
			for a in range(a_max):
				fg_tab_old[m][a] = fg_tab_c[m][a]

		# convergence condition
		if avg_diff < 0.001:
			break


	# a1 = np.array([1.0, 2.0, 3.0, 4.0])
	# a2 = np.array([4.0, 5.0, 6.0, 7.0])
	# print("wala1")
	# print(cy_cosine(a1, a2))
	# exit()

	# ------ Create train_fg	
	cdef double[:,:] train_fg = np.zeros((fg_index_size, a_max), dtype=NP_DTYPE_F)
	for i in range(fg_index_size):
		node_id = fg_index_int[i]
		if node_id != -1:
			train_fg[i] = fg_tab_c[node_id]
		# else:
		# 	# ----- Compute
		# 	for i in range(node_size):



			
	return np.asarray(fg_tab_c), np.asarray(train_fg), np.asarray(conv_itr_c)


