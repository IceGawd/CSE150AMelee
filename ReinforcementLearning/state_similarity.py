import melee
import os
import heapq
import numpy as np
from collections import defaultdict
import pickle

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database import *

class Action():
	def __init__(self, magnitude_divisions=0.5):
		self.magnitude_divisions = magnitude_divisions

	def give_action(self, gamestate, port):
		player = gamestate.players[port]

		state = PickleableControllerState(player.controller_state)

		return np.concatenate([
			np.array([int(v) for v in state.button.values()]),
			(np.array(state.c_stick) / self.magnitude_divisions).astype('int'),
			(np.array([state.l_shoulder]) / self.magnitude_divisions).astype('int'),
			(np.array(state.main_stick) / self.magnitude_divisions).astype('int'),
			(np.array([state.r_shoulder]) / self.magnitude_divisions).astype('int')
		])



class State():
	def __init__(self, percent_divisions=30, position_divisons=30, velocity_divisions=1, state_divs=100, transitions=None, action=None):
		self.percent_divisions = percent_divisions
		self.position_divisons = position_divisons
		self.velocity_divisions = velocity_divisions
		self.action = action
		self.state_groups = N_set_simplify(transitions, state_divs)

	def state_index(self, element):
	    for index, s in enumerate(self.state_groups):
	        if element in s:
	            return index
	    return -1

	def give_state(self, gamestate, myPort):
		player = gamestate.players[myPort]

		opPort = otherPort(gamestate, myPort)

		value = []

		for port in [myPort, opPort]:
			playerstate = gamestate.players[port]
			# value.append(int(playerstate.facing))
			# value.append(int(playerstate.jumps_left))
			# value.append(int(playerstate.stock))
			value.append(int(playerstate.position.x / self.position_divisons))
			value.append(int(playerstate.position.y / self.position_divisons))
			# value.append(int(playerstate.speed_air_x_self / self.velocity_divisions))
			# value.append(int(playerstate.speed_ground_x_self / self.velocity_divisions))
			# value.append(int(playerstate.speed_x_attack / self.velocity_divisions))
			# value.append(int(playerstate.speed_y_attack / self.velocity_divisions))
			# value.append(int(playerstate.speed_y_self / self.velocity_divisions))
			# value.append(int(playerstate.percent / self.percent_divisions))
			# value.append(self.state_index(action_stringify(playerstate.action)))

		return np.concatenate([
			np.array(value),
			# self.action.give_action(gamestate, myPort)
		])

def add(action, prev, transitions):
	if prev in transitions:
		if action in transitions[prev]:
			transitions[pplayerstaterev][action] += 1
			return
	else:
		transitions[prev] = {}

	transitions[prev][action] = 1

def action_stringify(action):
	return str(action).split('_')[0]

def state_grouping(slippi_files, savefile):
	transitions = {}

	fn = filefunctions[savefile][0]

	for slp_file in slippi_files:
		if "012" in slp_file: # and "2023" not in slp_file
			console = melee.Console(system="file", path=slp_file)
			console.connect()

			try:
				gamestate = console.step()
			except:
				print(slp_file + " (Weird???)")
				continue

			if (gamestate == None):
				print(slp_file + " (Zero/One Frame Game)")
				continue

			prev = None

			port = fn(gamestate)
			if (port != -1):
				port2 = otherPort(gamestate, port)
				print(slp_file + " (Valid)")
				x_min = 0
				x_max = 0
				y_min = 0
				y_max = 0
				while gamestate != None:
					action = action_stringify(gamestate.players[port].action)

					# x_min = min(x_min, gamestate.players[port].position.x)
					# x_max = max(x_max, gamestate.players[port].position.x)
					# y_min = min(y_min, gamestate.players[port].position.y)
					# y_max = max(y_max, gamestate.players[port].position.y)

					if (prev != None and prev != action):
						add(max(action, prev), min(action, prev), transitions)

					prev = action

					gamestate = console.step()

				# print(x_min)
				# print(x_max)
				# print(y_min)
				# print(y_max)

			else:
				print(slp_file + " (Invalid)")

	with open("./pickles/" + savefile + "_transitions.pkl", 'wb') as f:
		pickle.dump(transitions, f)

	return transitions


def N_set_simplify(transitions, N):
	terms = set(transitions.keys())
	for key in transitions:
		terms.update(transitions[key].keys())
	terms = list(terms)
	
	heap = []
	for term1 in transitions:
		for term2, sim in transitions[term1].items():
			heapq.heappush(heap, (-sim, term1, term2))

	clusters = {term: {term} for term in terms}

	while len(clusters) > N:
		while heap:
			_, term1, term2 = heapq.heappop(heap)
			if term1 in clusters and term2 in clusters:
				break
		else:
			break

		new_cluster = clusters[term1] | clusters[term2]
		del clusters[term1], clusters[term2]

		new_label = "-".join(sorted(new_cluster))
		clusters[new_label] = new_cluster

		for other in list(clusters.keys()):
			if other == new_label:
				continue
			new_similarity = max(
				transitions.get(t1, {}).get(t2, 0)
				for t1 in new_cluster for t2 in clusters[other]
			)
			heapq.heappush(heap, (-new_similarity, new_label, other))

	return list(clusters.values())

if __name__ == "__main__":
	slippi_root = "/home/avighna/Slippi"
	# slippi_root = "/home/avighna/Documents/python/CSE150AMelee/Bayes"

	savefile = "fox_falco"
	fn = filefunctions[savefile][0]

	if os.path.exists("./pickles/" + savefile + "_transitions.pkl"):
		with open("./pickles/" + savefile + "_transitions.pkl", 'rb') as f:
			transitions = pickle.load(f)
	else:
		slippi_files = []
		for root, _, files in os.walk(slippi_root):
			for file in files:
				if (file[-4:] == ".slp"):
					slippi_files.append(os.path.join(root, file))

		slippi_files.sort(key = lambda x: x.split('/')[-1])

		transitions = state_grouping(slippi_files, savefile)

	# """
	for x in transitions:
		print("{" + str(x))
		total = sum([transitions[x][y] for y in transitions[x]])
		for y in transitions[x]:
			print(", " + str(y) + ": " + str(transitions[x][y]))
			transitions[x][y] /= total
		print("}")
	# """

	sets = N_set_simplify(transitions, 100)
	for x in sets:
		print(x)