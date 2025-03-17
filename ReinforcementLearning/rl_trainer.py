import os
import melee
import pickle
import time
import itertools

from state_similarity import *

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database import *

action_obj = None
state_obj = None

def saveData(file, data):
	# print(data)
	print("DON'T QUIT Saving...")

	with open("./pickles/" + file + ".pkl", 'wb') as f:
		pickle.dump(data, f)

	print("Saved!")


def addData(data, gamestate, myPort, opPort):
	global action_obj
	global state_obj

	key = stringEnumerate(keyInfo(gamestate, myPort, opPort))

	action = action_obj.give_action(gamestate, myPort)
	state = state_obj.give_state(gamestate, myPort)

	if ("action" in data["data"]["prev"]):
		experience = {
			"prev_action": data["data"]["prev"]["action"],
			"prev_state": data["data"]["prev"]["state"],
		}
		
		if (key in data["data"]):
			if str(state) in data["data"][key]:
				data["data"][key][str(state)].append(experience)
			else:
				data["data"][key][str(state)] = [experience]
		else:
			data["data"][key] = {}
			data["data"][key][str(state)] = [experience]

	if "minimum" in data["data"]:
		data["data"]["maximum"] = np.maximum(data["data"]["maximum"], state)
		data["data"]["minimum"] = np.minimum(data["data"]["minimum"], state)
	else:
		data["data"]["maximum"] = state
		data["data"]["minimum"] = state

	data["data"]["prev"]["action"] = action
	data["data"]["prev"]["state"] = state

def loadData(file):
	print("Loading data...")

	data = {"data": {}, "last_file": None}
	
	data_path = "./pickles/" + file + ".pkl"
	if os.path.exists(data_path):
		with open(data_path, 'rb') as f:
			data = pickle.load(f)
	
	print("Data Loaded!")    
	return data

if __name__ == "__main__":
	savefile = "fox_falco"

	action_obj = Action()

	with open("./pickles/" + savefile + "_transitions.pkl", 'rb') as f:
		transitions = pickle.load(f)

	state_obj = State(transitions=transitions, action=action_obj)

	data = loopThrough(addData, saveData, loadData, filesFrom=["012"], savefile=savefile)

	# """
	# ranges = [list(range(min_v, max_v + 1)) for min_v, max_v in zip(data["data"]["minimum"], data["data"]["maximum"])]
 
	l = 0
	for key in data["data"]:
		l += len(data["data"][key])
	print(l)
	# """
 
	# print(data["data"]["minimum"])
	# print(data["data"]["maximum"])