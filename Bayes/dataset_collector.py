import os
import melee
import pickle
import time

class PickleableControllerState(melee.controller.ControllerState):
	def __init__(self, controller_state):
		self.button = controller_state.button
		self.c_stick = controller_state.c_stick
		self.l_shoulder = controller_state.l_shoulder
		self.main_stick = controller_state.main_stick
		self.r_shoulder = controller_state.r_shoulder
	def __getstate__(self):
		state = {
			"button": self.button,
			"c_stick": self.c_stick,
			"l_shoulder": self.l_shoulder,
			"main_stick": self.main_stick,
			"r_shoulder": self.r_shoulder
		}
		return state

	def __setstate__(self, state):
		self.button = state["button"]
		self.c_stick = state["c_stick"]
		self.l_shoulder = state["l_shoulder"]
		self.main_stick = state["main_stick"]
		self.r_shoulder = state["r_shoulder"]

def ice_god_samus(gamestate):
	if len(gamestate.players.keys()) != 2:
		return -1

	samusPort = -1

	for port in gamestate.players.keys():
		if (gamestate.players[port].character == melee.enums.Character.SAMUS):
			if ("ice" in gamestate.players[port].connectCode.lower()):
				return port
			else:
				if (samusPort == -1):
					samusPort = port
				else:
					return -1

	return samusPort

def fox_fox_FD(gamestate):
	if len(gamestate.players.keys()) != 2:
		return -1

	if gamestate.stage != melee.enums.Stage.FINAL_DESTINATION:
		return -1

	for port in gamestate.players.keys():
		if (gamestate.players[port].character != melee.enums.Character.FOX):
			return -1

	return port

def saveData(file, data):
	# print(data)
	print("DON'T QUIT Saving...")
	with open(file, 'wb') as f:
		pickle.dump(data, f)
	print("Saved!")

def otherPort(gamestate, myPort):
	for port in gamestate.players.keys():
		if myPort != port:
			return port

	raise Exception("How?")

def keyInfo(gamestate, myPort, opPort):
	info = []
	for port in [myPort, opPort]:
		playerstate = gamestate.players[port]
		info.append(playerstate.character)
		info.append(playerstate.action)
	info.append(gamestate.stage)
	return info

def valueFn(gamestate, myPort, opPort):
	value = []
	for port in [myPort, opPort]:
		playerstate = gamestate.players[port]
		value.append(float(playerstate.facing) * 100)
		value.append(float(playerstate.jumps_left) * 20)
		value.append(float(playerstate.percent))
		value.append(float(playerstate.hitstun_frames_left))
		value.append(float(playerstate.invulnerability_left))
		value.append(float(playerstate.position.x))
		value.append(float(playerstate.position.y))
		value.append(float(playerstate.shield_strength))
		value.append(float(playerstate.speed_air_x_self))
		value.append(float(playerstate.speed_ground_x_self))
		value.append(float(playerstate.speed_x_attack))
		value.append(float(playerstate.speed_y_attack))
		value.append(float(playerstate.speed_y_self))
		value.append(float(playerstate.stock))

	vDict = {}
	vDict["input"] = PickleableControllerState(gamestate.players[myPort].controller_state)
	vDict["value"] = value

	return vDict

def stringEnumerate(info):
	s = ""
	for i in info:
		s += str(i)
	return s

def addData(data, gamestate, myPort, opPort):
	key = stringEnumerate(keyInfo(gamestate, myPort, opPort))
	vDict = valueFn(gamestate, myPort, opPort)

	if key in data["data"]:
		data["data"][key].append(vDict)
	else:
		data["data"][key] = [vDict]

def loadData(savefile):
	with open("./" + savefile, "rb") as file:
	    return pickle.load(file)

if __name__ == "__main__":
	slippi_root = "/home/avighna/Slippi"

	filefunctions = {"ice_god_samus": ice_god_samus, "fox_fox_FD": fox_fox_FD}

	savefile = "fox_fox_FD.pkl"
	fn = filefunctions[savefile.split('.')[0]]

	data = {
		"last_file": None,
		"data": {}
	}

	if os.path.exists("./" + savefile):
		data = loadData(savefile)

	slippi_files = []
	for root, _, files in os.walk(slippi_root):
		for file in sorted(files):
			slippi_files.append(os.path.join(root, file))

	lastSaved = time.time()
	reading = data["last_file"] == None

	for slp_file in slippi_files:
		if reading:
			console = melee.Console(system="file", path=slp_file)
			console.connect()

			gamestate = console.step()
			port = fn(gamestate)
			if (port != -1):
				port2 = otherPort(gamestate, port)
				print(slp_file + " (Valid)")
				while gamestate != None:
					addData(data, gamestate, port, port2)
					gamestate = console.step()
			else:
				print(slp_file + " (Invalid)")

			data["last_file"] = slp_file

			if time.time() - lastSaved > 30:						# Save after one minute of processing
				saveData(savefile, data)
				lastSaved = time.time()

			console.stop()
		else:
			reading = slp_file == data["last_file"]

	saveData(savefile, data)