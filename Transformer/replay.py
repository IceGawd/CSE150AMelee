import melee
import os
import signal

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database import *

savefile = "ice_god_falco"
fn = filefunctions[savefile][0]

# slp_file = "./Game_20250317T165404.slp"
slp_file = "./Game_20250317T172627.slp"

console = melee.Console(system="file", path=slp_file)
console.connect()

try:
	gamestate = console.step()
except:
	print(slp_file + " (Weird???)")

if (gamestate == None):
	print(slp_file + " (Zero/One Frame Game)")

states = []

port = fn(gamestate)
if (port != -1):
	port2 = otherPort(gamestate, port)
	print(slp_file + " (Valid)")

	while gamestate != None:
		states.append(PickleableControllerState(gamestate.players[port].controller_state))

		gamestate = console.step()
else:
	print(slp_file + " (Invalid)")


console.stop()

console = melee.Console(path="/home/avighna/Downloads/Slippi_Online-x86_64.AppImage")

myPort = 1
opPort = 2

controller = melee.Controller(console=console, port=myPort)
controller_human = melee.Controller(console=console,
									port=opPort,
									type=melee.ControllerType.GCN_ADAPTER)

console.run()
console.connect()

controller.connect()
controller_human.connect()

def signal_handler(sig, frame):
	console.stop()
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

i = 0

while True:
	# start = time.time()

	gamestate = console.step()
	if gamestate is None:
		continue

	if console.processingtime * 1000 > 17:
		print("WARNING: Last frame took " + str(console.processingtime*1000) + "ms to process.")

	if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
		i += 1
		if (i >= len(states)):
			i = 0
		set_controller_state(controller, states[i])
	else:
		i = 0
		melee.MenuHelper.menu_helper_simple(gamestate,
											controller,
											melee.Character.FALCO,
											melee.Stage.BATTLEFIELD,
											"",
											costume=2,
											autostart=True,
											swag=False)

	# end = time.time()

	# print("Frame: " + str(end - start))