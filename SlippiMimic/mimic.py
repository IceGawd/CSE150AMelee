import melee
import os
import signal
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database import *

console = melee.Console(path="/home/avighna/Downloads/Slippi_Online-x86_64_3.5.2.AppImage")

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

while True:
	# start = time.time()

	gamestate = console.step()
	if gamestate is None:
		continue

	if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
		set_controller_state(controller, PickleableControllerState(gamestate.players[opPort].controller_state))
	else:
		melee.MenuHelper.menu_helper_simple(gamestate,
											controller,
											melee.Character.FALCO,
											melee.Stage.BATTLEFIELD,
											"",
											costume=2,
											autostart=True,
											swag=False)