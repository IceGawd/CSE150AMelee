import melee
import signal

console = melee.Console(path="/home/avighna/Downloads/Slippi_Online-x86_64.AppImage")

controller = melee.Controller(console=console, port=1)
controller_human = melee.Controller(console=console,
									port=2,
									type=melee.ControllerType.GCN_ADAPTER)

console.run()
console.connect()

controller.connect()
controller_human.connect()

def signal_handler(sig, frame):
	console.stop()
	if args.debug:
		log.writelog()
		print("") #because the ^C will be on the terminal
		print("Log file created: " + log.filename)
	print("Shutting down cleanly...")
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


while True:
	gamestate = console.step()
	# Press buttons on your controller based on the GameState here!
	