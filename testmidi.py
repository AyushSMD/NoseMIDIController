import sys
import rtmidi

midiin = rtmidi.MidiIn()
ports = midiin.get_ports()
if not ports:
    print("No MIDI output ports available. Please install a loopback device like loopMIDI.")
    sys.exit(1)

print("Available MIDI Output Ports:")
for i, port in enumerate(ports):
    print(f"  [{i}] {port}")

try:
    midi_port = int(input("Enter port number: "))
    midiin.open_port(midi_port)
    print(f"Connected to MIDI port: {ports[midi_port]}")
except Exception as e:
    print(f"Failed to open MIDI port: {e}")
    sys.exit(1)

print("Listening to MIDI input...")
while True:
    msg = midiin.get_message()
    if msg:
        message, delta = msg
        print(message)