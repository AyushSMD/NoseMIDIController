import cv2
from cv2_enumerate_cameras import enumerate_cameras
import mediapipe as mp
import rtmidi
import platform
import sys
import time

# -----------------------------
# MIDI Setup (Cross-platform)
# -----------------------------
midiout = rtmidi.MidiOut()
system = platform.system()

if system in ["Linux", "Darwin"]:
    try:
        midiout.open_virtual_port("Nose MIDI Controller")
        print("Virtual MIDI port opened (Linux/macOS)")
    except Exception as e:
        print(f"Failed to open virtual port: {e}")
        sys.exit(1)
else:
    ports = midiout.get_ports()
    if not ports:
        print("No MIDI output ports available. Please install a loopback device like loopMIDI.")
        sys.exit(1)

    print("Available MIDI Output Ports:")
    for i, port in enumerate(ports):
        print(f"  [{i}] {port}")
    
    try:
        midi_port = int(input("Enter port number: "))
        midiout.open_port(midi_port)
        print(f"Connected to MIDI port: {ports[midi_port]}")
    except Exception as e:
        print(f"Failed to open MIDI port: {e}")
        sys.exit(1)

def send_midi_control(value):
    msg = [0xB0, 74, value]
    midiout.send_message(msg)

# -----------------------------
# Face Tracking Setup
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
camera_list = enumerate_cameras()

if not camera_list:
    print("No camera devices found.")
    sys.exit(1)
else:
    print(f"Found {len(camera_list)} camera devices:")
    for i, cam_info in enumerate(camera_list):
        print(f"  [{i}] Index: {cam_info.index}, Name: {cam_info.name}")

    camera_index = int(input("Enter camera index:"))
    selected_cam = camera_list[camera_index]
    cap = cv2.VideoCapture(selected_cam.index, selected_cam.backend)
    if not cap.isOpened():
        print(f"Failed to open camera: {selected_cam.name}")
        sys.exit(1)
    else:
        print(f"Successfully opened camera: {selected_cam.name}")

# -----------------------------
# Calibration Function
# -----------------------------
def calibrate(cap, face_mesh):
    instruction = "Calibration: LOWEST point first..."
    start_time = time.time()
    min_y = None
    max_y = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        # Display instruction text
        cv2.putText(frame, instruction, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        if results.multi_face_landmarks:
            nose_tip = results.multi_face_landmarks[0].landmark[1]
            cx, cy = int(nose_tip.x * w), int(nose_tip.y * h)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            if "LOWEST" in instruction:
                min_y = nose_tip.y
            else:
                max_y = nose_tip.y

        cv2.imshow("Nose Tracker", frame)
        key = cv2.waitKey(1) & 0xFF

        # Wait 2 seconds per step
        if time.time() - start_time > 2:
            if "LOWEST" in instruction:
                instruction = "Calibration: HIGHEST point next..."
                start_time = time.time()
            else:
                break

        # Allow exit during calibration
        if key == ord('q'):
            sys.exit(0)

    if min_y > max_y:
        min_y, max_y = max_y, min_y

    return min_y, max_y

# -----------------------------
# Initial Calibration
# -----------------------------
min_y, max_y = calibrate(cap, face_mesh)
print(f"Calibration complete: min_y={min_y:.3f}, max_y={max_y:.3f}")

# -----------------------------
# Main Loop
# -----------------------------
prev_midi_value = -1  # previous MIDI value to reduce unnecessary messages

print("🎥 Running Nose MIDI Controller...")
print("👉 Press 'q' to quit, 'r' to recalibrate")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    # Overlay instructions
    cv2.putText(frame, "Press 'R' to recalibrate, 'Q' to quit", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if results.multi_face_landmarks:
        nose_tip = results.multi_face_landmarks[0].landmark[1]
        cx, cy = int(nose_tip.x * w), int(nose_tip.y * h)

        # Draw vertical line first (background)
        low_cy = int(min_y * h)
        high_cy = int(max_y * h)
        cv2.line(frame, (cx, high_cy), (cx, low_cy), (255, 255, 255), 2)

        # Draw red dot behind
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

        # Draw min/max dots on top
        cv2.circle(frame, (cx, low_cy), 6, (0, 255, 0), -1)   # lowest
        cv2.circle(frame, (cx, high_cy), 6, (255, 0, 0), -1)  # highest

        # Map Y to MIDI 0-127
        midi_value = int((max_y - nose_tip.y) / (max_y - min_y) * 127)
        midi_value = max(0, min(127, midi_value))

        alpha = 0.2  # smoothing factor
        smoothed_midi = prev_midi_value if prev_midi_value != -1 else midi_value
        smoothed_midi = int(prev_midi_value * (1 - alpha) + midi_value * alpha)

        # Send MIDI only if smoothed value changed
        if smoothed_midi != prev_midi_value:
            send_midi_control(smoothed_midi)
            prev_midi_value = smoothed_midi

        # Show MIDI value
        cv2.putText(frame, f"MIDI: {midi_value}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Nose Tracker", frame)
    key = cv2.waitKey(1) & 0xFF

    # Recalibrate
    if key == ord('r'):
        min_y, max_y = calibrate(cap, face_mesh)
        prev_midi_value = -1  # reset previous value
        print(f"Recalibration complete: min_y={min_y:.3f}, max_y={max_y:.3f}")

    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
