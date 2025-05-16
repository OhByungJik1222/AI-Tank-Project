from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
import random

# Initialize Flask server
app = Flask(__name__)

# Load YOLO model
model = YOLO('yolov8n.pt')

# Global variables for tank commands
# Movement commands
move_command = ["W", "W", "W", "D", "D", "D", "A", "A", "S", "S", "STOP"]
# Action commands
action_command = ["Q", "Q", "Q", "Q", "E", "E", "E", "E", "F", "F", "R", "R", "R", "R", "FIRE"]


@app.route('/detect', methods=['POST'])
def detect():
    """Receives an image from the simulator, performs object detection, and returns filtered results."""
    image = request.files.get('image')

    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)  # Save temporary image

    # Perform detection
    results = model(image_path)
    detections = results[0].boxes.data.cpu().numpy()  # Extract bounding boxes

    # Filter only specific object classes
    target_classes = {0: "person", 2: "car", 7: "truck", 15: "rock"}
    filtered_results = []

    for box in detections:
        class_id = int(box[5])
        if class_id in target_classes:
            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4])
            })

    return jsonify(filtered_results)


@app.route('/info', methods=['POST'])
def info():
    """Receives general data from the simulator and prints it for debugging."""
    data = request.get_json(force=True)

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    app.logger.info("Received /info data: %s", data)
    print("Received /info data:", data)

    return jsonify({"status": "success", "message": "Data received"}), 200


@app.route('/update_position', methods=['POST'])
def update_position(): 
    """Updates the tank's current position in the simulator."""
    data = request.get_json()

    if not data or "position" not in data:
        return jsonify({"status": "ERROR", "message": "Missing position data"}), 400

    try:
        x, y, z = map(float, data["position"].split(","))
        current_position = (int(x), int(z))  # Ignore height (y)
        print(f"Updated Position: {current_position}")
        return jsonify({"status": "OK", "current_position": current_position})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 400


# @app.route('/get_move', methods=['GET'])
# def get_move():
#     """Provides the next movement command to the simulator."""
#     global move_command

#     if move_command:
#         command = move_command.pop(0)
#         print(f"Sent Move Command: {command}")
#         return jsonify({"move": command})
#     else:
#         return jsonify({"move": "STOP"})

@app.route('/get_move', methods=['GET'])
def get_move():
    """랜덤한 방향으로 탱크 이동 명령을 반환합니다."""
    directions = ["W", "A", "S", "D", "STOP"]  # 앞으로, 왼쪽, 뒤로, 오른쪽, 정지
    command = random.choice(directions)
    print(f"🚶 Sent Random Move Command: {command}")
    return jsonify({"move": command})

# @app.route('/get_action', methods=['GET'])
# def get_action():
#     """Provides the next turret action command to the simulator."""
#     global action_command

#     if action_command:
#         command = action_command.pop(0)
#         print(f"Sent Action Command: {command}")
#         return jsonify({"turret": command})
#     else:
#         return jsonify({"turret": " "})

@app.route('/get_action', methods=['GET'])
def get_action():
    """랜덤한 포탑 명령을 반환합니다."""
    actions = ["LEFT", "RIGHT", "FIRE", ""]  # 왼쪽 회전, 오른쪽 회전, 발사, 아무 동작 없음
    command = random.choice(actions)
    print(f"🎯 Sent Random Turret Command: {command}")
    return jsonify({"turret": command})

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    """Receives bullet collision data from the simulator and logs it."""
    data = request.get_json()

    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    print(f"Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})


@app.route('/set_destination', methods=['POST'])
def set_destination():
    """Receives a destination from the simulator and calculates the shortest path."""
    global move_command, current_position
    data = request.get_json()

    if not data or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    if current_position is None:
        return jsonify({"status": "ERROR", "message": "Current position not set. Call /update_position first."}), 400

    try:
        x_dest, _, z_dest = map(float, data["destination"].split(","))
        goal = (int(x_dest), int(z_dest))

        obstacles = [(int(x), int(z)) for x, _, z in (o.split(",") for o in data.get("obstacles", []))]

        # Compute shortest path (A* algorithm implementation needed)
        move_command = path_to_wasd(a_star(current_position, goal, obstacles))
        print(f"Generated Move Commands: {move_command}")

        return jsonify({"status": "OK", "command": move_command})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 400


@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    """Receives obstacle data from the simulator and logs it."""
    obstacle_data = request.get_json()

    if not obstacle_data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    print("Received obstacle data:", obstacle_data)
    return jsonify({'status': 'success', 'message': 'Obstacle data received'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
