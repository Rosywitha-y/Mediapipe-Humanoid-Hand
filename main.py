
import time
import math

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
BaseOptions = mp.tasks.BaseOptions

model_path = "hand_landmarker.task"

def vec(a, b): # form 3D vectors
    return b.x - a.x, b.y - a.y, b.z - a.z

def dot(u, v): # according to math...dot product connects vectors to angles
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

def angle_degree(a, b, c):
    v1 = vec(b, a)
    v2 = vec(b, c)
    v1_length = dot(v1, v1)
    v2_length = dot(v2, v2)
    if v1_length < 1e-6 or v2_length < 1e-6: # prevents devision by 0
        return None
    cosine_angle = dot(v1, v2) / (math.sqrt(v1_length) * math.sqrt(v2_length)) # dot product formula
    cosine_angle = max(-1.0, min(1.0, cosine_angle))
    return math.degrees(math.acos(cosine_angle))

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),  # middle
    (9, 13), (13, 14), (14, 15), (15, 16),# ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17)                               # root of hand yk
]

fingers = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9,10,11,12],
    "ring":   [13,14,15,16],
    "pinky":  [17,18,19,20],
}

def angle_in_joints(hand):
    angles = {}

    for each in ["index", "middle", "ring", "pinky"]:
        mcp, pip, dip, tip = fingers[each]
        angles[each] = {
            "MCP": angle_degree(hand[0], hand[mcp], hand[pip]),
            "PIP": angle_degree(hand[mcp], hand[pip], hand[dip]),
            "DIP": angle_degree(hand[pip], hand[dip], hand[tip]),
        }

    cmc, mcp, ip, tip = fingers["thumb"]
    angles["thumb"] = {
        "CMC": angle_degree(hand[0], hand[cmc], hand[mcp]),  # wrist->CMC->MCP (optional)
        "MCP": angle_degree(hand[cmc], hand[mcp], hand[ip]),
        "IP": angle_degree(hand[mcp], hand[ip], hand[tip]),
    }
    return angles

def decide_bend(angle):
    if angle is None:
        return "?"
    if angle > 165:
        return "straight"
    if angle > 145:
        return "slight"
    if angle > 120:
        return "bent"
    return "very"

def draw_landmarks_on_image(rgb_image, detection_result):
    annotated = rgb_image.copy() # make a copy of the original
    h = annotated.shape[0]
    w = annotated.shape[1]

    if not detection_result.hand_landmarks:
        return annotated
    for hand in detection_result.hand_landmarks:
        for landmark in hand:
            x, y = int(landmark.x * w), int(landmark.y * h) # these coverts the x and y given by mediapipe to pixel coordinates
            cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1) # draw circles
            for startIDX, endIDX in HAND_CONNECTIONS:
                p1 = hand[startIDX]
                p2 = hand[endIDX]
                x1 = int(p1.x * w)
                y1 = int(p1.y * h)
                x2 = int(p2.x * w)
                y2 = int(p2.y * h)

                cv2.line(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2) # draw line
    return annotated

latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = (result, output_image.numpy_view())

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path = model_path),
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands= 2,
    result_callback=result_callback
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
start_time = time.monotonic()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = int((time.monotonic() - start_time) * 1000)
    detector.detect_async(mp_image, timestamp_ms)

    if latest_result:
        result, rgb_output = latest_result
        annotated = draw_landmarks_on_image(rgb_output, result)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            angles = angle_in_joints(hand)

            y0 = 200
            for fname, joints in angles.items():
                parts = []
                for jname, ang in joints.items():
                    if ang is None:
                        parts.append(f"{jname}: ?")
                    else:
                        parts.append(f"{jname}:{ang:5.1f}°({decide_bend(ang)})")
                line = f"{fname}: " + "  ".join(parts)

                cv2.putText(
                    annotated, line, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
                )
                y0 += 18

        frame = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    cv2.imshow("Hand Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
