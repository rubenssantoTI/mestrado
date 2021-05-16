
PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]
HEADERS = [
    'noseX', 'noseY',
    'leftEyeX', 'leftEyeY',
    'rightEyeX', 'rightEyeY',
    'leftEarX', 'leftEarY',
    'rightEarX', 'rightEarY',
    'leftShoulderX', 'leftShoulderY',
    'rightShoulderX', 'rightShoulderY',
    'leftElbowX', 'leftElbowY',
    'rightElbowX', 'rightElbowY',
    'leftWristX', 'leftWristY',
    'rightWristX', 'rightWristY',
    'leftHipX', 'leftHipY',
    'rightHipX', 'rightHipY',
    'leftKneeX', 'leftKneeY',
    'rightKneeX', 'rightKneeY',
    'leftAnkleX', 'leftAnkleY',
    'rightAnkleX', 'rightAnkleY'
]

COORDENADAS = [
"\"noseX\": \"% s\", \"noseY\": \"% s\", ",
"\"leftEyeX\": \"% s\", \"leftEyeY\": \"% s\", ",
"\"rightEyeX\": \"% s\", \"rightEyeY\": \"% s\", ",
"\"leftEarX\": \"% s\", \"leftEarY\": \"% s\", ",
"\"rightEarX\": \"% s\", \"rightEarY\": \"% s\", ",
"\"leftShoulderX\": \"% s\", \"leftShoulderY\": \"% s\",",
"\"rightShoulderX\": \"% s\", \"rightShoulderY\": \"% s\", ",
"\"leftElbowX\": \"% s\", \"leftElbowY\": \"% s\", ",
"\"rightElbowX\": \"% s\", \"rightElbowY\": \"% s\", ",
"\"leftWristX\": \"% s\", \"leftWristY\": \"% s\", ",
"\"rightWristX\": \"% s\", \"rightWristY\": \"% s\", ",
"\"leftHipX\": \"% s\", \"leftHipY\": \"% s\", ",
"\"rightHipX\": \"% s\", \"rightHipY\": \"% s\", ",
"\"leftKneeX\": \"% s\", \"leftKneeY\": \"% s\", ",
"\"rightKneeX\": \"% s\", \"rightKneeY\": \"% s\", ",
"\"leftAnkleX\": \"% s\", \"leftAnkleY\": \"% s\", ",
"\"rightAnkleX\": \"% s\", \"rightAnkleY\": \"% s\" "
]

NUM_KEYPOINTS = len(PART_NAMES)

PART_IDS = {pn: pid for pid, pn in enumerate(PART_NAMES)}

CONNECTED_PART_NAMES = [
    ("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")
]

CONNECTED_PART_INDICES = [(PART_IDS[a], PART_IDS[b]) for a, b in CONNECTED_PART_NAMES]

LOCAL_MAXIMUM_RADIUS = 1

POSE_CHAIN = [
    ("nose", "leftEye"), ("leftEye", "leftEar"), ("nose", "rightEye"),
    ("rightEye", "rightEar"), ("nose", "leftShoulder"),
    ("leftShoulder", "leftElbow"), ("leftElbow", "leftWrist"),
    ("leftShoulder", "leftHip"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("nose", "rightShoulder"),
    ("rightShoulder", "rightElbow"), ("rightElbow", "rightWrist"),
    ("rightShoulder", "rightHip"), ("rightHip", "rightKnee"),
    ("rightKnee", "rightAnkle")
]

PARENT_CHILD_TUPLES = [(PART_IDS[parent], PART_IDS[child]) for parent, child in POSE_CHAIN]

PART_CHANNELS = [
  'left_face',
  'right_face',
  'right_upper_leg_front',
  'right_lower_leg_back',
  'right_upper_leg_back',
  'left_lower_leg_front',
  'left_upper_leg_front',
  'left_upper_leg_back',
  'left_lower_leg_back',
  'right_feet',
  'right_lower_leg_front',
  'left_feet',
  'torso_front',
  'torso_back',
  'right_upper_arm_front',
  'right_upper_arm_back',
  'right_lower_arm_back',
  'left_lower_arm_front',
  'left_upper_arm_front',
  'left_upper_arm_back',
  'left_lower_arm_back',
  'right_hand',
  'right_lower_arm_front',
  'left_hand'
]