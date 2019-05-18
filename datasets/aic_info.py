MAX_COLORS = 42

AIC_ATTR = {
    'sex': ['Female', 'other'],
    'age': ['Age17-30', 'Age31-45'],
    'body': ['BodyNormal', 'BodyThin'],
    'hair': ['BaldHead', 'LongHair', 'other'],
    'hair_color': ['BlackHair', 'other'],
    'hat': ['Hat', 'other'],
    'muffler': ['Muffler', 'other'],
    'upper': ['Shirt', 'Sweater', 'Jacket', 'TightHood', 'other'],
    'short_sleeve': ['ShortSleeve', 'other'],
    'bottom': ['LongTrousers', 'Skirt', 'Jeans', 'Tights', 'other'],
    'shoes': ['shoes-Leather', 'shoes-Sport', 'shoes-Boots', 'other'],
    'backpack': ['Backpack', 'other'],
    'glasses': ['Glasses'],
}

ATT_ORDER = [
    'sex',
    'age',
    'body',
    'hair',
    'hair_color',
    'hat',
    'muffler',
    'upper',
    'short_sleeve',
    'bottom',
    'shoes',
    'backpack',
    'glasses',
]


SINGLE_ATTR = [
    'Female',
    'Age17-30',
    'Age31-45',
    'BodyNormal',
    'BodyThin',
    'BaldHead',
    'LongHair',
    'BlackHair',
    'Hat',
    'Muffler',
    'Shirt',
    'Sweater',
    'Jacket',
    'TightHood',
    'ShortSleeve',
    'LongTrousers',
    'Skirt',
    'Jeans',
    'Tights',
    'shoes-Leather',
    'shoes-Sport',
    'shoes-Boots',
    'Backpack',
    'Glasses',
]

LIMBS = [
    (0, 1),  # head_top -> head_center
    (1, 2),  # head_center -> neck
    (2, 3),  # neck -> right_clavicle
    (3, 4),  # right_clavicle -> right_shoulder
    (4, 5),  # right_shoulder -> right_elbow
    (5, 6),  # right_elbow -> right_wrist
    (2, 7),  # neck -> left_clavicle
    (7, 8),  # left_clavicle -> left_shoulder
    (8, 9),  # left_shoulder -> left_elbow
    (9, 10),  # left_elbow -> left_wrist
    (2, 11),  # neck -> spine0
    (11, 12),  # spine0 -> spine1
    (12, 13),  # spine1 -> spine2
    (13, 14),  # spine2 -> spine3
    (14, 15),  # spine3 -> spine4
    (15, 16),  # spine4 -> right_hip
    (16, 17),  # right_hip -> right_knee
    (17, 18),  # right_knee -> right_ankle
    (15, 19),  # spine4 -> left_hip
    (19, 20),  # left_hip -> left_knee
    (20, 21)  # left_knee -> left_ankle
]

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

JOINTS = 22
NJ = 14

# NO
NO_JOINTS = [1, 3, 7, 11, 12, 13, 14, 15]


def person_id(person):
    return person['id']


def person_attributes(person):
    return person['attributes']


def person_joints(person):
    return person['pose']


def get_joints(person):
    return person['']
