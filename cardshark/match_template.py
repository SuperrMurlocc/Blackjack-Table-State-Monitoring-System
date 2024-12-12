import os
import cv2
import matplotlib.pyplot as plt

template_dir = 'cardshark/templates'
templates = {}

for template_file in os.listdir(template_dir):
    if template_file.endswith('.jpg'):
        template_name = template_file.split('.')[0]
        template_path = os.path.join(template_dir, template_file)
        templates[template_name] = cv2.imread(template_path)


name_to_sign = {
    'Ace': 'A',
    'King': 'K',
    'Queen': 'Q',
    'Jack': 'J',
    'Ten': '10',
    'Nine': '9',
    'Eight': '8',
    'Seven': '7',
    'Six': '6',
    'Five': '5',
    'Four': '4',
    'Three': '3',
    'Two': '2',
    'Hidden': '?',
}

name_to_points = {
    'Ace': 11,
    'King': 10,
    'Queen': 10,
    'Jack': 10,
    'Ten': 10,
    'Nine': 9,
    'Eight': 8,
    'Seven': 7,
    'Six': 6,
    'Five': 5,
    'Four': 4,
    'Three': 3,
    'Two': 2,
    'Hidden': 0,
}


def match_template(card_image, template_image):
    # Convert the template to grayscale (if needed)
    template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

    # Apply template matching
    result = cv2.matchTemplate(card_image, template_gray, cv2.TM_CCOEFF_NORMED)

    # Get the best match position
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    return round(max_val, 4), max_loc


def get_templates_matches(card_corner):
    return sorted([[name, *match_template(card_corner, template)] for name, template in templates.items()],key=lambda x: x[1], reverse=True)


def get_best_template(card_corner, scale: float = 1.):
    template_matches = get_templates_matches(card_corner)
    best_similarity = template_matches[0][1]

    # if best_similarity <= 0.45 + scale / 10:
    #     return 'Hidden'

    return template_matches[0][0]


def print_template_matches_table(templates_matches):
    table = plt.table(
        colLabels=['Card', 'Similarity', 'Position where similarity was found'],
        cellText=templates_matches,
        cellLoc='center',
        rowLoc='center',
        loc='bottom'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)

    plt.plot()
