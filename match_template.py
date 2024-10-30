import os
import cv2

template_dir = 'templates'
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
}


def match_template(card_image, template_image):
    # Convert the template to grayscale (if needed)
    template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

    # Apply template matching
    result = cv2.matchTemplate(card_image, template_gray, cv2.TM_CCOEFF_NORMED)

    # Get the best match position
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    return round(max_val, 4), max_loc
