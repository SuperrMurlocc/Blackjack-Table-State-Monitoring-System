import cv2
import matplotlib.pyplot as plt
import numpy as np
import einops
from typing import Callable, Literal

from .utils import get_lines_intersection, four_point_transform
from .match_template import get_templates_matches, get_best_template, print_template_matches_table, name_to_sign


CARD_MIN_AREA = 70_000

imageType = np.ndarray
transformType = Callable[[imageType], imageType]
contourType = np.ndarray
cardRank = Literal['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']
logLevel = Literal['NONE', 'IMPORTANT', 'ALL']


class Transforms:
    @staticmethod
    def resize(new_size: tuple[int, int]) -> transformType:
        return lambda image: cv2.resize(image, new_size)

    @staticmethod
    def to_grayscale() -> transformType:
        return lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def clahe(clipLimit: float, titleGridSize: tuple[int, int]) -> transformType:
        return lambda image: cv2.createCLAHE(clipLimit, titleGridSize).apply(image)

    @staticmethod
    def blur(kernel_sizes: tuple[int, int], sigmaX: int) -> transformType:
        return lambda image: cv2.GaussianBlur(image, kernel_sizes, sigmaX)

    @staticmethod
    def binarize(_min: int, _max: int, _type: int) -> transformType:
        return lambda image: cv2.threshold(image, _min, _max, _type)[1]

    @staticmethod
    def close_areas(kernel: imageType) -> transformType:
        return lambda image: cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


class Vision:
    @staticmethod
    def imshow(img: imageType, ax=None, cmap='gray') -> None:
        if ax is None:
            plt.figure(figsize=(10, 10))
            ax = plt
        ax.imshow(img, cmap=cmap)
        ax.axis('off')
        if ax == plt:
            plt.plot()

    @staticmethod
    def sequential(transforms: list[transformType], base_image: imageType, *, log_level: logLevel = False) -> imageType:
        image = base_image.copy()
        for transform in transforms:
            image = transform(image)
            if log_level == 'ALL':
                Vision.imshow(image)
        return image

    def __init__(self, image_path: str, *, cvtColorCode: int = cv2.COLOR_BGR2RGB, log_level: logLevel = 'NONE'):
        self.log_level = log_level

        self.base_image = cv2.cvtColor(cv2.imread(image_path), cvtColorCode)
        self.areas_image = self.preprocess_image(self.base_image)
        self.pile_contours, self.cards_contours = self.segment_image()
        self.piles_contents = self.analyze_piles_contents()

        if self.log_level in ['IMPORTANT', 'ALL']:
            self.imshow(self.base_image)
            self.imshow(self.areas_image)

            display = self.base_image.copy()

            for pile_contour, pile_contents, card_contours in zip(self.pile_contours, self.piles_contents, self.cards_contours):
                cv2.drawContours(display, [pile_contour], 0, (0, 255, 0), 22)

                x, y, w, h = cv2.boundingRect(pile_contour)
                cv2.putText(display, ', '. join(pile_contents), (x + w // 2, y + h + 100), 0, 3.,(0, 0, 0), 11)

                for card_contour in card_contours:
                    rect = cv2.minAreaRect(card_contour)
                    box = np.int32(cv2.boxPoints(rect))
                    cv2.drawContours(display, [box], 0, (0, 0, 255), 11)

            self.imshow(display)

    def preprocess_image(self, image: imageType) -> imageType:
        return Vision.sequential([
            Transforms.to_grayscale(),
            Transforms.clahe(2.0, (8, 8)),
            Transforms.blur((21, 21), 0),
            Transforms.binarize(0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU),
            Transforms.close_areas(np.ones((7, 7))),
        ], image, log_level=self.log_level)

    @staticmethod
    def get_cropped_image(image: imageType, contour: imageType) -> (imageType, contourType):
        x, y, w, h = cv2.boundingRect(contour)

        mask = np.zeros_like(image)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        masked_image = cv2.bitwise_and(image, mask)

        return masked_image[y:y + h, x:x + w].copy(), contour.copy() - (x, y)

    def segment_image(self) -> tuple[list[contourType], list[list[contourType]]]:
        contours, _ = cv2.findContours(self.areas_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        pile_contours = []
        cards_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < CARD_MIN_AREA:
                break

            pile_contours.append(contour)
            cards_contours.append(self.segment_pile_image(self.areas_image, contour))

        return pile_contours, cards_contours

    def fix_hard_example(self, pile_image: imageType, contour: contourType) -> imageType:
        display, contour = self.get_cropped_image(self.base_image, contour)

        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        points = np.array(approx)
        points = einops.rearrange(points, 'n_points 1 coords -> n_points coords')

        if self.log_level == 'ALL':
            for i, point in enumerate(points):
                cv2.putText(display, f'{i}', point, 0, 2., (0, 0, 255), 5)

        for i in range(1, len(points) // 2 - 1, 2):
            point_1_a = points[i]
            point_1_b = points[i + 1]
            point_2_a = points[len(points) - i]
            point_2_b = points[len(points) - i - 1]

            intersection = np.round(get_lines_intersection(point_1_a, point_1_b, point_2_a, point_2_b)).astype(np.int32)

            cv2.line(pile_image, point_1_b, intersection, color=(0, 0, 0), thickness=10)
            cv2.line(pile_image, point_2_b, intersection, color=(0, 0, 0), thickness=10)

            if self.log_level == 'ALL':
                cv2.putText(display, f'x', intersection, 0, 2., (0, 255, 0), 5)
                cv2.line(display, point_1_b, intersection, color=(255, 0, 0), thickness=10)
                cv2.line(display, point_2_b, intersection, color=(255, 0, 0), thickness=10)

        if self.log_level == 'ALL':
            self.imshow(display)

        return pile_image

    def segment_pile_image(self, image: imageType, contour: contourType) -> list[contourType]:
        x, y, *_ = cv2.boundingRect(contour)

        pile_image, pile_contour = self.get_cropped_image(image, contour)

        if self.log_level == 'ALL':
            self.imshow(pile_image)

        self.fix_hard_example(pile_image, contour)

        if self.log_level == 'ALL':
            self.imshow(pile_image)

        _contours, _ = cv2.findContours(pile_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _contours = sorted(_contours, key=cv2.contourArea, reverse=True)

        card_contours = []

        for _contour in _contours:
            area = cv2.contourArea(_contour)
            if area < CARD_MIN_AREA:
                break

            card_contours.append(_contour + (x, y))

        return card_contours

    def analyze_piles_contents(self) -> list[list[cardRank]]:
        piles_contents = []

        for pile_contour, card_contours in zip(self.pile_contours, self.cards_contours):
            pile_contents = []

            for card_contour in card_contours:
                rect = cv2.minAreaRect(card_contour)
                box = np.int32(cv2.boxPoints(rect))

                transformed = four_point_transform(self.base_image, box)

                if self.log_level == 'ALL':
                    self.imshow(transformed)

                gray_transformed = self.sequential([
                    Transforms.to_grayscale(),
                    Transforms.resize((200, 300)),
                ], transformed, log_level=self.log_level)

                gray_transformed_corner = gray_transformed[0:84, 0:32]

                if self.log_level == 'ALL':
                    self.imshow(gray_transformed_corner)

                binarized_transformed_corner = self.sequential([
                    Transforms.resize((32*4, 84*4)),
                    Transforms.binarize(155, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                ], gray_transformed_corner, log_level=self.log_level)

                if self.log_level == 'ALL':
                    self.imshow(binarized_transformed_corner)
                    print_template_matches_table(get_templates_matches(binarized_transformed_corner))

                pile_contents.append(name_to_sign[get_best_template(binarized_transformed_corner)])

            piles_contents.append(pile_contents)

        return piles_contents
