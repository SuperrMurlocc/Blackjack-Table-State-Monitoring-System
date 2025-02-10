# Blackjack Table State Monitoring System

![Classified images card](https://github.com/SuperrMurlocc/aoc-projekt/blob/main/outputs_results/classification.png)  

This project presents a **Blackjack Table State Monitoring System**, designed for **automatic analysis of a Blackjack game** based on a captured image representing a specific game stage. The system performs tasks such as detecting card stacks, recognizing individual cards, calculating their point values, and evaluating the game's outcome at that stage.  

The project relies on **classical image processing algorithms** rather than neural networks, achieving a **100% classification accuracy** on high-quality images. This approach ensures high efficiency on devices with limited computational resources.

---

## Table of Contents  
- [Overview](#overview)  
- [Dataset](#dataset)  
- [System Workflow](#system-workflow)  
- [Technologies Used](#technologies-used)  
- [Results](#results)  
- [Limitations and Future Work](#limitations-and-future-work)  
- [Contributors](#contributors)  

---

## Overview  
The goal of this project was to create a system capable of analyzing a given moment in a Blackjack game using an image of the table. The main functionalities include:  
- Detecting card stack locations.  
- Recognizing individual cards and their point values.  
- Calculating the total points for each stack.  
- Determining the outcome of the game (win, loss, draw).  

---

## Dataset  
A custom dataset was created, consisting of **65 image-label pairs**, each representing a unique, valid Blackjack game scenario. All images were captured using an **iPhone 15 Pro** at a resolution of 5712x4284.  

**Database Summary:**  
- 65 high-resolution images of card arrangements.  
- Images depict legal Blackjack hands, ensuring no card repetition.  
- Each image is labeled with the point values of the dealer and players' stacks.  

Sample images and templates were created for card recognition using template matching techniques.  

---

## System Workflow  
1. **Preprocessing**  
   - Grayscale conversion  
   - CLAHE for contrast enhancement  
   - Gaussian filtering for noise reduction  

2. **Segmentation**  
   - Adaptive thresholding (Otsu’s method) for binary image creation  
   - Card stack detection and contour-based segmentation  
   - Missing edge reconstruction for overlapping cards  

![Segmentation image cards](https://github.com/SuperrMurlocc/aoc-projekt/blob/main/outputs_results/more_segmentation.png)

3. **Card Recognition**  
   - Perspective transformation for accurate card alignment  
   - Template matching for symbol recognition  

4. **Game Outcome Evaluation**  
   - Conversion of card symbols to points  
   - Summing up the points for each stack  
   - Determining the winner based on the dealer's and players' scores  

---

## Technologies Used  
- **Python 3**  
- **OpenCV** – for image processing and feature detection  
- **NumPy** – for efficient numerical operations  
- **Matplotlib** – for visualization  
- **einops** – for tensor manipulation  
- **Git/GitHub** – for version control and collaboration  

---

## Results  
The system achieved **100% accuracy** for full-resolution images and **96.875% accuracy** for images scaled down 4 times. The processing time for one image is approximately **0.31 seconds**, making it suitable for real-time applications.  

| Width and Height Scale | Exact Accuracy | Positional Accuracy | Average Processing Time |  
|-------------|----------------|---------------------|-------------------------|  
| 1.0         | 100%           | 100%               | 0.310 s                |  
| 0.75        | 100%           | 100%               | 0.283 s                |  
| 0.5         | 96.875%        | 99.34%             | 0.230 s                |  
| 0.25        | 43.75%         | 79.54%             | 0.206 s                |  

---

## Limitations and Future Work  
While the system works well with high-quality images, it struggles in the following scenarios:  
- **Low-quality images** with reflections, poor lighting, or overlapping cards in non-standard orientations.  
- **Images with rotated cards** beyond typical orientations.  

Future improvements could focus on enhancing robustness against low-quality images and developing a more generalized algorithm for real-world applications in casinos.  

---

## Contributors  
- **Jakub Bednarski** – Conceptualization, Methodology, Software Development, Project Administration  
- **Julia Komorowska** – Software Development, Investigation  
- **Hubert Woziński** – Software Development, Investigation  

---

## License
This project is licensed under the MIT License.

