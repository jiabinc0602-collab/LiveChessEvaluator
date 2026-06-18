import cv2
import numpy as np

# Load the template
template = cv2.imread('templates/white_pawn.png', cv2.IMREAD_GRAYSCALE)
h, w = template.shape

print("Loading static screenshot...")
img = cv2.imread('templates/test_board.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template_edges = cv2.Canny(template, 50, 150)
board_edges = cv2.Canny(gray, 50, 150)

res = cv2.matchTemplate(board_edges, template_edges, cv2.TM_CCOEFF_NORMED)

threshold = 0.50

locations = np.where(res >= threshold)

pawn_count = 0
for pt in zip(*locations[::-1]):
    top_left = pt
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
    pawn_count += 1

print(f"Total overlapping boxes found: {pawn_count}")

cv2.namedWindow('What the Computer Sees', cv2.WINDOW_NORMAL)
cv2.imshow('What the Computer Sees', board_edges)

cv2.namedWindow('Matched', cv2.WINDOW_NORMAL)
cv2.imshow('Matched', img)

print("Press any key to close image")
cv2.waitKey(0)
cv2.destroyAllWindows()