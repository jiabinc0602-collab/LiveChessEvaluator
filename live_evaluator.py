import cv2
import mss
import numpy as np

# 1. Map piece prefixes to their exact channel in your (1, 12, 8, 8) tensor
piece_channels = {
    'wp': 0, 'wn': 1, 'wb': 2, 'wr': 3, 'wq': 4, 'wk': 5,
    'bp': 6, 'bn': 7, 'bb': 8, 'br': 9, 'bq': 10, 'bk': 11
}

# 2. Load all 24 templates into memory
raw_templates = {}
for piece in piece_channels.keys():
    raw_templates[f"{piece}_light"] = cv2.imread(f"templates/{piece}_light.png", cv2.IMREAD_GRAYSCALE)
    raw_templates[f"{piece}_dark"]  = cv2.imread(f"templates/{piece}_dark.png", cv2.IMREAD_GRAYSCALE)

# 3. Load the Neural Network
print("Loading ONNX Model...")
net = cv2.dnn.readNetFromONNX("chess_evaluator_sim.onnx")

with mss.MSS() as sct:
    monitor = sct.monitors[0]

    # --- INITIAL SETUP: DRAW THE BOX ---
    print("Capturing initial frame. Draw a box around the board and press ENTER.")
    initial_img = np.array(sct.grab(monitor))
    initial_img = cv2.cvtColor(initial_img, cv2.COLOR_BGRA2BGR)
    
    roi = cv2.selectROI("Select Chess Board", initial_img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Chess Board")

    x, y, w, h = roi
    square_size = w // 8
    print(f"Tracking locked. Square size: {square_size}px")

    # Resize all 24 templates to perfectly match the user's drawn squares
    templates = {}
    for name, img in raw_templates.items():
        if img is not None:
             templates[name] = cv2.resize(img, (square_size, square_size))
        else:
             print(f"WARNING: Could not load templates/{name}.png")

    # --- LIVE EVALUATION LOOP ---
    print("Starting Live Evaluation! Press 'q' to quit.")
    while True:
        # 1. Grab screen and crop to board
        screenshot = np.array(sct.grab(monitor))
        board_color = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)[y:y+h, x:x+w]
        board_gray = cv2.cvtColor(board_color, cv2.COLOR_BGR2GRAY)

        # 2. Initialize empty tensor
        tensor = np.zeros((1, 12, 8, 8), dtype=np.float32)

        # 3. The 8x8 Grid Slicer
        for rank in range(8):
            for file in range(8):
                # Extract the single square
                square = board_gray[rank*square_size:(rank+1)*square_size, file*square_size:(file+1)*square_size]
                
                best_score = 0.0
                best_piece = None

                # Check all 24 templates against this square
                for template_name, template_img in templates.items():
                    res = cv2.matchTemplate(square, template_img, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    
                    if max_val > best_score:
                        best_score = max_val
                        best_piece = template_name

                # If confidence is > 75%, add it to the tensor!
                if best_score > 0.75 and best_piece is not None:
                    # Strip the "_light" or "_dark" to get just "wp", "bq", etc.
                    piece_prefix = best_piece.split('_')[0] 
                    channel = piece_channels[piece_prefix]
                    tensor[0, channel, rank, file] = 1.0

        # 4. Feed the built tensor to the Neural Network
        net.setInput(tensor)
        evaluation = net.forward()[0][0] # Extract the raw win probability float
        
        # 5. Display the visual feed and the evaluation
        # We overlay the text right onto the video feed
        eval_text = f"White Win Prob: {evaluation*100:.1f}%"
        cv2.putText(board_color, eval_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow('Live Chess AI', board_color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()