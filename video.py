import pytesseract
import cv2
import re
from misc import onClick, level_box_coord, round_box_coord
import numpy as np

# ==== Initial Setup ==== #
# Load in video
vid = cv2.VideoCapture('./vid/vid_1080.mp4')
start_frame = 5000
vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
# Get Vid info
fps = vid.get(cv2.CAP_PROP_FPS)
count = start_frame
# Pytesseract setup
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Initalize comparing variable
previous_level_box = None
previous_round_box = None
previous_level = None
previous_round = None

while (vid.isOpened()):
    # ==== Read in frame ==== #
    ret, frame = vid.read()
    if frame is not None:
        y, x, _ = frame.shape
    else:
        print("End of video")
        break
    time = int(count / fps)
    img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ==== Extract ROI ==== #
    # Show Level Box
    level_coord_1, level_coord_2 = level_box_coord(x, y)
    level_box = img_grey[level_coord_1[1]:level_coord_2[1], level_coord_1[0]:level_coord_2[0]]
    level_box = cv2.resize(level_box, (200, 100), interpolation=cv2.INTER_AREA)
    level_box = cv2.bilateralFilter(level_box, 9, 200, 100)

    # Experience Level Box
    round_coord_1, round_coord_2 = round_box_coord(x, y)
    round_box = img_grey[round_coord_1[1]:round_coord_2[1], round_coord_1[0]:round_coord_2[0]]
    round_box = cv2.resize(round_box, (200, 100))
    round_box = cv2.GaussianBlur(round_box, (11, 11), 0)
    round_box = cv2.medianBlur(round_box, 9)

    # ==== OCR ==== #
    # If first run then set some value
    if previous_round_box is None:
        previous_level_box = level_box
        previous_round_box = round_box
        level = pytesseract.image_to_string(level_box, config='--psm 7 digits').replace(".", "").strip()
        round_ = pytesseract.image_to_string(round_box, config='--psm 7 digits').replace(".", "").strip()
        print(f"Initial set: Level: {level}, Round: {round_}")

    # Compare current and previous frame
    ## Level
    diff_level = cv2.absdiff(previous_level_box, level_box).astype(np.uint8)
    diff_level = round(np.count_nonzero(diff_level)/ diff_level.size, 6)
    ## Round
    diff_round = cv2.absdiff(previous_round_box, round_box).astype(np.uint8)
    diff_round = round(np.count_nonzero(diff_round) / diff_round.size, 6)

    # Perform OCR if difference is big enough
    if diff_round > 0.2 or (0.5 < diff_level < 0.8):
        trigger_round = diff_round > 0.2
        trigger_level = 0.5 < diff_level < 0.8
        level = pytesseract.image_to_string(level_box, config='--psm 7 digits').replace(".", "")
        round_ = pytesseract.image_to_string(round_box, config='--psm 7 digits')
        # Check if input is valid
        round_format_bool = bool(re.match(r"\d{1}\-\d{1}", round_))
        level_format_bool = bool(re.match(r"\d{1}", level))
        if round_format_bool and level_format_bool:
            if previous_level != level or previous_round != round_:
                print(f"Frame: {count}, Level: {level}, Round: {round_}")
                previous_round = round_
                previous_level = level
    # Set pervious frame and current frame
    previous_level_box = level_box
    previous_round_box = round_box

    """
    Output stuff
    """
    # Draw on plot
    cv2.rectangle(frame, level_coord_1, level_coord_2, (255, 0, 0), 2)
    cv2.rectangle(frame, round_coord_1, round_coord_2, (255, 0, 0), 2)

    # Custom Input
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('w'):
        cv2.imwrite("./img/round.jpg", round_box)
        cv2.imwrite("./img/level_box.jpg", level_box)
        print("Wrote image")

    # Show Video
    cv2.imshow('frame', frame)
    cv2.setMouseCallback('frame', onClick)

    # Show box
    cv2.imshow("Level", level_box)
    cv2.imshow("Round", round_box)

    # Increase count
    count += 1

vid.release()
cv2.destroyAllWindows()

# Tmp stuff
# """
# OCR
# """
# if previous_round_box is None or previous_level_box is None:
#     previous_round_box = round_box
#     previous_level_box = level_box
#     # round_string = pytesseract.image_to_string(round_box, config='digits')
#     # level_string = pytesseract.image_to_string(level_box, config='digits')
#     previous_level = pytesseract.image_to_string(level_box, config='--psm 8 digits')
#
# diff_round_box = cv2.absdiff(previous_round_box, round_box).astype(np.uint8)
# diff_round_box = np.count_nonzero(diff_round_box) / diff_round_box.size
# diff_level_box = cv2.absdiff(previous_level_box, level_box)
# diff_level_box = np.count_nonzero(diff_level_box) / diff_level_box.size
# if diff_round_box > 0.2 and diff_level_box > 0.2:
#     round_string = pytesseract.image_to_string(round_box, config='--psm 7 digits').replace(".", "")
#     level_string = pytesseract.image_to_string(level_box, config='--psm 8 digits')
#     level_string = re.findall('\d+', level_string)
#     print(f"Round: {round_string} ({round(diff_round_box * 100, 2)}%), "
#           f"Level: {level_string} ({round(diff_level_box * 100, 2)}%), "
#           f"Frame: {vid.get(cv2.CAP_PROP_POS_FRAMES)}")
#
#     if level_string == []:
#         level_string = None
#     else:
#         level_string = level_string[0]
#     if level_string is None or re.match('^\d{1}$', round_string):
#         pass
#     elif previous_level != level_string or previous_round != round_string:
#         previous_level = level_string
#         previous_round = round_string
#         # print(f"Round: {round_string} ({round(diff_round_box*100, 2)}%), "
#         #       f"Level: {level_string} ({round(diff_level_box*100, 2)}%), "
#         #       f"Frame: {vid.get(cv2.CAP_PROP_POS_FRAMES)}")
#
# previous_round_box = round_box
# previous_level_box = level_box
