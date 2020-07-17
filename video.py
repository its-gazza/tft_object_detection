import pytesseract
import cv2
import re
from misc import onClick, level_box_coord, round_box_coord
import numpy as np

vid = cv2.VideoCapture('./vid/vid_1080.mp4')
vid.set(cv2.CAP_PROP_POS_FRAMES, 30537)
fps = vid.get(cv2.CAP_PROP_FPS)
count = 30537
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
previous_level_box = None
previous_round_box = None
previous_level = -1
previous_round = "1-1"

while (vid.isOpened()):
    ret, frame = vid.read()
    y, x, _ = frame.shape
    time = int(count / fps)
    img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    """
    Extract ROI Box
    """
    # Show Level Box
    level_coord_1, level_coord_2 = level_box_coord(x, y)
    level_box = img_grey[level_coord_1[1]:level_coord_2[1], level_coord_1[0]:level_coord_2[0]]
    level_box = cv2.resize(level_box, (100, 200))
    level_box = cv2.GaussianBlur(level_box, (11, 11), 0)
    level_box = cv2.medianBlur(level_box, 9)
    level_box = cv2.GaussianBlur(level_box, (11, 11), 0)
    level_box = cv2.medianBlur(level_box, 9)
    kernel = np.ones((1, 1), np.uint8)
    level_box = cv2.dilate(level_box, kernel, iterations=1)
    level_box = cv2.erode(level_box, kernel, iterations=1)
    cv2.threshold(cv2.GaussianBlur(level_box, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.threshold(cv2.bilateralFilter(level_box, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.threshold(cv2.medianBlur(level_box, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.adaptiveThreshold(cv2.GaussianBlur(level_box, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31,
                          2)
    cv2.adaptiveThreshold(cv2.bilateralFilter(level_box, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                          31, 2)
    cv2.adaptiveThreshold(cv2.medianBlur(level_box, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # Experience Level Box
    round_coord_1, round_coord_2 = round_box_coord(x, y)
    round_box = img_grey[round_coord_1[1]:round_coord_2[1], round_coord_1[0]:round_coord_2[0]]
    round_box = cv2.resize(round_box, (200, 100))
    # Box is too blurry, need to sharpen
    round_box = cv2.GaussianBlur(round_box, (11, 11), 0)
    round_box = cv2.medianBlur(round_box, 9)


    """
    OCR
    """
    if previous_round_box is None or previous_level_box is None:
        previous_round_box = round_box
        previous_level_box = level_box
        # round_string = pytesseract.image_to_string(round_box, config='digits')
        # level_string = pytesseract.image_to_string(level_box, config='digits')
        previous_level = pytesseract.image_to_string(level_box, config='--psm 7 digits')

    diff_round_box = cv2.absdiff(previous_round_box, round_box).astype(np.uint8)
    diff_round_box = np.count_nonzero(diff_round_box)/ diff_round_box.size
    diff_level_box = cv2.absdiff(previous_level_box, level_box)
    diff_level_box = np.count_nonzero(diff_level_box) / diff_level_box.size

    if diff_round_box > 0.2 and diff_level_box > 0.2:
        round_string = pytesseract.image_to_string(round_box, config='--psm 7').replace(".", "")
        level_string = pytesseract.image_to_string(level_box, config='--psm 8')
        level_string = re.findall('\d+', level_string)
        print(f"Round: {round_string} ({round(diff_round_box * 100, 2)}%), "
              f"Level: {level_string} ({round(diff_level_box * 100, 2)}%), "
              f"Frame: {vid.get(cv2.CAP_PROP_POS_FRAMES)}")

        if level_string == []:
            level_string = None
        else:
            level_string = level_string[0]
        if level_string is None or re.match('^\d{1}$', round_string):
            pass
        elif previous_level != level_string or previous_round != round_string:
            previous_level = level_string
            previous_round = round_string
            # print(f"Round: {round_string} ({round(diff_round_box*100, 2)}%), "
            #       f"Level: {level_string} ({round(diff_level_box*100, 2)}%), "
            #       f"Frame: {vid.get(cv2.CAP_PROP_POS_FRAMES)}")


    previous_round_box = round_box
    previous_level_box = level_box

    """
    Output stuff
    """
    # Draw on plot
    cv2.rectangle(frame, level_coord_1, level_coord_2, (255,0,0), 2)
    cv2.rectangle(frame, round_coord_1, round_coord_2, (255, 0, 0), 2)

    # Custom Input
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('w'):
        cv2.imwrite("./img/round.jpg", round_box)
        cv2.imwrite("./img/level_box.jpg", level_box)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        tmp_round_string = pytesseract.image_to_string(round_box, config='--psm 7 digits')
        tmp_level_string = pytesseract.image_to_string(level_box, config='--psm 7 digits')
        print(f"TEST: Previous Level: {previous_level}, Round: {tmp_round_string}, "
              f"Level: {tmp_level_string}")

    # Show Video
    # cv2.imshow('frame', frame)
    # cv2.setMouseCallback('frame', onClick)

    # Show box
    cv2.imshow("Level", level_box)
    cv2.imshow("Round", round_box)

vid.release()
cv2.destroyAllWindows()
