import cv2
from misc import level_box_coord, round_box_coord

# ==== Load in Video ==== #
vid = cv2.VideoCapture("../vid/vid_1080.mp4")

# ==== Set up Parameters ==== #
count = 0
level = 0
fps = vid.get(cv2.CAP_PROP_FPS)


while (vid.isOpened()):
    # ==== Setup ==== #
    ret, frame = vid.read()
    y, x, _ = frame.shape
    time = round(count/fps, 4)

    # ==== Get ROI ==== #
    # Level Box
    level_coord_1, level_coord_2 = level_box_coord(x, y)
    level_box = frame[level_coord_1[1]:level_coord_2[1], level_coord_1[0]:level_coord_2[0]]
    level_box = cv2.resize(level_box, (300, 300), interpolation=cv2.INTER_AREA)

    # Experience Level Box
    round_coord_1, round_coord_2 = round_box_coord(x, y)
    round_box = frame[round_coord_1[1]:round_coord_2[1], round_coord_1[0]:round_coord_2[0]]
    round_box = cv2.resize(round_box, (200, 100))


    # ==== Show ROI and video ==== #
    # Draw ROI on video
    cv2.rectangle(frame, level_coord_1, level_coord_2, (255, 0, 0), 2)
    cv2.rectangle(frame, round_coord_1, round_coord_2, (255, 0, 0), 2)

    # Show video
    cv2.imshow("Video", frame)
    cv2.imshow("Level", level_box)
    cv2.imshow("Round", round_box)

    # ==== Write image out ==== #
    # Print current level
    print(f"Current Level: {level}, Current Frame: {count}")
    if level >= 2:
        path = f"./data/{level}/{count}.jpg"
        cv2.imwrite(path, level_box)
        print(f"Print to {path}")


    # ==== User inputs ==== #
    if cv2.waitKey(1) & 0xFF == ord("q"):
        level += 1

    count += 1


# ==== When end ==== #
vid.release()
cv2.destroyAllWindows()