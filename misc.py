import cv2


# Functions
def onClick(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(f"x = {x}({100*x/img_max_x}%, y = {y}({100*y/img_max_y}%)")
        x_pct = round(x * 100 / 1280, 2)
        y_pct = round(y * 100 / 720, 2)
        print(f"({x}, {y}), (x: {x_pct}%, y: {y_pct}%)")
        mouseX, mouseY = x, y


def level_box_coord(shape_x, shape_y):
    #x1, y1 = int(0.1438 * shape_x), int(0.8111 * shape_y)
    x1, y1 = int(0.1641 * shape_x), int(0.8111 * shape_y)
    x2, y2 = int(0.1730 * shape_x), int(0.8389 * shape_y)
    return (x1, y1), (x2, y2)


def round_box_coord(shape_x, shape_y):
    x1, y1 = int(0.404 * shape_x), int(0.01 * shape_y)
    x2, y2 = int(0.4250 * shape_x), int(0.0306 * shape_y)
    return (x1, y1), (x2, y2)