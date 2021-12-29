import cv2

def draw_H_line(frame, x, y, lenght, color) -> None:
    frame[y,x:x+lenght] = color

def draw_V_line(frame, x, y, lenght, color) -> None:
    frame[y:y+lenght, x] = color

def draw_rect(frame, pt1, pt2, color) -> None:
    horizontal_lenght = pt2[1] - pt1[1]
    vertical_lenght = pt2[0] - pt1[0]
    draw_H_line(frame, pt1[1], pt1[0], horizontal_lenght, color)
    draw_H_line(frame, pt1[1], pt2[0], horizontal_lenght, color)
    draw_V_line(frame, pt1[1], pt1[0], vertical_lenght, color)
    draw_V_line(frame, pt2[1], pt1[0], vertical_lenght, color)

def draw_limit(frame, limit, roi) -> None:
    draw_H_line(frame, roi[0][1], limit[0], roi[1][1]-roi[0][1], [0,0,255])
    draw_H_line(frame, roi[0][1], limit[1], roi[1][1]-roi[0][1], [0,0,255])

def draw_pred(frame, levels, foam, roi) -> None:
    num_lev = len(levels)-1
    #draw levels
    for i, lev in enumerate(levels):
        center = lev['center'] + roi[0][0]
        if i == num_lev:

            draw_H_line(frame, roi[0][1], center, 40, [255,0,0])
            cv2.arrowedLine(frame, (frame.shape[1]-50, center), (roi[1][1] + 2, center), (0,0,255), 2)
            score = '%d%%' % int(lev['score']*100)
            cv2.putText(frame, score, (frame.shape[1]-48, center+2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
        else:
            draw_H_line(frame, roi[0][1], center, 40, [200,125,0])
            score = '%d%%' % int(lev['score']*100)
            cv2.putText(frame, score, (roi[1][1]+20, center+2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20,20,20), 1, cv2.LINE_AA)
    #draw foam
    for val in foam:
        draw_V_line(frame, roi[1][1]+3, val['start_pt'], val['end_pt']-val['start_pt'], [255,0,0])

def draw_passed(frame, color) -> None:
    cv2.circle(frame, (200,20), 15, color, -1)

def draw_true_lev(frame, lev, roi) -> None:
    if lev:
        lev = lev + roi[0][0]
        start_point = (65, lev)
        end_point = (roi[0][1] - 2, lev)
        cv2.arrowedLine(frame, start_point, end_point, (0,0,255), 2)
        start_point_txt = (10, lev + 2)
        cv2.putText(frame, 'true lev', start_point_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
