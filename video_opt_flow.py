import cv2
import numpy as np
import pandas as pd
import argparse
from graph import get_graph
import motion_magnification as mm
import mediapy as media


# Ввод данных в командной строке
parser = argparse.ArgumentParser(description='A program for building a pseudopulse wave of a person on video')
parser.add_argument('out_path', type=str, help='The path to save the time series')
parser.add_argument('in_path', type=str, help='Video for processing')
args = parser.parse_args()


# Параметры для усиления видео
magnification_factor = 4
fl = .04
fh = .4
fs = 1
attenuate_other_frequencies=True
pyr_type = "octave"
sigma = 0
temporal_filter = mm.difference_of_iir
scale_video = .8

# Производство усиленного видео
video_f = mm.load_video(args.in_path)
amplified_video = mm.phase_amplify(video_f, magnification_factor, fl, fh, fs, attenuate_other_frequencies=attenuate_other_frequencies, pyramid_type=pyr_type, sigma=sigma, temporal_filter=temporal_filter)
media.write_video('amplified_1.mp4', amplified_video)


# ЗАМЕНИТЬ args.in_path на путь к усиленному видео
cap = cv2.VideoCapture('amplified_1.mp4')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

ret, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

lk_params = dict(winSize=(30, 30),
                 maxLevel=2,
                 criteria=(cv2.TermCriteria_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def select_point(gray):
    """
    Отслеживает новое положение точек интереса в кадре
    :param gray: изображение фрейма в ЧБ
    :return:
    """
    global point, point_selected, old_points, points_arr, old_points_arr, eyes_found
    faces = face_cascade.detectMultiScale(gray, 1.1, 19)

    for (x, y, w, h) in faces:
        if len(faces > 0):
            cv2.rectangle(frame2, (x, y), (x + w, y+h), (255, 0, 0), 2)
            middle_x = (2 * faces[0][0] + faces[0][2]) / 2
            middle_y = (2 * faces[0][1] + faces[0][3]) / 3
            arr_x = np.random.normal(middle_x, 25, size=10)
            arr_y = np.random.normal(middle_y, 10, size=10)
            for i in range(len(arr_x)):
                point = (int(arr_x[i]), int(arr_y[i]))
                points_arr.append(point)
                point_selected = True
                old_points = np.array([[arr_x[i], arr_y[i]]], dtype=np.float32)
                old_points_arr.append(old_points)
            vectors.append(np.zeros(np.array(points_arr).shape))
            v_time.append(0)


def new_time_series(points, time):
    """
    Создание файла с временным рядом
    :param points: координаты точек интереса
    :param time: время измерения
    :return:
    """
    n_steps, n_vectors = points.shape[:2]
    data = {'time': time}
    for i in range(n_vectors):
        label = "vector_" + str(i+1)
        data[label] = list(np.round(points[:, i, 1], 2))
    return pd.DataFrame(data)


point_selected = False
point = ()
old_points = np.array([[]])
old_points_arr = []
points_arr = []
eyes_found = False

fps = cap.get(cv2.CAP_PROP_FPS)
k_fr = 0
v_time = []
vectors = []
while ret:
    k_fr += 1
    ret, frame = cap.read()
    try:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame2 = frame.copy()
    except Exception:
        break
    if not point_selected:
        select_point(gray_frame)
    new_points_arr = []

    if point_selected:

        for p in points_arr:
            cv2.circle(frame2, p, 5, (0, 0, 255), 2)

        for op in old_points_arr:
            new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, op, None, **lk_params)
            new_points_arr.append(new_points)

        v_time.append(round(k_fr / fps, 2))
        move_vector = np.array(points_arr) - np.array(new_points_arr).reshape(-1, 2)

        vectors.append(move_vector)
        old_gray = gray_frame.copy()
        old_points_arr = new_points_arr.copy()

        for new_points in new_points_arr:
            x, y = new_points.ravel()
            cv2.circle(frame2, (int(x), int(y)), 5, (0, 255, 0), -1)

    cv2.imshow("Test", frame2)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

time_series = new_time_series(np.array(vectors), v_time)
time_series.to_csv(path_or_buf=r"\time_series", index=False)

cap.release()
cv2.destroyAllWindows()

# Построение графиков
series = pd.read_csv(r'\time_series', header=0)
vectors = series.drop('time', axis=1)
time = series['time']

get_graph(time, vectors)