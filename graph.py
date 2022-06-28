from scipy.signal import butter, lfilter, find_peaks
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    res = butter(order, [low, high], btype='band')
    return res[0], res[1]


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_graph(time, vectors):
    fig, ax = plt.subplots(3, 1)
    plt.subplots_adjust(wspace=1, hspace=1)
    plt.gcf().canvas.set_window_title("Псевдо-пульсовая волна")

    # Обычный график
    ax[0].plot(time, vectors, label=vectors.columns)
    ax[0].set_title('Отклонение от первоначальных точек')
    ax[0].set_xlabel("время, с")
    ax[0].set_ylabel("отклонение, px")
    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Фильтр Баттерворта
    lowcut = 0.75
    highcut = 5
    T = time.iloc[-1]
    n_samples = time.shape[0]
    fs = n_samples / T
    y = butter_bandpass_filter(vectors, lowcut, highcut, fs, order=5)

    # График фильтра
    ax[1].set_title('Фильтр Баттерворта')
    ax[1].set_xlabel("время, с")
    ax[1].set_ylabel("отклонение, px")
    ax[1].plot(time, y, label=vectors.columns)
    box2 = ax[1].get_position()
    ax[1].set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Метод главных компонент
    with_filter = y
    pca = PCA(n_components=1)
    y2 = pca.fit_transform(with_filter).ravel()

    # Поиск пиков
    peaks, _ = find_peaks(y2, height=0.01, width=(0.1,None), distance=30)

    # График PCA
    ax[2].set_title("Метод главных компонент")
    ax[2].set_xlabel("время, с")
    ax[2].set_ylabel("отклонение, px")
    ax[2].plot(time, y2, label="Пульс")
    ax[2].plot(time[peaks].values, y2[peaks], "x", label="Пики")
    box3 = ax[2].get_position()
    ax[2].set_position([box3.x0, box3.y0, box3.width * 0.8, box3.height])
    ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()
    return y