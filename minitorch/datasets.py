import math
import random
from dataclasses import dataclass
from typing import List, Tuple

'''
    Генерирует N случайных точек в квадрате [0, 1] × [0, 1].
    Каждая точка — это пара (x1, x2), где оба координат равномерно распределены.
'''
def make_pts(N: int) -> List[Tuple[float, float]]:
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    '''
        Простая структура для хранения датасета:
        N — количество точек,
        X — список координат,
        y — список меток классов (0 или 1).
    '''
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    '''
        Линейно разделимый датасет.
        Класс 1, если x1 < 0.5, иначе класс 0.
        Граница — вертикальная линия x = 0.5.
    '''
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    '''
        Линейно разделимый датасет.
        Класс 1, если x1 + x2 < 0.5, иначе класс 0.
        Граница — диагональная линия x1 + x2 = 0.5.
    '''
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    '''
        Нелинейно разделимый датасет.
        Класс 1, если x1 < 0.2 или x1 > 0.8.
        Получаются две вертикальные полосы по краям.
        Одной прямой разделить невозможно.
    '''
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    '''
        Классический XOR.
        Класс 1, если точка в одном из двух противоположных квадрантов:
        (x1 < 0.5, x2 > 0.5) или (x1 > 0.5, x2 < 0.5).
        Линейно неразделимый датасет.
    '''
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    '''
        Круговая граница.
        Центр круга — (0.5, 0.5).
        Класс 1, если точка находится вне круга радиуса sqrt(0.1),
        иначе класс 0.
        Нелинейно разделимый датасет.
    '''
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    '''
        Две закрученные спирали — один из самых сложных датасетов.
        Первая спираль — класс 0, вторая — класс 1.
        Линейно и даже просто нелинейно разделить сложно.
        Используется для демонстрации возможностей нейросетей.
    '''
    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
