from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Изменяет форму тензора изображения для 2D pooling.

    Args:
        input: тензор формы batch x channel x height x width
        kernel: высота x ширина окна pooling

    Returns:
        Тензор размера batch x channel x new_height x new_width x
        (kernel_height * kernel_width), а также значения new_height и new_width.
    """

    # Достаём размеры входного тензора:
    # batch — количество объектов в batch,
    # channel — количество каналов,
    # height — высота изображения,
    # width — ширина изображения.
    batch, channel, height, width = input.shape

    # Достаём размеры ядра pooling:
    # kh — высота окна,
    # kw — ширина окна.
    kh, kw = kernel

    # Проверяем, что высота делится на высоту окна без остатка.
    assert height % kh == 0

    # Проверяем, что ширина делится на ширину окна без остатка.
    assert width % kw == 0

    # Новая высота после pooling.
    new_height = height // kh

    # Новая ширина после pooling.
    new_width = width // kw

    # Делаем input contiguous, чтобы его можно было безопасно менять через view.
    out = input.contiguous()

    # Разбиваем height на две части:
    # new_height и kh.
    # Разбиваем width на две части:
    # new_width и kw.
    #
    # Было:
    # batch x channel x height x width
    #
    # Стало:
    # batch x channel x new_height x kh x new_width x kw
    out = out.view(batch, channel, new_height, kh, new_width, kw)

    # Меняем порядок измерений так, чтобы kh и kw оказались рядом в конце.
    #
    # Было:
    # batch x channel x new_height x kh x new_width x kw
    #
    # Стало:
    # batch x channel x new_height x new_width x kh x kw
    out = out.permute(0, 1, 2, 4, 3, 5)

    # Снова делаем contiguous после permute,
    # потому что permute меняет strides, а view требует contiguous tensor.
    out = out.contiguous()

    # Склеиваем kh и kw в одно последнее измерение.
    #
    # Было:
    # batch x channel x new_height x new_width x kh x kw
    #
    # Стало:
    # batch x channel x new_height x new_width x (kh * kw)
    out = out.view(batch, channel, new_height, new_width, kh * kw)

    # Возвращаем tiled tensor и новые размеры height/width.
    return out, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    2D average pooling через разбиение тензора на окна.

    Args:
        input : тензор формы batch x channel x height x width
        kernel : высота x ширина окна pooling

    Returns:
        Тензор после average pooling
    """

    # Достаём размеры входного тензора.
    batch, channel, height, width = input.shape

    # Применяем tile:
    # tiled получает форму
    # batch x channel x new_height x new_width x kernel_size.
    tiled, new_height, new_width = tile(input, kernel)

    # Берём среднее по последнему измерению,
    # то есть усредняем элементы внутри каждого pooling-окна.
    out = tiled.mean(4)

    # mean оставляет размерность, по которой делали reduction, как 1.
    # Поэтому форма будет:
    # batch x channel x new_height x new_width x 1
    #
    # Нам нужна форма:
    # batch x channel x new_height x new_width.
    out = out.view(batch, channel, new_height, new_width)

    # Возвращаем результат average pooling.
    return out


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"

        # dim приходит как Tensor, поэтому достаём из него обычное число int.
        dim_int = int(dim.item())

        # Сохраняем input и dim_int, потому что они понадобятся в backward.
        ctx.save_for_backward(input, dim_int)

        # max_reduce применяет операцию max вдоль нужной размерности.
        return max_reduce(input, dim_int)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"

        # Достаём сохранённые input и dim_int из forward.
        input, dim_int = ctx.saved_values

        # argmax(input, dim_int) даёт 1 там, где был максимум, и 0 в остальных местах.
        mask = argmax(input, dim_int)

        # Градиент проходит только через максимальный элемент.
        return grad_output * mask, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Вычисляет softmax для тензора.

    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : входной тензор
        dim : размерность, по которой применяется softmax

    Returns:
        тензор softmax
    """

    # Для численной стабильности сначала вычитаем максимум по dim.
    # Это не меняет результат softmax, но защищает exp от слишком больших чисел.
    shifted = input - max(input, dim)

    # Возводим e в степень каждого элемента shifted.
    exp_values = shifted.exp()

    # Суммируем exp_values вдоль нужной размерности.
    exp_sum = exp_values.sum(dim)

    # Делим каждый exp на сумму exp по этой же размерности.
    return exp_values / exp_sum

def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Вычисляет log-softmax для тензора.

    $z_i = x_i - \log \sum_i e^{x_i}$

    Используется log-sum-exp trick для численной стабильности.

    Args:
        input : входной тензор
        dim : размерность, по которой применяется log-softmax

    Returns:
         логарифм softmax
    """

    # Для стабильности вычитаем максимум по выбранной размерности.
    shifted = input - max(input, dim)

    # Считаем сумму exp вдоль нужной размерности.
    exp_sum = shifted.exp().sum(dim)

    # logsoftmax = shifted - log(sum(exp(shifted))).
    return shifted - exp_sum.log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    2D max pooling через разбиение на плитки.

    Args:
        input: тензор формы batch x channel x height x width
        kernel: высота x ширина pooling-окна

    Returns:
        Tensor : тензор после max pooling
    """

    # Достаём размеры входного тензора:
    # batch — количество объектов,
    # channel — количество каналов,
    # height — высота,
    # width — ширина.
    batch, channel, height, width = input.shape

    # Разбиваем входной тензор на pooling-окна.
    # tiled имеет форму:
    # batch x channel x new_height x new_width x kernel_size
    tiled, new_height, new_width = tile(input, kernel)

    # Берём максимум по последней размерности,
    # то есть внутри каждого pooling-окна.
    out = max(tiled, 4)

    # После max остаётся лишняя последняя размерность размера 1:
    # batch x channel x new_height x new_width x 1
    # Убираем её через view.
    return out.view(batch, channel, new_height, new_width)

def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Случайно зануляет позиции в тензоре.

    Args:
        input : входной тензор
        rate : вероятность [0, 1] занулить каждый элемент
        ignore : если True, dropout не применяется

    Returns:
        тензор со случайно занулёнными позициями
    """

    # Если ignore=True, ничего не делаем и возвращаем input как есть.
    if ignore:
        return input

    # Создаём случайный тензор той же формы, что и input.
    noise = rand(input.shape, backend=input.backend)

    # mask будет 1 там, где случайное число больше rate,
    # и 0 там, где элемент надо занулить.
    mask = noise > rate

    # Умножаем input на mask: часть элементов остаётся, часть становится 0.
    return input * mask