from typing import Tuple

import numpy as np
from numba import njit, prange

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Strides,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def _tensor_conv1d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    
    """
    Реализация 1D-свёртки.

    Дан входной тензор вида

       `batch, in_channels, width`

    и тензор весов вида

       `out_channels, in_channels, k_width`

    Вычисляет дополненный (padded) выход вида

       `batch, out_channels, width`

    `reverse` определяет, закреплён ли вес слева (False) или справа.
    (См. диаграммы)

    Args:
        out (Storage): хранилище для тензора `out`.
        out_shape (Shape): форма тензора `out`.
        out_strides (Strides): шаги (strides) тензора `out`.
        out_size (int): размер тензора `out`.
        input (Storage): хранилище для тензора `input`.
        input_shape (Shape): форма тензора `input`.
        input_strides (Strides): шаги (strides) тензора `input`.
        weight (Storage): хранилище для тензора `input`.
        weight_shape (Shape): форма тензора `input`.
        weight_strides (Strides): шаги (strides) тензора `input`.
        reverse (bool): закрепить вес слева или справа
    """
    # Распаковываем форму выходного тензора.
    # batch_ — размер батча в out.
    # out_channels — количество выходных каналов.
    # out_width — ширина результата.
    batch_, out_channels, out_width = out_shape

    # Распаковываем форму входного тензора.
    # batch — размер батча в input.
    # in_channels — количество входных каналов.
    # width — ширина входа.
    batch, in_channels, width = input_shape

    # Распаковываем форму весов.
    # out_channels_ — количество выходных каналов в weight.
    # in_channels_ — количество входных каналов в weight.
    # kw — ширина ядра свёртки.
    out_channels_, in_channels_, kw = weight_shape

    # Проверяем, что формы совместимы:
    # batch input должен совпадать с batch output,
    # input channels должны совпадать с channels в weight,
    # output channels должны совпадать с channels в weight.
    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    # Сохраняем strides входного тензора в короткую переменную.
    s1 = input_strides

    # Сохраняем strides весов в короткую переменную.
    s2 = weight_strides

    # Strides для input.
    # s10 — шаг по batch.
    # s11 — шаг по input channel.
    # s12 — шаг по width.
    s10, s11, s12 = s1[0], s1[1], s1[2]

    # Strides для weight.
    # s20 — шаг по output channel.
    # s21 — шаг по input channel.
    # s22 — шаг по kernel width.
    s20, s21, s22 = s2[0], s2[1], s2[2]

    # Strides для output.
    # so0 — шаг по batch.
    # so1 — шаг по output channel.
    # so2 — шаг по output width.
    so0, so1, so2 = out_strides[0], out_strides[1], out_strides[2]

    # Проходим по всем элементам выходного тензора.
    # prange — параллельный range от numba.
    for ordinal in prange(out_size):

        # Переводим плоский индекс ordinal в индекс batch.
        # В одном batch лежит out_channels * out_width элементов.
        b = ordinal // (out_channels * out_width)

        # rem — остаток внутри одного batch.
        rem = ordinal - b * out_channels * out_width

        # Из остатка получаем output channel.
        # В одном output channel лежит out_width элементов.
        oc = rem // out_width

        # Из остатка получаем позицию по ширине output.
        ow = rem - oc * out_width

        # acc — аккумулятор суммы свёртки для одного элемента out[b, oc, ow].
        acc = 0.0

        # Проходим по всем входным каналам.
        for ic in range(in_channels):

            # Проходим по всем элементам ядра свёртки.
            for k in range(kw):

                # Если reverse=True, ядро закреплено справа.
                # Тогда смотрим влево: ow - k.
                if reverse:
                    iw = ow - k

                # Если reverse=False, ядро закреплено слева.
                # Тогда смотрим вправо: ow + k.
                else:
                    iw = ow + k

                # Проверяем padding-границы.
                # Если iw вышел за input, такой элемент просто пропускаем.
                if 0 <= iw < width:

                    # Считаем позицию input[b, ic, iw] в плоском storage.
                    input_pos = b * s10 + ic * s11 + iw * s12

                    # Считаем позицию weight[oc, ic, k] в плоском storage.
                    weight_pos = oc * s20 + ic * s21 + k * s22

                    # Добавляем произведение input и weight к сумме.
                    acc += input[input_pos] * weight[weight_pos]

        # Считаем позицию out[b, oc, ow] в плоском storage.
        out_pos = b * so0 + oc * so1 + ow * so2

        # Записываем итоговую сумму в output.
        out[out_pos] = acc


tensor_conv1d = njit(parallel=True)(_tensor_conv1d)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
            batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    Реализация 2D-свёртки.

    Дан входной тензор вида

       `batch, in_channels, height, width`

    и тензор весов вида

       `out_channels, in_channels, k_height, k_width`

    Вычисляет дополненный (padded) выход вида

       `batch, out_channels, height, width`

    `reverse` определяет, закреплён ли вес в левом верхнем углу (False)
    или в правом нижнем углу (True).
    (См. диаграммы)


    Args:
        out (Storage): хранилище для тензора `out`.
        out_shape (Shape): форма тензора `out`.
        out_strides (Strides): шаги (strides) тензора `out`.
        out_size (int): размер тензора `out`.
        input (Storage): хранилище для тензора `input`.
        input_shape (Shape): форма тензора `input`.
        input_strides (Strides): шаги (strides) тензора `input`.
        weight (Storage): хранилище для тензора `weight`.
        weight_shape (Shape): форма тензора `weight`.
        weight_strides (Strides): шаги (strides) тензора `weight`.
        reverse (bool): закрепить вес в левом верхнем или правом нижнем углу
    """

    # Достаём размеры выходного тензора:
    # batch_ — количество объектов в batch,
    # out_channels — количество выходных каналов,
    # out_height — высота выхода,
    # out_width — ширина выхода.
    batch_, out_channels, out_height, out_width = out_shape

    # Достаём размеры входного тензора:
    # batch — количество объектов в batch,
    # in_channels — количество входных каналов,
    # height — высота входа,
    # width — ширина входа.
    batch, in_channels, height, width = input_shape

    # Достаём размеры тензора весов:
    # out_channels_ — количество выходных каналов,
    # in_channels_ — количество входных каналов,
    # kh — высота ядра,
    # kw — ширина ядра.
    out_channels_, in_channels_, kh, kw = weight_shape

    # Проверяем, что размеры согласованы:
    # batch у входа и выхода должен совпадать,
    # количество входных каналов у input и weight должно совпадать,
    # количество выходных каналов у out и weight должно совпадать.
    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    # Сохраняем strides входного тензора в короткую переменную.
    s1 = input_strides

    # Сохраняем strides тензора весов в короткую переменную.
    s2 = weight_strides

    # Strides для input:
    # s10 — шаг по batch,
    # s11 — шаг по input channel,
    # s12 — шаг по height,
    # s13 — шаг по width.
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]

    # Strides для weight:
    # s20 — шаг по output channel,
    # s21 — шаг по input channel,
    # s22 — шаг по kernel height,
    # s23 — шаг по kernel width.
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    # Strides для output:
    # so0 — шаг по batch,
    # so1 — шаг по output channel,
    # so2 — шаг по height,
    # so3 — шаг по width.
    so0, so1, so2, so3 = out_strides[0], out_strides[1], out_strides[2], out_strides[3]

    # Идём параллельно по каждому элементу выходного тензора.
    for ordinal in prange(out_size):

        # Находим индекс batch для текущего ordinal.
        b = ordinal // (out_channels * out_height * out_width)

        # Убираем часть ordinal, отвечающую за batch.
        rem = ordinal - b * out_channels * out_height * out_width

        # Находим индекс выходного канала.
        oc = rem // (out_height * out_width)

        # Убираем часть rem, отвечающую за output channel.
        rem = rem - oc * out_height * out_width

        # Находим координату по высоте в выходном тензоре.
        oh = rem // out_width

        # Находим координату по ширине в выходном тензоре.
        ow = rem - oh * out_width

        # Здесь будем накапливать сумму свёртки.
        acc = 0.0

        # Перебираем все входные каналы.
        for ic in range(in_channels):

            # Перебираем высоту ядра.
            for kh_i in range(kh):

                # Перебираем ширину ядра.
                for kw_i in range(kw):

                    # Если reverse=True, ядро закреплено в правом нижнем углу.
                    if reverse:
                        # Тогда по высоте идём назад от текущей позиции выхода.
                        ih = oh - kh_i

                        # И по ширине тоже идём назад.
                        iw = ow - kw_i

                    # Если reverse=False, ядро закреплено в левом верхнем углу.
                    else:
                        # Тогда по высоте идём вперёд от текущей позиции выхода.
                        ih = oh + kh_i

                        # И по ширине тоже идём вперёд.
                        iw = ow + kw_i

                    # Проверяем, что координаты input не вышли за границы.
                    if 0 <= ih < height and 0 <= iw < width:

                        # Считаем позицию нужного элемента во входном storage.
                        input_pos = b * s10 + ic * s11 + ih * s12 + iw * s13

                        # Считаем позицию нужного веса в weight storage.
                        weight_pos = oc * s20 + ic * s21 + kh_i * s22 + kw_i * s23

                        # Добавляем произведение input на weight в сумму.
                        acc += input[input_pos] * weight[weight_pos]

        # Считаем позицию текущего элемента output в storage.
        out_pos = b * so0 + oc * so1 + oh * so2 + ow * so3

        # Записываем накопленную сумму в output.
        out[out_pos] = acc


tensor_conv2d = njit(parallel=True, fastmath=True)(_tensor_conv2d)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
