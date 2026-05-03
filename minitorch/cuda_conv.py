from typing import Tuple

# Импортируем CUDA из numba.
# Через него мы можем писать функции, которые выполняются на GPU.
from numba import cuda

# Context нужен для сохранения значений в forward,
# чтобы потом использовать их в backward.
from .autodiff import Context

# Tensor — основной класс тензора MiniTorch.
from .tensor import Tensor

# Shape, Storage, Strides — типы для формы, хранилища и strides тензора.
from .tensor_data import Shape, Storage, Strides

# Function — базовый класс для операций с autograd.
from .tensor_functions import Function


# Количество потоков CUDA в одном блоке.
# Каждый поток будет считать один элемент выходного тензора.
THREADS_PER_BLOCK = 256


@cuda.jit
def _tensor_conv1d_cuda(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    CUDA-реализация 1D-свёртки.

    input: batch x in_channels x width
    weight: out_channels x in_channels x kernel_width
    output: batch x out_channels x width
    """

    # Вычисляем глобальный номер текущего CUDA-потока.
    # Каждый поток отвечает за один элемент output.
    ordinal = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # Если поток вышел за размер output, ничего не делаем.
    if ordinal >= out_size:
        return

    # Достаём количество объектов в batch.
    batch = input_shape[0]

    # Достаём количество входных каналов.
    in_channels = input_shape[1]

    # Достаём ширину входного тензора.
    width = input_shape[2]

    # Достаём количество выходных каналов.
    out_channels = weight_shape[0]

    # Достаём размер ядра свёртки по ширине.
    kw = weight_shape[2]

    # Достаём ширину выходного тензора.
    out_width = out_shape[2]

    # Strides входного тензора:
    # s10 — шаг по batch,
    # s11 — шаг по input channel,
    # s12 — шаг по width.
    s10 = input_strides[0]
    s11 = input_strides[1]
    s12 = input_strides[2]

    # Strides тензора весов:
    # s20 — шаг по output channel,
    # s21 — шаг по input channel,
    # s22 — шаг по kernel width.
    s20 = weight_strides[0]
    s21 = weight_strides[1]
    s22 = weight_strides[2]

    # Strides выходного тензора:
    # so0 — шаг по batch,
    # so1 — шаг по output channel,
    # so2 — шаг по width.
    so0 = out_strides[0]
    so1 = out_strides[1]
    so2 = out_strides[2]

    # По ordinal восстанавливаем индекс batch.
    b = ordinal // (out_channels * out_width)

    # Убираем из ordinal часть, отвечающую за batch.
    rem = ordinal - b * out_channels * out_width

    # Находим индекс выходного канала.
    oc = rem // out_width

    # Находим позицию по ширине в output.
    ow = rem - oc * out_width

    # В acc будем накапливать сумму свёртки.
    acc = 0.0

    # Перебираем все входные каналы.
    for ic in range(in_channels):

        # Перебираем элементы ядра свёртки.
        for k in range(kw):

            # Если reverse=True, ядро закреплено справа.
            if reverse:
                iw = ow - k

            # Если reverse=False, ядро закреплено слева.
            else:
                iw = ow + k

            # Проверяем, что индекс input не вышел за границы.
            if 0 <= iw < width:

                # Вычисляем позицию нужного input-элемента в storage.
                input_pos = b * s10 + ic * s11 + iw * s12

                # Вычисляем позицию нужного weight-элемента в storage.
                weight_pos = oc * s20 + ic * s21 + k * s22

                # Добавляем произведение input на weight в сумму.
                acc += input[input_pos] * weight[weight_pos]

    # Вычисляем позицию текущего output-элемента в storage.
    out_pos = b * so0 + oc * so1 + ow * so2

    # Записываем результат свёртки в output.
    out[out_pos] = acc


def tensor_conv1d_cuda(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    # Задаём количество потоков в одном блоке.
    threadsperblock = THREADS_PER_BLOCK

    # Вычисляем количество блоков так, чтобы потоков хватило на весь output.
    blockspergrid = (out_size + threadsperblock - 1) // threadsperblock

    # Запускаем CUDA kernel.
    # Синтаксис [blockspergrid, threadsperblock] задаёт конфигурацию GPU-запуска.
    _tensor_conv1d_cuda[blockspergrid, threadsperblock](
        out,
        out_shape,
        out_strides,
        out_size,
        input,
        input_shape,
        input_strides,
        weight,
        weight_shape,
        weight_strides,
        reverse,
    )


class CudaConv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        CUDA forward для 1D-свёртки.

        input: batch x in_channels x width
        weight: out_channels x in_channels x kernel_width
        output: batch x out_channels x width
        """

        # Сохраняем input и weight для backward.
        ctx.save_for_backward(input, weight)

        # Достаём размеры input.
        batch, in_channels, width = input.shape

        # Достаём размеры weight.
        out_channels, in_channels2, kw = weight.shape

        # Проверяем, что количество входных каналов совпадает.
        assert in_channels == in_channels2

        # Создаём output нужной формы.
        output = input.zeros((batch, out_channels, width))

        # Запускаем CUDA-реализацию forward-свёртки.
        tensor_conv1d_cuda(
            *output.tuple(),
            output.size,
            *input.tuple(),
            *weight.tuple(),
            False,
        )

        # Возвращаем результат forward.
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        # Достаём input и weight, сохранённые в forward.
        input, weight = ctx.saved_values

        # Достаём размеры input.
        batch, in_channels, width = input.shape

        # Достаём размеры weight.
        out_channels, in_channels2, kw = weight.shape

        # Создаём тензор для градиента по weight.
        # Сначала форма временная: in_channels x out_channels x kw.
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))

        # Меняем порядок размерностей input:
        # batch и in_channels меняются местами.
        new_input = input.permute(1, 0, 2)

        # Меняем порядок размерностей grad_output:
        # batch и out_channels меняются местами.
        new_grad_output = grad_output.permute(1, 0, 2)

        # Считаем градиент по весам через свёртку.
        tensor_conv1d_cuda(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )

        # Возвращаем grad_weight к форме:
        # out_channels x in_channels x kw.
        grad_weight = grad_weight.permute(1, 0, 2)

        # Создаём тензор для градиента по input.
        grad_input = input.zeros((batch, in_channels, width))

        # Меняем порядок weight:
        # out_channels и in_channels меняются местами.
        new_weight = weight.permute(1, 0, 2)

        # Считаем градиент по input.
        # Здесь reverse=True, потому что для backward по input ядро идёт в обратную сторону.
        tensor_conv1d_cuda(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )

        # Возвращаем градиенты по input и weight.
        return grad_input, grad_weight


# Публичная функция conv1d.
# Теперь conv1d(input, weight) вызывает CudaConv1dFun.apply.
conv1d = CudaConv1dFun.apply


@cuda.jit
def _tensor_conv2d_cuda(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    CUDA-реализация 2D-свёртки.

    input: batch x in_channels x height x width
    weight: out_channels x in_channels x kernel_height x kernel_width
    output: batch x out_channels x height x width
    """

    # Вычисляем глобальный номер CUDA-потока.
    ordinal = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # Если поток не соответствует реальному элементу output, выходим.
    if ordinal >= out_size:
        return

    # Достаём количество входных каналов.
    in_channels = input_shape[1]

    # Достаём высоту input.
    height = input_shape[2]

    # Достаём ширину input.
    width = input_shape[3]

    # Достаём количество выходных каналов.
    out_channels = weight_shape[0]

    # Достаём высоту kernel.
    kh = weight_shape[2]

    # Достаём ширину kernel.
    kw = weight_shape[3]

    # Достаём высоту output.
    out_height = out_shape[2]

    # Достаём ширину output.
    out_width = out_shape[3]

    # Strides input:
    # s10 — batch,
    # s11 — input channel,
    # s12 — height,
    # s13 — width.
    s10 = input_strides[0]
    s11 = input_strides[1]
    s12 = input_strides[2]
    s13 = input_strides[3]

    # Strides weight:
    # s20 — output channel,
    # s21 — input channel,
    # s22 — kernel height,
    # s23 — kernel width.
    s20 = weight_strides[0]
    s21 = weight_strides[1]
    s22 = weight_strides[2]
    s23 = weight_strides[3]

    # Strides output:
    # so0 — batch,
    # so1 — output channel,
    # so2 — height,
    # so3 — width.
    so0 = out_strides[0]
    so1 = out_strides[1]
    so2 = out_strides[2]
    so3 = out_strides[3]

    # Из ordinal восстанавливаем индекс batch.
    b = ordinal // (out_channels * out_height * out_width)

    # Убираем часть ordinal, отвечающую за batch.
    rem = ordinal - b * out_channels * out_height * out_width

    # Восстанавливаем индекс output channel.
    oc = rem // (out_height * out_width)

    # Убираем часть rem, отвечающую за output channel.
    rem = rem - oc * out_height * out_width

    # Восстанавливаем координату output по высоте.
    oh = rem // out_width

    # Восстанавливаем координату output по ширине.
    ow = rem - oh * out_width

    # Накопитель суммы свёртки.
    acc = 0.0

    # Перебираем входные каналы.
    for ic in range(in_channels):

        # Перебираем kernel по высоте.
        for kh_i in range(kh):

            # Перебираем kernel по ширине.
            for kw_i in range(kw):

                # Если reverse=True, ядро закреплено в правом нижнем углу.
                if reverse:
                    ih = oh - kh_i
                    iw = ow - kw_i

                # Если reverse=False, ядро закреплено в левом верхнем углу.
                else:
                    ih = oh + kh_i
                    iw = ow + kw_i

                # Проверяем, что координаты input внутри границ.
                if 0 <= ih < height and 0 <= iw < width:

                    # Позиция input-элемента в storage.
                    input_pos = b * s10 + ic * s11 + ih * s12 + iw * s13

                    # Позиция weight-элемента в storage.
                    weight_pos = oc * s20 + ic * s21 + kh_i * s22 + kw_i * s23

                    # Добавляем произведение input на weight.
                    acc += input[input_pos] * weight[weight_pos]

    # Позиция output-элемента в storage.
    out_pos = b * so0 + oc * so1 + oh * so2 + ow * so3

    # Записываем результат.
    out[out_pos] = acc


def tensor_conv2d_cuda(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    # Количество потоков в одном CUDA-блоке.
    threadsperblock = THREADS_PER_BLOCK

    # Количество блоков, чтобы покрыть все элементы output.
    blockspergrid = (out_size + threadsperblock - 1) // threadsperblock

    # Запуск CUDA kernel.
    _tensor_conv2d_cuda[blockspergrid, threadsperblock](
        out,
        out_shape,
        out_strides,
        out_size,
        input,
        input_shape,
        input_strides,
        weight,
        weight_shape,
        weight_strides,
        reverse,
    )


class CudaConv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        CUDA forward для 2D-свёртки.

        input: batch x in_channels x height x width
        weight: out_channels x in_channels x kernel_height x kernel_width
        output: batch x out_channels x height x width
        """

        # Сохраняем input и weight для backward.
        ctx.save_for_backward(input, weight)

        # Достаём форму input.
        batch, in_channels, height, width = input.shape

        # Достаём форму weight.
        out_channels, in_channels2, kh, kw = weight.shape

        # Проверяем совпадение числа входных каналов.
        assert in_channels == in_channels2

        # Создаём output нужной формы.
        output = input.zeros((batch, out_channels, height, width))

        # Запускаем CUDA-свёртку.
        tensor_conv2d_cuda(
            *output.tuple(),
            output.size,
            *input.tuple(),
            *weight.tuple(),
            False,
        )

        # Возвращаем output.
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        # Достаём input и weight из forward.
        input, weight = ctx.saved_values

        # Достаём форму input.
        batch, in_channels, height, width = input.shape

        # Достаём форму weight.
        out_channels, in_channels2, kh, kw = weight.shape

        # Создаём временный grad_weight.
        # Сначала форма: in_channels x out_channels x kh x kw.
        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))

        # Меняем input: batch и in_channels переставляются местами.
        new_input = input.permute(1, 0, 2, 3)

        # Меняем grad_output: batch и out_channels переставляются местами.
        new_grad_output = grad_output.permute(1, 0, 2, 3)

        # Считаем градиент по weight через CUDA-свёртку.
        tensor_conv2d_cuda(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )

        # Возвращаем grad_weight к форме:
        # out_channels x in_channels x kh x kw.
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        # Создаём grad_input нужной формы.
        grad_input = input.zeros((batch, in_channels, height, width))

        # Переставляем каналы weight:
        # out_channels и in_channels меняются местами.
        new_weight = weight.permute(1, 0, 2, 3)

        # Считаем градиент по input.
        # reverse=True нужен, потому что backward по input идёт с обратным закреплением ядра.
        tensor_conv2d_cuda(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )

        # Возвращаем градиенты по input и weight.
        return grad_input, grad_weight


# Теперь conv2d(input, weight) вызывает CudaConv2dFun.apply.
conv2d = CudaConv2dFun.apply