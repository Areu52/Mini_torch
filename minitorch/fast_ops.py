from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)



class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"

        # This line JIT compiles your tensor_map
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        "See `tensor_ops.py`"

        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        "See `tensor_ops.py`"
        f = tensor_reduce(njit()(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
        """

        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    Создаёт быструю функцию map для тензоров.

    fn — обычная функция от одного числа к одному числу:
    neg, relu, sigmoid, log...
    """

    def _map(
        out: Storage,          # массив выходного тензора
        out_shape: Shape,      # shape выходного тензора
        out_strides: Strides,  # strides выходного тензора
        in_storage: Storage,   # массив входного тензора
        in_shape: Shape,       # shape входного тензора
        in_strides: Strides,   # strides входного тензора
    ) -> None:
        
        # Проверяем, совпадают ли размерности shape у out и input
        same_shape = len(out_shape) == len(in_shape)

        # Если количество измерений совпало, проверяем каждое измерение
        if same_shape:
            for i in range(len(out_shape)):
                # Если хотя бы одно измерение отличается
                # значит формы не полностью одинаковые
                if out_shape[i] != in_shape[i]:
                    same_shape = False
                    break

        # Проверяем, совпадает ли количество strides.
        same_strides = len(out_strides) == len(in_strides)

        # Если количество strides совпало, проверяем каждый stride
        if same_strides:
            for i in range(len(out_strides)):
                # Если хотя бы один stride отличается
                # значит расположeниe в памяти не одинаковоe
                if out_strides[i] != in_strides[i]:
                    same_strides = False
                    break

        # Быстрый случай:
        # shape и strides одинаковые.
        # Это значит, что i-й элемент out соответствует i-му элементу input.
        # Не нужно считать многомерные индексы.
        if same_shape and same_strides:
            # prange — параллельный range из Numba
            # Numba распределяeт итерации между потоками процeссора
            for i in prange(out.size):
                # Берём i-й элемент входа,
                # применяем fn,
                # кладём результат в i-й элемент выхода
                out[i] = fn(in_storage[i])

            # Всё сделали
            return

        # Общий случай:
        # shape/strides могут отличаться
        for i in prange(out.size):
            # Буфер для многомерного индекса выходного тензора
            # MAX_DIMS = максимальное число измерений
            out_index = np.empty(MAX_DIMS, dtype=np.int32)

            # Буфер для многомерного индекса входного тензора
            in_index = np.empty(MAX_DIMS, dtype=np.int32)

            # Превращаем порядковый номер i в многомерный индекс out
            # Например, i=5, shape=(2, 3) -> out_index=(1, 2)
            to_index(i, out_shape, out_index)

            # Переводим индекс out в индекс input broadcastingом
            # Например, out_shape=(3, 4), in_shape=(1, 4),
            # тогда индекс по первой оси во входе всегда будет 0
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # Переводим многомерный индекс out в позицию в одномерном storage
            out_pos = index_to_position(out_index, out_strides)

            # Переводим многомерный индекс input в позицию в одномерном storage
            in_pos = index_to_position(in_index, in_strides)

            # Применяем функцию к нужному элементу input
            # и записываем в нужную позицию output
            out[out_pos] = fn(in_storage[in_pos])

    # Компилируем _map через Numba.
    # parallel=True разрешает использовать prange
    return njit(parallel=True)(_map)  


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:

    def _zip(
        out: Storage,          # массив storage выходного тензора, куда записываем результат
        out_shape: Shape,      # shape выходного тензора
        out_strides: Strides,  # strides выходного тензора

        a_storage: Storage,    # массив storage первого входного тензора а, откуда читаем значения
        a_shape: Shape,        # shape первого тензора a
        a_strides: Strides,    # strides первого тензора a

        b_storage: Storage,    # массив storage второго входного тензора b
        b_shape: Shape,        # shape второго тензора b
        b_strides: Strides,    # strides второго тензора b
    ) -> None:

            # Проверяем совпадает ли shape у out и a
            same_a = len(out_shape) == len(a_shape)
            if same_a:
                for i in range(len(out_shape)):
                    if out_shape[i] != a_shape[i]:
                        same_a = False
                        break

            # Проверяем совпадает ли shape у out и b
            same_b = len(out_shape) == len(b_shape)
            if same_b:
                for i in range(len(out_shape)):
                    if out_shape[i] != b_shape[i]:
                        same_b = False
                        break

            # Проверяем совпадают ли strides у out и a
            same_a_strides = len(out_strides) == len(a_strides)
            if same_a_strides:
                for i in range(len(out_strides)):
                    if out_strides[i] != a_strides[i]:
                        same_a_strides = False
                        break

            # Проверяем совпадают ли strides у out и b
            same_b_strides = len(out_strides) == len(b_strides)
            if same_b_strides:
                for i in range(len(out_strides)):
                    if out_strides[i] != b_strides[i]:
                        same_b_strides = False
                        break

            # Быстрый случай
            # out a b имеют одинаковые shape и strides
            # значит i элемент out соответствует i элементу a и i элементу b
            if same_a and same_b and same_a_strides and same_b_strides:
                for i in prange(out.size):
                    out[i] = fn(a_storage[i], b_storage[i])
                return

            # Общий случай
            for i in prange(out.size):
                # Индекс выходного тензора
                out_index = np.empty(MAX_DIMS, dtype=np.int32)

                # Индекс первого входного тензора
                a_index = np.empty(MAX_DIMS, dtype=np.int32)

                # Индекс второго входного тензора
                b_index = np.empty(MAX_DIMS, dtype=np.int32)

                # Переводим порядковый номер i в многомерный индекс out
                to_index(i, out_shape, out_index)

                # Переводим индекс out в индекс a broadcastingом
                broadcast_index(out_index, out_shape, a_shape, a_index)

                # Переводим индекс out в индекс b broadcastingом
                broadcast_index(out_index, out_shape, b_shape, b_index)

                # Переводим многомерный индекс out в позицию в storage out
                out_pos = index_to_position(out_index, out_strides)

                # Переводим многомерный индекс a в позицию в storage a
                a_pos = index_to_position(a_index, a_strides)

                # Переводим многомерный индекс b в позицию в storage b
                b_pos = index_to_position(b_index, b_strides)

                # Берём нужные элементы из a и b
                # применяем fn
                # записываем результат в out
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return njit(parallel=True)(_zip)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:

    def _reduce(
        out: Storage,          # storage выходного тензора
        out_shape: Shape,      # shape выходного тензора
        out_strides: Strides,  # strides выходного тензора

        a_storage: Storage,    # storage входного тензора
        a_shape: Shape,        # shape входного тензора
        a_strides: Strides,    # strides входного тензора

        reduce_dim: int,       # ось по которой делаем reduce
    ) -> None:

        # Идём по каждому элементу out
        # Каждый out[i] считается отдельно
        for i in prange(out.size):

            # многомерный индекс out
            out_index = np.empty(MAX_DIMS, dtype=np.int32)

            # Переводим обычный номер i
            # в многомерный индекс out
            to_index(i, out_shape, out_index)

            # Находим позицию в storage для out
            out_pos = index_to_position(out_index, out_strides)

            # Начальная позиция во входном тензоре
            # Используем тот же индекс что и у out
            #
            # Пример
            # a_shape   = (3, 2)
            # out_shape = (3, 1)
            #
            # out_index = (1, 0)
            # значит начинаeм примeнять функцию с a[1, 0]
            a_pos = index_to_position(out_index, a_strides)

            # Берём стартовое значение
            # оно уже заранее записано в out
            #
            # Для sum это 0
            # Для prod это 1
            acc = out[out_pos]

            # Идём вдоль reduce_dim
            for j in range(a_shape[reduce_dim]):

                # Двигаемся по нужной оси
                #
                # например
                # a[1,0] -> a[1,1]
                acc = fn(
                    acc,
                    a_storage[a_pos + j * a_strides[reduce_dim]]
                )

            # Записываем итог
            out[out_pos] = acc

    return njit(parallel=True)(_reduce)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    
    """
    NUMBA-функция для быстрого умножения тензоров как матриц.

    Работает для любых форм тензоров с broadcasting,
    если выполняется условие:

        a_shape[-1] == b_shape[-2]

    То есть:
    число столбцов у a должно быть равно
    числу строк у b.

    Функция считает:

        out[n, i, j] =
        sum_k a[n, i, k] * b[n, k, j]

    где:
        n — batch
        i — строка
        j — столбец
        k — внутренняя размерность

    Оптимизации:

    * Внешний цикл выполняется параллельно
    * Не используются буферы индексов и лишние вызовы функций
    * Во внутреннем цикле нет записи в глобальную память,
      только одно умножение и накопление суммы

    Аргументы:
        out (Storage):
            storage выходного тензора `out`

        out_shape (Shape):
            shape выходного тензора `out`

        out_strides (Strides):
            strides выходного тензора `out`

        a_storage (Storage):
            storage входного тензора `a`

        a_shape (Shape):
            shape входного тензора `a`

        a_strides (Strides):
            strides входного тензора `a`

        b_storage (Storage):
            storage входного тензора `b`

        b_shape (Shape):
            shape входного тензора `b`

        b_strides (Strides):
            strides входного тензора `b`

    Возвращает:
        None

    Функция ничего не возвращает,
    а заполняет значения прямо в `out`.
    """

    # Если batch у a больше 1,
    # используем обычный stride по batch.
    #
    # Если batch == 1,
    # значит a broadcastится по batch,
    # поэтому stride должен быть 0
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0

    # То же самое для b
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # Размер batch
    # сколько матриц нужно перемножить
    batch = out_shape[0]

    # Количество строк результата
    out_rows = out_shape[1]

    # Количество столбцов результата
    out_cols = out_shape[2]

    # Внутренняя размерность:
    #
    # a.shape = (batch, rows, inner)
    # b.shape = (batch, inner, cols)
    #
    # Именно по этой оси идёт суммирование
    inner = a_shape[2]

    # Параллельно идём по каждому элементу результата
    #
    # Всего элементов:
    # batch * out_rows * out_cols
    for ordinal in prange(batch * out_rows * out_cols):

        # Находим номер матрицы - batch 
        #
        # Например:
        # ordinal = 17
        # batch_index = 1
        # out_rows = 3
        # out_cols = 4
        batch_index = ordinal // (out_rows * out_cols)

        # Остаток внутри одной batch-матрицы
        rest = ordinal - batch_index * out_rows * out_cols

        # Номер строки результата
        row = rest // out_cols

        # Номер столбца результата
        col = rest - row * out_cols

        # Позиция элемента
        # out[batch_index, row, col]
        # в одномерном storage
        out_pos = (
            batch_index * out_strides[0]
            + row * out_strides[1]
            + col * out_strides[2]
        )

        # Начальная позиция строки:
        # a[batch_index, row, 0]
        #
        # Если a broadcastится по batch,
        # то a_batch_stride = 0, вродe всe логично
        a_pos = (
            batch_index * a_batch_stride
            + row * a_strides[1]
        )

        # Начальная позиция столбца:
        # b[batch_index, 0, col]
        #
        # Если b broadcastится по batch,
        # то b_batch_stride = 0, вродe всe логично
        b_pos = (
            batch_index * b_batch_stride
            + col * b_strides[2]
        )

        # сюда засунeм сумму
        acc = 0.0

        # Считаем скалярное произведение:
        #
        # out[batch, row, col] =
        # сумма a[batch, row, k] * b[batch, k, col]
        for k in range(inner):
            acc += (
                a_storage[a_pos + k * a_strides[2]]
                *
                b_storage[b_pos + k * b_strides[1]]
            )

        out[out_pos] = acc


tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
