"""
===========================================================
   ОБЪЯСНЕНИЕ НИЗКОУРОВНЕВЫХ ОПЕРАЦИЙ ДЛЯ TENSOR_OPS.PY
===========================================================

В этом файле мы реализуем три фундаментальные операции:
    1. tensor_map
    2. tensor_zip
    3. tensor_reduce

Эти функции — основа всего тензорного движка MiniTorch.
На них строятся:
    - сложение, умножение, сравнения
    - сигмоида, ReLU, логарифм, экспонента
    - суммирование по оси
    - все операции autograd
    - визуализация графа в Streamlit

Они работают на СЫРОМ storage (обычный Python‑список или numpy‑массив),
и используют shape, strides и индексацию.

-----------------------------------------------------------
1. tensor_map(fn)
-----------------------------------------------------------
Назначение:
    Применить функцию fn(x) к КАЖДОМУ элементу входного тензора.

Пример:
    fn = neg
    in = [1, 2, 3]
    out = [-1, -2, -3]

Но MiniTorch должен поддерживать:
    - broadcasting
    - произвольные strides
    - view‑тензоры
    - транспонированные тензоры

Поэтому алгоритм такой:
    Для каждого элемента out:
        1. ordinal → многомерный индекс out
        2. broadcast_index → индекс входного тензора
        3. index_to_position → позиция в памяти
        4. out[pos] = fn(in[pos])

-----------------------------------------------------------
2. tensor_zip(fn)
-----------------------------------------------------------
Назначение:
    Применить функцию fn(a, b) к двум тензорам.

Пример:
    fn = add
    a = [1, 2, 3]
    b = [10, 20, 30]
    out = [11, 22, 33]

Но опять же:
    - формы могут быть разными
    - broadcasting обязателен
    - strides могут быть любыми

Алгоритм:
    Для каждого элемента out:
        1. ordinal → индекс out
        2. broadcast_index → индекс в a
        3. broadcast_index → индекс в b
        4. вычислить позиции в памяти
        5. out[pos] = fn(a[pos], b[pos])

-----------------------------------------------------------
3. tensor_reduce(fn)
-----------------------------------------------------------
Назначение:
    Выполнить свёртку по одной оси (reduce_dim).

Пример:
    a = [[1, 2, 3],
         [4, 5, 6]]

    reduce_dim = 1
    fn = add

    out = [[6],
           [15]]

Алгоритм:
    Для каждого элемента out:
        1. ordinal → индекс out
        2. копируем индекс в индекс входного тензора
        3. перебираем ВСЕ значения вдоль reduce_dim
        4. применяем fn(acc, value)
        5. записываем результат в out

Важно:
    out уже заполнен начальными значениями (start),
    поэтому reduce работает как fold/accumulate.

-----------------------------------------------------------
ИТОГ:
-----------------------------------------------------------
Эти три функции дают MiniTorch возможность выполнять
все тензорные операции, как в PyTorch, но на чистом Python.

Дальше поверх них строятся:
    - TensorOps.map / zip / reduce
    - TensorFunctions (Add, Mul, Sigmoid, ReLU, Sum, ...)
    - Tensor методы (__add__, sum, mean, relu, ...)
    - autograd
    - визуализация графа

===========================================================
"""


from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        pass

    @staticmethod
    def cmap(fn: Callable[[float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        pass

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """
        Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
            ops : tensor operations object see `tensor_ops.py`


        Returns :
            A collection of tensor functions

        """

        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.id_cmap = ops.cmap(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """
        Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
            new tensor data
        """

        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float]
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """
        Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
            :class:`TensorData` : new tensor data
        """

        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """
        Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to reduce over
            dim (int): int of dim to reduce

        Returns:
            :class:`TensorData` : new tensor
        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.

def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
        fn: function from float-to-float to apply

    Returns:
        Tensor map function.
    """
    # есть тензор a и надо получить новый тензор out, где каждый элемент — это результат применения функции fn к соответствующему элементу a.
    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        
        # Количество элементов в выходном тензоре
        size = int(np.prod(out_shape))
        
        # Временные массивы индексов
        out_index = np.zeros_like(out_shape)
        in_index = np.zeros_like(in_shape)

        for ordinal in range(size):
            to_index(ordinal, out_shape, out_index)

            # Применяем правила broadcast из предыдущего файла:
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # Вычисляем позиции в storage
            pos_out = index_to_position(out_index, out_strides)
            pos_in = index_to_position(in_index, in_strides)

            # Применяем функцию fn к каждому элементу входного тензора
            out[pos_out] = fn(float(in_storage[pos_in]))
    return _map



def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
        fn: function mapping two floats to float to apply

    Returns:
        Tensor zip function.
    """
    # out[i] = fn(a[i], b[i])
    def _zip(
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
   
        size = int(np.prod(out_shape))

        out_index = np.zeros_like(out_shape)
        a_index = np.zeros_like(a_shape)
        b_index = np.zeros_like(b_shape)

        for ordinal in range(size):
            # Индекс в out
            to_index(ordinal, out_shape, out_index)

            # Индекс в a с учётом broadcast
            broadcast_index(out_index, out_shape, a_shape, a_index)

            # Индекс в b с учётом broadcast
            broadcast_index(out_index, out_shape, b_shape, b_index)

            # 4. Позиции в массиве, на котором все реализовано
            pos_out = index_to_position(out_index, out_strides)
            pos_a = index_to_position(a_index, a_strides)
            pos_b = index_to_position(b_index, b_strides)

            # Применяем функцию как от нас хотели
            out[pos_out] = fn(float(a_storage[pos_a]), float(b_storage[pos_b]))

    return _zip



def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
        fn: reduction function mapping two floats to float

    Returns:
        Tensor reduce function.
    """

    """
    reduce_dim = 1
    a = [[1, 2, 3],
     [4, 5, 6]]

    sum по dim=1 → [[6],[15]]
    """
    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        size = int(np.prod(out_shape))
        out_index = np.zeros_like(out_shape)
        a_index = np.zeros_like(a_shape)

        for ordinal in range(size):
            # Индекс в out
            to_index(ordinal, out_shape, out_index)
            pos_out = index_to_position(out_index, out_strides)
            acc = float(out[pos_out])

            # Копируем индекс out → a_index
            a_index[:] = out_index

            for k in range(a_shape[reduce_dim]):
                a_index[reduce_dim] = k
                pos_a = index_to_position(a_index, a_strides)
                acc = fn(acc, float(a_storage[pos_a]))

            out[pos_out] = acc

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
