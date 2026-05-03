from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]

# есть многомерный индекс, например (i, j, k), и strides, например (12, 4, 1). нужно pos = i * 12 + j * 4 + k * 1
def index_to_position(index: Index, strides: Strides) -> int:
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in storage based on strides.
    Преобразует многомерный тензор `index` в одномерную позицию в хранилище на основе шагов.

    Args:
    Аргументы:

    index : index tuple of ints
    index : кортеж индексов из целых чисел

    strides : tensor strides
    strides : шаги тензора

    Returns:
    Возвращает:

    Position in storage
    Позицию в хранилище
    """

    pos = 0
    for i, s in zip(index, strides):
        pos += int(i) * int(s)
    return pos


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert an `ordinal` to an index in the `shape`.
    Преобразует `ordinal` в индекс внутри `shape`.

    Should ensure that enumerating position 0 ... size of a tensor produces every index exactly once.
    Должно гарантировать, что перечисление позиций от 0 до размера тензора порождает каждый индекс ровно один раз.

    It may not be the inverse of `index_to_position`.
    Это может не быть обратной операцией к `index_to_position`.

    Args:
    Аргументы:

        ordinal: ordinal position to convert.
        ordinal: порядковая позиция, которую нужно преобразовать.

        shape : tensor shape.
        shape : форма (shape) тензора.

        out_index : return index corresponding to position.
        out_index : возвращаемый индекс, соответствующий позиции.
        
    Пример:
    ordinal = 5
    shape = (2, 3)
    5 % 3 = 2 → последняя координата = 2
    5 // 3 = 1
    1 % 2 = 1 → первая координата = 1
    out_index = (1, 2)
    """
    # От конца к началу
    for i in range(len(shape)):
        cur = ordinal
        for j in range(i + 1, len(shape)):
            cur = cur // shape[j]
        out_index[i] = cur % shape[i]



def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index` into `shape` following broadcasting rules.
    Преобразует `big_index` в `big_shape` в меньший `out_index` в `shape` согласно правилам broadcasting.

    In this case it may be larger or with more dimensions than the `shape` given.
    В этом случае он может быть больше или иметь больше измерений, чем заданный `shape`.

    Additional dimensions may need to be mapped to 0 or removed.
    Дополнительные измерения могут быть сопоставлены с 0 или удалены.

    Args:
    Аргументы:

        big_index : multidimensional index of bigger tensor
        big_index : многомерный индекс большего тензора

        big_shape : tensor shape of bigger tensor
        big_shape : форма (shape) большего тензора

        shape : tensor shape of smaller tensor
        shape : форма (shape) меньшего тензора

        out_index : multidimensional index of smaller tensor
        out_index : многомерный индекс меньшего тензора

    Returns:
    Возвращает: None

    Пример:
    big_shape = (3, 4, 5)
    shape     = (4, 1)
    big_index = (2, 1, 3)
    

    """

    big_dims = len(big_shape)
    small_dims = len(shape)


    i = big_dims - 1
    j = small_dims - 1
    # Если у big_shape больше измерений, чем у shape,
    # то эти измерения просто игнорируются (broadcasting)

     # Пока есть измерения в маленькой shape
    while j >= 0:
        if shape[j] == 1:
            # Если размер = 1, то broadcasting индекс 0 по условию
            out_index[j] = 0
        else:
            out_index[j] = big_index[i]
        i -= 1
        j -= 1


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to create a new union shape.

    Args:
        shape1 : first shape
        shape2 : second shape

    Returns:
        broadcasted shape

    Raises:
        IndexingError : if cannot broadcast
    """
    len1, len2 = len(shape1), len(shape2)
    max_len = max(len1, len2)

    # Делаем одинаковой длины + добавляя 1 слева
    s1 = (1,) * (max_len - len1) + tuple(shape1)
    s2 = (1,) * (max_len - len2) + tuple(shape2)

    out = []
    for a, b in zip(s1, s2):
        if a == b:
            out.append(a)
        elif a == 1:
            out.append(b)
        elif b == 1:
            out.append(a)
        else:
            raise IndexingError(f"Невозможно выполнить broadcast для размеров {shape1} и {shape2}")
    
    return tuple(out)


def strides_from_shape(shape: UserShape) -> UserStrides:
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    # Меняет порядок осей тензора
    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            *order: a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.
        Пример:
        shape  = (2, 3)
        strides = (3, 1)

        a00 a01 a02
        a10 a11 a12

        Итог:
        a00 a10
        a01 a11
        a02 a12

        """

        # self._storage - сам одномерный массив, на котором реализован тензор
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        new_shape = tuple(self.shape[i] for i in order)
        new_strides = tuple(self.strides[i] for i in order)
        return TensorData(self._storage, new_shape, new_strides)

    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
