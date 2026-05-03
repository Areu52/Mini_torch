from typing import Callable, Optional

import numba
from numba import cuda

from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        f = tensor_map(cuda.jit(device=True)(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        f = tensor_zip(cuda.jit(device=True)(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        f = tensor_reduce(cuda.jit(device=True)(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
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

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement

# Теперь реализуем многопоточный map чeрeз cuda
def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
       
    """
    CUDA higher-order tensor map function. ::

    fn_map = tensor_map(fn)
    fn_map(out, ... )

    Аргументы:
        fn: функция, которая применяется к каждому float-значению

    Возвращает:
        Tensor map function
    """

    def _map(
        out: Storage,          # storage выходного тензора
        out_shape: Shape,      # shape выходного тензора
        out_strides: Strides,  # strides выходного тензора
        out_size: int,         # количество элементов в выходном тензоре
        in_storage: Storage,   # storage входного тензора
        in_shape: Shape,       # shape входного тензора
        in_strides: Strides,   # strides входного тензора
    ) -> None:

        # Локальный массив для многомерного индекса out
        out_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Локальный массив для многомерного индекса input
        in_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Глобальный номер CUDA-потока
        # Каждый поток отвечает за один элемент out
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Проверяем, что поток не вышел за размер out
        if i < out_size:

            # Переводим номер i в многомерный индекс out
            to_index(i, out_shape, out_index)

            # Переводим индекс out в индекс input с учётом broadcasting
            broadcast_index(out_index, out_shape, in_shape, in_index)

            # Находим позицию элемента out в storage
            out_pos = index_to_position(out_index, out_strides)

            # Находим позицию элемента input в storage
            in_pos = index_to_position(in_index, in_strides)

            # Применяем fn к input и записываем результат в out
            out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Аргументы:
        fn: функция, которая принимает два float-значения
            и возвращает одно float-значение

    Возвращает:
        Tensor zip function
    """

    def _zip(
        out: Storage,          # storage выходного тензора
        out_shape: Shape,      # shape выходного тензора
        out_strides: Strides,  # strides выходного тензора
        out_size: int,         # количество элементов в выходном тензоре

        a_storage: Storage,    # storage первого входного тензора
        a_shape: Shape,        # shape первого входного тензора
        a_strides: Strides,    # strides первого входного тензора

        b_storage: Storage,    # storage второго входного тензора
        b_shape: Shape,        # shape второго входного тензора
        b_strides: Strides,    # strides второго входного тензора
    ) -> None:

        # Локальный массив для многомерного индекса out
        out_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Локальный массив для многомерного индекса a
        a_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Локальный массив для многомерного индекса b
        b_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Глобальный номер CUDA-потока
        # Каждый поток отвечает за один элемент out
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # Проверяем, что поток не вышел за размер out
        if i < out_size:

            # Переводим номер i в многомерный индекс out
            to_index(i, out_shape, out_index)

            # Переводим индекс out в индекс a с учётом broadcasting
            broadcast_index(out_index, out_shape, a_shape, a_index)

            # Переводим индекс out в индекс b с учётом broadcasting
            broadcast_index(out_index, out_shape, b_shape, b_index)

            # Находим позицию элемента out в storage
            out_pos = index_to_position(out_index, out_strides)

            # Находим позицию элемента a в storage
            a_pos = index_to_position(a_index, a_strides)

            # Находим позицию элемента b в storage
            b_pos = index_to_position(b_index, b_strides)

            # Применяем fn к элементам a и b
            # записываем результат в out
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """
    This is a practice sum kernel to prepare for reduce.

    Дан массив длины n и out размера n // blockDim.
    Нужно просуммировать каждые blockDim элементов
    и записать сумму в одну ячейку out.

    Пример:

        [a1, a2, ..., a100]

    превращается в:

        [a1 + ... + a31, a32 + ... + a64, ...]

    Важно:
        каждый block должен делать сумму через shared memory

    Аргументы:
        out (Storage):
            storage выходного тензора

        a (Storage):
            storage входного тензора

        size (int):
            длина массива a
    """

    BLOCK_DIM = 32

    # Shared memory внутри одного CUDA block
    # Все потоки этого block могут читать и писать сюда
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)

    # Глобальный индекс элемента a
    # Каждый CUDA-поток отвечает за один элемент
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # Локальный индекс потока внутри block
    # От 0 до BLOCK_DIM - 1
    pos = cuda.threadIdx.x

    # Если глобальный индекс не вышел за границы массива,
    # кладём значение a[i] в shared memory
    if i < size:
        cache[pos] = a[i]

    # Если поток вышел за границы массива,
    # кладём 0 чтобы он не влиял на сумму
    else:
        cache[pos] = 0.0

    # Ждём пока все потоки block запишут данные в cache
    cuda.syncthreads()

    # Начинаем параллельное суммирование внутри cache
    # Сначала складываем пары на расстоянии 16,
    # потом 8, потом 4, потом 2, потом 1
    step = BLOCK_DIM // 2

    while step > 0:
        # Только первые step потоков делают сложение
        if pos < step:
            cache[pos] += cache[pos + step]

        # Ждём пока все сложения этого шага закончатся
        cuda.syncthreads()

        # Уменьшаем расстояние в 2 раза
        step = step // 2

    # После reduce внутри block итоговая сумма лежит в cache[0]
    # Только первый поток block записывает её в out
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    CUDA higher-order tensor reduce function.

    Аргументы:
        fn: функция reduce, которая принимает два float-значения
            и возвращает одно float-значение

    Возвращает:
        Tensor reduce function
    """

    def _reduce(
        out: Storage,              # storage выходного тензора
        out_shape: Shape,          # shape выходного тензора
        out_strides: Strides,      # strides выходного тензора
        out_size: int,             # количество элементов в out

        a_storage: Storage,        # storage входного тензора
        a_shape: Shape,            # shape входного тензора
        a_strides: Strides,        # strides входного тензора

        reduce_dim: int,           # ось по которой делаем reduce
        reduce_value: float,       # стартовое значение reduce, например 0 для sum
    ) -> None:
        BLOCK_DIM = 1024

        # Shared memory для reduce внутри одного CUDA block
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)

        # Многомерный индекс выходного элемента
        out_index = cuda.local.array(MAX_DIMS, numba.int32)

        # Каждый block считает один элемент out
        out_pos = cuda.blockIdx.x

        # Номер потока внутри block
        pos = cuda.threadIdx.x

        # Проверяем что block не вышел за размер out
        if out_pos < out_size:

            # Переводим номер out_pos в многомерный индекс out
            to_index(out_pos, out_shape, out_index)

            # Размер оси которую сворачиваем
            reduce_size = a_shape[reduce_dim]

            # Stride по оси которую сворачиваем
            reduce_stride = a_strides[reduce_dim]

            # Начальная позиция во входном storage
            #
            # out_index совпадает с индексом a,
            # только по reduce_dim в out стоит номер чанка reduce
            base_pos = index_to_position(out_index, a_strides)

            # Каждый block считает один chunk длиной BLOCK_DIM
            #
            # Например:
            # block 0 считает элементы 0..1023
            # block 1 считает элементы 1024..2047
            start = out_index[reduce_dim] * BLOCK_DIM

            # Конкретный элемент reduce-оси для этого потока
            offset = start + pos

            # Если offset не вышел за reduce_size,
            # кладём значение из a в shared memory
            if offset < reduce_size:
                cache[pos] = a_storage[base_pos + offset * reduce_stride]

            # Если вышли за границу,
            # кладём стартовое значение reduce
            else:
                cache[pos] = reduce_value

            # Ждём пока все потоки запишут значения в cache
            cuda.syncthreads()

            # Параллельно сворачиваем cache
            step = BLOCK_DIM // 2

            while step > 0:
                if pos < step:
                    cache[pos] = fn(cache[pos], cache[pos + step])

                cuda.syncthreads()
                step = step // 2

            # Первый поток записывает итог reduce в out
            if pos == 0:
                out[out_pos] = cache[0]

    return cuda.jit()(_reduce)  # type: ignore

# shared_memory нужна здeсь так как один и тот жe элeмeнт матрицы нужeн многим потокам. Бeз нee один блок потоков читал бы каждый раз a[i,j], а с нeй будeт каждый блок быстро читать этот элeмeнт и с ним работать
def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """
    Тренировочное CUDA-ядро для умножения квадратных матриц,
    чтобы подготовиться к matmul.

    Даны storage `out`, `a`, `b`.

    Мы знаем, что `a` и `b` имеют shape:

        [size, size]

    и strides:

        [size, 1]

    Размер size всегда меньше 32.

    Требования:

    * Все данные сначала нужно перенести в shared memory(shared memory — это быстрая общая память внутри одного CUDA block.)
    * Каждую ячейку `a` и `b` нужно прочитать только один раз
    * В global memory нужно записать только один раз на kernel

    Считаем:

        for i:
            for j:
                for k:
                    out[i, j] += a[i, k] * b[k, j]

    Аргументы:
        out (Storage): storage для тензора `out`
        a (Storage): storage для тензора `a`
        b (Storage): storage для тензора `b`
        size (int): размер квадратной матрицы
    """
    BLOCK_DIM = 32

    # Shared memory для матрицы a
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Shared memory для матрицы b
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Локальная строка внутри block
    i = cuda.threadIdx.x

    # Локальный столбец внутри block
    j = cuda.threadIdx.y

    # Если индекс внутри реального размера матрицы,
    # переносим элементы из global memory в shared memory
    if i < size and j < size:
        a_shared[i, j] = a[i * size + j]
        b_shared[i, j] = b[i * size + j]

    # Если поток вне реального размера,
    # кладём 0 чтобы он не влиял на вычисления
    else:
        a_shared[i, j] = 0.0
        b_shared[i, j] = 0.0

    # Ждём пока все потоки загрузят данные в shared memory
    cuda.syncthreads()

    # Локальный аккумулятор для out[i, j]
    acc = 0.0

    # Считаем скалярное произведение:
    # строка i из a умножается на столбец j из b
    if i < size and j < size:
        for k in range(size):
            acc += a_shared[i, k] * b_shared[k, j]

        # Записываем результат один раз в global memory
        out[i * size + j] = acc


jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    
    """
    CUDA-функция умножения тензоров как матриц.

    Требования:

    * Все данные сначала должны быть перенесены в shared memory.
    * Каждая ячейка `a` и `b` должна быть прочитана только один раз.
    * В global memory нужно записывать только один раз на kernel.

    Должна работать для любых форм тензоров,
    которые можно broadcast-ить, если выполняется условие:

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```

    Возвращает:
        None: заполняет `out`.
    """
    # Если batch у a больше 1, используем обычный stride по batch.
    # Если batch == 1, значит a broadcastится по batch,
    # поэтому stride по batch должен быть 0.
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0

    # То же самое для b.
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # Номер batch фиксирован для этого CUDA block.
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Финальная позиция элемента результата out[i, j].
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

     # Локальная позиция потока внутри CUDA block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y


    # План кода:
    #
    # 1) Идём по общей внутренней размерности кусками размера BLOCK_DIM.
    #    a) Копируем кусок матрицы a в shared memory.
    #
    #    b) Копируем кусок матрицы b в shared memory.
    #
    #    c) Считаем скалярное произведение для позиции c[i, j].

    # Количество строк в результате
    out_rows = out_shape[1]

    # Количество столбцов в результате
    out_cols = out_shape[2]

    # Внутренняя размерность:
    # a.shape = (batch, rows, inner)
    # b.shape = (batch, inner, cols)
    inner = a_shape[2]

    # Локальная сумма для одного элемента out[batch, i, j]
    acc = 0.0

    # Идём по внутренней размерности кусками по BLOCK_DIM
    #
    # Например inner = 100
    # BLOCK_DIM = 32
    #
    # tile_start будет:
    # 0, 32, 64, 96
    for tile_start in range(0, inner, BLOCK_DIM):

        # Индекс k для загрузки элемента из a
        #
        # a[batch, i, tile_start + pj]
        a_k = tile_start + pj

        # Индекс k для загрузки элемента из b
        #
        # b[batch, tile_start + pi, j]
        b_k = tile_start + pi

        # Загружаем кусок матрицы a в shared memory.
        #
        # Каждый поток загружает один элемент:
        # a_shared[pi, pj] = a[batch, i, a_k]
        #
        # Если вышли за границы, кладём 0
        if i < out_rows and a_k < inner:
            a_shared[pi, pj] = a_storage[
                batch * a_batch_stride
                + i * a_strides[1]
                + a_k * a_strides[2]
            ]
        else:
            a_shared[pi, pj] = 0.0

        # Загружаем кусок матрицы b в shared memory.
        #
        # Каждый поток загружает один элемент:
        # b_shared[pi, pj] = b[batch, b_k, j]
        #
        # Если вышли за границы, кладём 0
        if b_k < inner and j < out_cols:
            b_shared[pi, pj] = b_storage[
                batch * b_batch_stride
                + b_k * b_strides[1]
                + j * b_strides[2]
            ]
        else:
            b_shared[pi, pj] = 0.0

        # Ждём, пока все потоки загрузят данные в shared memory
        cuda.syncthreads()

        # Теперь считаем часть скалярного произведения
        #
        # out[batch, i, j] +=
        # a[batch, i, k] * b[batch, k, j]
        #
        # Но читаем уже из shared memory
        for k in range(BLOCK_DIM):
            acc += a_shared[pi, k] * b_shared[k, pj]

        # Ждём, пока все потоки закончат использовать shared memory,
        # прежде чем перезаписать её на следующем tile
        cuda.syncthreads()

    # Если поток соответствует реальному элементу out,
    # записываем итог в global memory
    if i < out_rows and j < out_cols:
        out_pos = (
            batch * out_strides[0]
            + i * out_strides[1]
            + j * out_strides[2]
        )

        out[out_pos] = acc


tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)