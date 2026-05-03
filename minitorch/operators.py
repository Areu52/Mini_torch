import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    # TODO: Implement for Task 0.1.
    return x * y


def id(x: float) -> float:
    "$f(x) = x$"
    # TODO: Implement for Task 0.1.
    return x


def add(x: float, y: float) -> float:
    "$f(x, y) = x + y$"
    # TODO: Implement for Task 0.1.
    return x + y


def neg(x: float) -> float:
    "$f(x) = -x$"
    # TODO: Implement for Task 0.1.
    return -x


def lt(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is less than y else 0.0"
    # TODO: Implement for Task 0.1.
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is equal to y else 0.0"
    # TODO: Implement for Task 0.1.
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    "$f(x) =$ x if x is greater than y else y"
    # TODO: Implement for Task 0.1.
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    "$f(x) = |x - y| < 1e-2$"
    # TODO: Implement for Task 0.1.
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.


    сигмоида:
    если x >= 0: 1 / (1 + e^{-x})
    иначе: e^x / (1 + e^x)
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    # TODO: Implement for Task 0.1.
    return x if x > 0 else 0


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"If $f = log$ as above, compute $d \times f'(x)$"
    # Типа рассчет градиента d = dL / df, dL/dx = d * df / dx, f = log(x + eps)
    return d * (1.0 / (x + EPS))


def inv(x: float) -> float:
    "$f(x) = 1/x$"
    # TODO: Implement for Task 0.1.
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    r"If $f(x) = 1/x$ compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    return d * (-1.0 / (x * x))


def relu_back(x: float, d: float) -> float:
    r"If $f = relu$ compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
         A function that takes a list, applies `fn` to each element, and returns a
         new list
    """
    # Нужно создать функцию func, которая применяет к
    # каждому элементу списка функцию fn и возвращает
    # новый список fn(x)
    def func(a):
        return [fn(x) for x in a]
    return func


def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    # Тут надо применяя функции map и neg вернуть по начальному списку ls новый список,
    # где к каждому списку применилась ф-я neg
    neg_fn = map(neg)
    return neg_fn(ls) # map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
         Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """
    # Надо принять функцию fn и вернуть новую функцию func,
    # которая принимает 2 списка ls1 ls2 одинаковой длины
    # и применяет fn к каждой паре элементов (x, y), возвращая новый список

    def func(ls1, ls2):
        if len(ls1) != len(ls2):
            raise ValueError("Длины списков разные")

        return [fn(ls1[i], ls2[i]) for i in range(len(ls1))]
    return func

ls1 = [2, 3, 4, 5]
ls2 = [-1, 4, 6, 9]

def pair(x, y):
    return (x, y)

add_lists = zipWith(pair)
print(*zip(ls1, ls2), '|||',  *add_lists(ls1, ls2))


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    # Тут нужно сложить два списка ls1, ls2 поэлементно с помонью zipWidth и add
    zipwidth_add = zipWith(add)
    return zipwidth_add(ls1, ls2)



def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
         Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """
    # На входе аргументы: функция fn(a, b), которая как-то объединяет a и b
    # И начальное значение start
    # Нужно внутри создать функцию func, которая примет на вход список ls = [x1, x2, ..., xn]
    # и вычислит fn(xn, fn(xn-1, fn(xn-2...fn(x1, start)))), start типа х0

    def func(ls1):
        arg_2 = start
        for x in ls1:
            arg_2 = fn(x, arg_2)
        return arg_2
    return func


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    # Мне нужно взять список чисел ls, пройтись по нему, складывая все элементы,
    # используя свою функцию add и вернуть итоговое значение
    func = reduce(add, 0.0)
    return func(ls)


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    # Мне нужно взять список чисел ls, пройтись по нему, перемножая все элементы,
    # используя свою функцию mul и вернуть итоговое значение
    func = reduce(mul, 1.0)
    return func(ls)
