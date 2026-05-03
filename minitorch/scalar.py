from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """
    `ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.
    `ScalarHistory` хранит историю операций `Function`,
    которые были использованы для построения текущей переменной.
    Attributes:
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.
    Атрибуты:
    last_fn : Последняя функция, которая была вызвана.
    ctx : Контекст для этой функции.
    inputs : Входы, которые были переданы при вызове `last_fn.forward`.
    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()

_var_count = 0


class Scalar:
    """
    A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.

    Переосмысление (повторная реализация) скалярных значений для
    отслеживания автоматического дифференцирования. Скалярные переменные
    ведут себя максимально похоже на обычные числа Python, одновременно
    отслеживая операции, которые привели к созданию этого числа.
    Они могут быть изменены только с помощью `ScalarFunction`.
    """


    history: Optional[ScalarHistory]
    derivative: Optional[float]
    data: float
    unique_id: int
    name: str

    def __init__(
        self,
        v: float,
        back: ScalarHistory = ScalarHistory(),
        name: Optional[str] = None,
    ):
        global _var_count
        _var_count += 1
        self.unique_id = _var_count
        self.data = float(v)
        self.history = back
        self.derivative = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

    def __repr__(self) -> str:
        return "Scalar(%f)" % self.data

    def __mul__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b: ScalarLike) -> Scalar:
        # Сложение: вызывает функцию Add и строит узел графа
        return Add.apply(self, b)

    def __lt__(self, b: ScalarLike) -> Scalar:
        # Операция "меньше": возвращает Scalar(0.0 или 1.0)
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        # Операция "больше": реализована через LT (b < self)
        return LT.apply(b, self)

    def __eq__(self, b: ScalarLike) -> Scalar:
        # Проверка равенства: возвращает Scalar(0.0 или 1.0)
        return EQ.apply(self, b)

    def __sub__(self, b: ScalarLike) -> Scalar:
        # x - y = x + (-y)
        return Add.apply(self, Neg.apply(b))

    def __neg__(self) -> Scalar:
        # -x
        return Neg.apply(self)


    def __radd__(self, b: ScalarLike) -> Scalar:
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        return self * b

    def log(self) -> Scalar:
        # Логарифм: log(x)
        return Log.apply(self)

    def exp(self) -> Scalar:
        # Экспонента: e^x
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        # Сигмоида: 1 / (1 + exp(-x))
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        # ReLU: max(0, x)
        return ReLU.apply(self)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """
        Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
            x: value to be accumulated

        Добавляет `val` к производной, накопленной на этой переменной.
        Должно вызываться только во время автоматического дифференцирования
        на листовых переменных.

        Аргументы:
            x: значение, которое нужно добавить
        """

        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.derivative = 0.0
        self.derivative += x

    def is_leaf(self) -> bool:
        "True if this variable created by the user (no `last_fn`)"
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """
        узнать, какая операция его создала
        вызвать backward этой операции
        получить градиенты по каждому входу
        вернуть пары:
        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        # Вызываем backward-функцию операции, которая создала этот Scalar.
        # Она возвращает градиенты по каждому входу.
        grads = h.last_fn._backward(h.ctx, d_output)

        # Возвращаем пары (родитель, его градиент).
        # Это нужно для backpropagate(), чтобы оно знало,
        # куда передавать градиенты дальше.
        return [(inp, grad) for inp, grad in zip(h.inputs, grads)]

    def backward(self, d_output: Optional[float] = None) -> None:
        """
        Calls autodiff to fill in the derivatives for the history of this object.

        Args:
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).
        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """
    Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Parameters:
        f : function from n-scalars to 1-scalar.
        *scalars  : n input scalar values.
    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
