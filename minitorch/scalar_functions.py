from __future__ import annotations

from typing import TYPE_CHECKING
import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple
    from .scalar import Scalar, ScalarLike


# Гарантирует, что backward всегда возвращает кортеж, даже если функция имеет один аргумент.
def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    # Преобразовать возможное значение в кортеж
    if isinstance(x, tuple):
        return x
    return (x,)


# Если кортеж содержит один элемент — возвращает сам элемент.
def unwrap_tuple(x):  # type: ignore
    "Turn a singleton tuple into a value"
    # Преобразовать одноэлементный кортеж в значение
    if len(x) == 1:
        return x[0]
    return x


# Это базовый класс для всех операций над скалярами:
#
# Add
# Mul
# Neg
# Sigmoid
# Log
# ReLU
# Exp
# LT
# EQ
class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces Scalar variables.
    Обёртка для математической функции, которая принимает и производит переменные типа Scalar.
    """

    # Когда ты пишешь:
    #     z = x * y
    # MiniTorch на самом деле делает:
    #     z = Mul.apply(x, y)
    #
    # Forward — это просто вычисление.
    # Backward — это просто формула.
    #
    # А вот apply — это то, что:
    #   • создаёт контекст
    #   • вызывает forward
    #   • сохраняет историю
    #   • создаёт новый Scalar
    #   • связывает всё в граф
    #
    # В данном контексте Scalar и число float — разные вещи.
    # Scalar — это объект, который содержит:
    #   1) .data — обычное число (float)
    #   2) .history — как это число было получено
    #   3) .grad — градиент
    #   4) методы backward(), __add__, __mul__, и т.д.

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        raw_vals = []
        scalars = []

        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        ctx = Context(False)

        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), f"Expected return type float got {type(c)}"

        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    "Addition function $f(x, y) = x + y$"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return float(a + b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        return d_output, d_output


class Log(ScalarFunction):
    "Log function $f(x) = log(x)$"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return float(operators.log(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        ctx.save_for_backward(a, b)
        return float(a * b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return float(operators.inv(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        return float(operators.neg(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return -d_output


class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        out = operators.sigmoid(a)
        ctx.save_for_backward(out)
        return float(out)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (out,) = ctx.saved_values
        return d_output * out * (1 - out)


class ReLU(ScalarFunction):
    "ReLU function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return float(operators.relu(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    "Exp function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        out = operators.exp(a)
        ctx.save_for_backward(out)
        return float(out)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (out,) = ctx.saved_values
        return d_output * out


class LT(ScalarFunction):
    "Less-than function"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return float(operators.lt(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        return 0.0, 0.0


class EQ(ScalarFunction):
    "Equal function"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return float(operators.eq(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        return 0.0, 0.0
