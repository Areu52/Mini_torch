from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.
    Вычисляет приближённое значение производной `f` по одному аргументу.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.
    См. :doc:`derivative` или https://en.wikipedia.org/wiki/Finite_difference для получения более подробной информации.

    Args:
        f : arbitrary function from n-scalar args to one value
        f : произвольная функция от n скалярных аргументов к одному значению

        *vals : n-float values $x_0 \ldots x_{n-1}$
        *vals : n значений с плавающей точкой $x_0 \ldots x_{n-1}$

        arg : the number $i$ of the arg to compute the derivative
        arg : номер $i$ аргумента, по которому вычисляется производная

        epsilon : a small constant
        epsilon : малая константа

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
        Приближённое значение $f'_i(x_0, \ldots, x_{n-1})$
    """
    # Создаём копии входных значений для +epsilon и -epsilon.
    vals_plus = list(vals) 
    vals_minus = list(vals)
    
    # Изменяем нужный аргумент, прибавляя/вычитая epsilon. 
    vals_plus[arg] += epsilon
    vals_minus[arg] -= epsilon

    
    # Что делает f(*vals_plus), если vals_plus = [1, 2, 3], то эта запись значит, что
    # вызывается f(1, 2, 3)
    
    # Вычисляем центральную разность: (f(x+eps) - f(x-eps)) / (2*eps). 
    return (f(*vals_plus) - f(*vals_minus)) / (2 * epsilon)

variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass

# нужно получить список узлов графа "вычислений", начиная от корневой и двигаясь к листьям
# (ну это поиск в грубину DFS)
# z = (x * y) + log(w). Порядок такой: [x, y, (x*y), w, log(w), z]
def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    Вычисляет топологический порядок графа вычислений.

    Аргументы:
        variable: самая правая переменная

    Возвращает:
        Неконстантные переменные в топологическом порядке, начиная с правой.
    """
    def dfs(v, visited, order):
        if v.unique_id in visited:# если в узле уже были - скип
            return
        visited.add(v.unique_id)  # иначе помечаем, что посетили

        # чиселки мы пропускам(у них нет как бы своих узлов) и они не влияют на градиенты
        if v.is_constant():
            return
        # мы так строили узел, что он содержит родителей, от которых он был вычислен(это на языке графа)
        # А на языке операций: z = x * y, z.parents = [x, y]
        # t = z + w, t.parents = [z, w]
        for parent in v.parents:
            dfs(parent, visited, order) # рекурсивно идем вглубь

        order.append(v)
        return order

    order = [] # итоговый список с порядком
    visited = set() # для запоминания того, что посетили

    return dfs(variable, visited, order)



def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    Запускает обратное распространение по графу вычислений для того,
    чтобы вычислить производные для листовых узлов.

    Аргументы:
        variable: самая правая переменная
        deriv   : её производная, которую мы хотим распространить назад к листьям.

    Ничего не возвращает. Должна записывать результаты в значения производных
    каждого листа через `accumulate_derivative`.
    """
    # 1. Получаем топологический порядок
    order = topological_sort(variable)

    # Словарь для хранения градиентов каждого узла
    grads = {variable.unique_id: deriv}

    # 3. Идём по узлам в обратном порядке (от результата к листьям)
    for v in reversed(order):
        grad_v = grads.get(v.unique_id, 0.0) # если нет градиента то выдай 0

        # Если это лист — просто накапливаем градиент
        if v.is_leaf():
            v.accumulate_derivative(grad_v)
            continue

        # Если не лист — распространяем градиент родителям
        # z = x * y >> dz/dz = 1 >> dz/dx = y dz/dy = x
        # grad_v = dz/dz = 1; chain_rule(1) → [(x, y), (y, x)]
        for parent, parent_grad in v.chain_rule(grad_v):
            # grad(parent) = (старый градиент от других путей) + (новый градиент от текущего пути)
            grads[parent.unique_id] = grads.get(parent.unique_id, 0.0) + parent_grad


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
