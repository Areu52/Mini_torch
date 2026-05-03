from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple


class Module:
    """
    Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes:
        _modules : Storage of the child modules
        _parameters : Storage of the module's parameters
        training : Whether the module is in training mode or evaluation mode

    """

    """
    Модули образуют дерево, которое хранит параметры и другие
    подмодули. Они составляют основу стеков нейронных сетей.

    Атрибуты:
        _modules : Хранилище дочерних модулей
        _parameters : Хранилище параметров этого модуля
        training : Показывает, находится ли модуль в режиме обучения
                   или в режиме оценки
    """


    _modules: Dict[str, Module]
    _parameters: Dict[str, Parameter]
    training: bool

    def __init__(self) -> None:
        self._modules = {} # Создаёт пустой словарь, куда будут автоматически попадать все дочерние модули.
        self._parameters = {} # Создаёт словарь для хранения параметров
        self.training = True # Устанавливает модуль в режим обучения по умолчанию.

    # Метод ничего не принимает, кроме self, и должен вернуть последовательность объектов Module
    def modules(self) -> Sequence[Module]:
        "Return the direct child modules of this module."
        #self.__dict__ — это внутренний словарь всех атрибутов объекта.
       #В нём хранится ключ "_modules", который был создан в __init__.
        m: Dict[str, Module] = self.__dict__["_modules"]
        return list(m.values())

    def train(self) -> None:
        "Set the mode of this module and all descendent modules to `train`."
        # «Перевести этот модуль и все его потомки (вложенные модули) в режим обучения (train).
        self.training = True
        for module in self._modules.values():
            module.train()

    def eval(self) -> None:
        "Set the mode of this module and all descendent modules to `eval`."
        # «Перевести этот модуль и все его потомки (вложенные модули) в режим оценки (`eval`).»
        self.training = False # выходим из режима обучения
        for module in self._modules.values():
            module.eval()

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """
        Collect all the parameters of this module and its descendents.


        Returns:
            The name and `Parameter` of each ancestor parameter.
        """
        # «Собрать все параметры этого модуля и всех его потомков (вложенных модулей).
        # Вернуть имя и объект `Parameter` для каждого параметра

        params = []
        # Параметры текущего модуля
        for name, param in self._parameters.items():
            params.append((name, param))

        # Параметры дочерних модулей
        for module_name, module in self._modules.items():
            for child_name, child_param in module.named_parameters():
                # Добавляем название потомка: module_name.child_name
                full_name = f"{module_name}.{child_name}"
                params.append((full_name, child_param))
        return params

    def parameters(self) -> Sequence[Parameter]:
        "Enumerate over all the parameters of this module and its descendents."
        # «Перечислить все параметры этого модуля и всех его потомков (вложенных модулей).»

        return [param for (name, param) in self.named_parameters()]

    def add_parameter(self, k: str, v: Any) -> Parameter:
        """
        Manually add a parameter. Useful helper for scalar parameters.

        Args:
            k: Local name of the parameter.
            v: Value for the parameter.

        Returns:
            Newly created parameter.
        """
        """
        Вручную добавить параметр. Полезный вспомогательный метод для скалярных параметров.
        Аргументы:
        k: Локальное имя параметра.
        v: Значение для параметра.

        Возвращает:
        Вновь созданный параметр.
        """

        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    '''
    def __setattr__(self, key: str, val: Any) -> None:
        if isinstance(val, Parameter):
            self._parameters[key] = val
        elif isinstance(val, Module):
            self._modules[key] = val
        super().__setattr__(key, val)
    '''
    def __setattr__(self, key: str, val: Parameter) -> None:
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        def _addindent(s_: str, numSpaces: int) -> str:
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]
            s = "\n".join(s2)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """
    A Parameter is a special container stored in a `Module`.

    It is designed to hold a `Variable`, but we allow it to hold
    any value for testing.
    """
    """
    Параметр — это специальный контейнер, который хранится внутри `Module`.
    Он предназначен для хранения `Variable`, но для целей тестирования мы допускаем,
    что он может содержать любое значение.
    """


    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        self.value = x
        self.name = name
        # Если есть метод requires_grad_, значит x - тензор
        if hasattr(x, "requires_grad_"):
            # Если значение — тензор, включается вычисление градиентов.
            self.value.requires_grad_(True)
            # Если параметру дали имя, оно передаётся и самому тензору.
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        "Update the parameter value."
        # Обновить значение параметра.
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    # Вывод технический тензора
    # >>> repr(tensor)
    # 'Tensor([1, 2, 3], requires_grad=True)'
    def __repr__(self) -> str:
        return repr(self.value)

    # для вывода значения в строковом типе
    # __str__(tensor)
    # [1, 2, 3]
    def __str__(self) -> str:
        return str(self.value)
