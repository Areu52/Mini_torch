import torch

import minitorch


def default_log_fn(epoch, total_loss, correct, losses):
    '''
        Функция логирования, вызываемая каждые несколько эпох.
        Выводит номер эпохи, текущее значение функции потерь и количество
        правильно классифицированных примеров.
    '''
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class Network(torch.nn.Module):
    '''
        Нейронная сеть из трёх полносвязных слоёв:
        - вход размерности 2 (точка x1, x2)
        - два скрытых слоя с ReLU
        - выходной слой с сигмоидой (вероятность класса)

        Это эталонная PyTorch‑модель, на которую нужно ориентироваться
        при реализации MiniTorch‑версии.
    '''
    def __init__(self, hidden_layers):
        super().__init__()

        # Submodules
        # Подмодули — линейные слои
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        '''
            Прямой проход сети:
            1) Линейный слой → ReLU
            2) Линейный слой → ReLU
            3) Линейный слой → Sigmoid (вероятность класса)
        '''
        h = self.layer1.forward(x).relu()
        h = self.layer2.forward(h).relu()
        return self.layer3.forward(h).sigmoid()


class Linear(torch.nn.Module):
    '''
        Реализация линейного слоя: y = xW + b.
        Параметры W и b — обучаемые тензоры PyTorch.
    '''
    def __init__(self, in_size, out_size):
        super().__init__()
        # Инициализация весов и смещений случайными значениями
        self.weight = torch.nn.Parameter(2 * (torch.rand((in_size, out_size)) - 0.5))
        self.bias = torch.nn.Parameter(2 * (torch.rand((out_size,)) - 0.5))

    def forward(self, x):
        '''
            Прямой проход линейного слоя.
            x @ weight — матричное умножение.
        '''
        return x @ self.weight + self.bias


class TorchTrain:
    '''
        Класс‑обёртка для обучения модели на PyTorch.
        Содержит:
        - модель
        - методы для запуска forward на одной или многих точках
        - цикл обучения с ручным обновлением параметров
    '''
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        '''
            Прогоняет одну точку через модель.
        '''
        return self.model.forward(torch.tensor([x]))

    def run_many(self, X):
        '''
            Прогоняет множество точек через модель.
            detach() — отключает вычисление градиентов.
        '''
        return self.model.forward(torch.tensor(X)).detach()

    def train(
        self,
        data,
        learning_rate,
        max_epochs=500,
        log_fn=default_log_fn,
    ):
        '''
            Основной цикл обучения:
            1) Прямой проход
            2) Вычисление логарифмической функции потерь
            3) Обратное распространение ошибки (backward)
            4) Ручное обновление параметров модели
            5) Логирование каждые 10 эпох
        '''
        self.model = Network(self.hidden_layers)
        self.max_epochs = max_epochs
        model = self.model

        losses = []
        for epoch in range(1, max_epochs + 1):

            # Forward: получаем предсказания
            out = model.forward(torch.tensor(data.X, requires_grad=True)).view(data.N)
            y = torch.tensor(data.y)

            # Лог‑лосс: -log(p) для правильного класса
            probs = (out * y) + (out - 1.0) * (y - 1.0)
            loss = -probs.log().sum()

            # Backward: вычисление градиентов
            loss.view(1).backward()

            # Обновление параметров вручную
            for p in model.parameters():
                if p.grad is not None:
                    p.data = p.data - learning_rate * (p.grad / float(data.N))
                    p.grad.zero_()

            # Подсчёт точности
            pred = out > 0.5
            correct = ((y == 1) * (pred)).sum() + ((y == 0) * (~pred)).sum()

            # Логирование
            loss_num = loss.reshape(-1).item()
            losses.append(loss_num)

            if epoch % 10 == 0 or epoch == max_epochs:
                log_fn(epoch, loss_num, correct.item(), losses)


if __name__ == "__main__":
    '''
        Пример запуска обучения на датасете XOR.
        Это демонстрация того, как работает PyTorch‑версия модели.
    '''
    PTS = 250
    HIDDEN = 10
    RATE = 0.5
    TorchTrain(HIDDEN).train(minitorch.datasets["Xor"](PTS), RATE)
