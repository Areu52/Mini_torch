"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch
import matplotlib.pyplot as plt
import time


def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        # ASSIGN2.5
        h = self.layer1.forward(x).relu()
        h = self.layer2.forward(h).relu()
        return self.layer3.forward(h).sigmoid()
        # END ASSIGN2.5


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        # ASSIGN2.5
        batch, in_size = x.shape
        return (
            self.weights.value.view(1, in_size, self.out_size)
            * x.view(batch, in_size, 1)
        ).sum(1).view(batch, self.out_size) + self.bias.value.view(self.out_size)
        # END ASSIGN2.5


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):

        start = time.time() # Я добавил
        
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        correct = 0
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            optim.zero_grad()
            correct = 0
            
            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)

        total_time = time.time() - start
        epoch_time = total_time / self.max_epochs
        return correct, float(total_loss), epoch_time, losses


# Функция для визуализации
def visualize(data, model, name):
    X = data.X
    y = data.y
    plt.figure(figsize=(5, 5))

    # Рисуем точки датасета
    for i in range(len(X)):
        color = "red" if y[i] == 1 else "blue" # Класс 1 — красный, класс 0 — синий
        plt.scatter(X[i][0], X[i][1], color=color)
        
    # cетка
    xs = ys = [i / 100 for i in range(100)]
    grid = [[model.run_many([[x, y]]).item() for x in xs] for y in ys]# Для каждой точки сетки считаем предсказание модели
    
    plt.imshow(grid, extent=[0, 1, 0, 1], origin="lower", cmap="RdBu", alpha=0.5)
    plt.title(f"{name} — Tensor Model")
    plt.savefig(f"images_tensor/{name}_tensor.png")
    plt.close()


if __name__ == "__main__":
    PTS = 50
    RATE = 0.1
    datasets = ["Simple", "Split", "Xor", "Circle"]
    # Для каждого датасета — своё количество нейронов в скрытом слое.
    hidden_sizes = {"Simple": 2, "Split": 2, "Xor": 10, "Circle": 10}

    print("\n=== Tensor Model Training Results ===\n")
    print(f"{'Dataset':10} | {'Correct':7} | {'Loss':10} | {'Epoch Time (s)':14}")
    print("-" * 50)

    for name in datasets:
        data = minitorch.datasets[name](PTS)
        HIDDEN = hidden_sizes[name]
        trainer = TensorTrain(HIDDEN)
        correct, loss, epoch_time, losses = trainer.train(data, RATE)
        print(f"{name:10} | {correct:7} | {loss:10.4f} | {epoch_time:14.4f}")
        visualize(data, trainer, name)
