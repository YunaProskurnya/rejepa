# Данные

Датасеты хранятся отдельно от Git. В этой папке отслеживается только инструкция;
изображения и архивы игнорируются. Все команды выполняются из корня проекта.
Если данные уже есть на диске или сервере, достаточно указать путь к ним через
`data` в конфиге или `--data` в командной строке.

## ImageNet100

Основной датасет — [ambityga/imagenet100 на Kaggle](https://www.kaggle.com/datasets/ambityga/imagenet100).
Рабочие команды скачивания и объединения частей находятся
в [основном README](../README.md#imagenet100) и [скрипте подготовки](../scripts/setup_imagenet100.sh).
После их выполнения структура выглядит так:

```text
data/imagenet100/
  train/
    n01440764/       # пример имени класса
      image.jpg
    ...
  val/
    n01440764/
      image.jpg
    ...
```

В `train` и `val` должны быть одинаковые 100 папок классов, с разными изображениями.
Используйте одинаковые классы и разбиение во всех сравниваемых экспериментах:
одно название ImageNet100 этого не гарантирует. Список классов сохраняется
в `runs/<имя-запуска>/config.json` и в checkpoint.

Шаблон `configs/imagenet100_vit_base.json` уже указывает на `data/imagenet100/train`.
Если папка находится в другом месте, переопределите только путь:

```sh
python scripts/train.py --config configs/imagenet100_vit_base.json --data /path/to/imagenet100/train --output runs/imagenet100-check
```

Указывайте папку `train`, а не её родителя и не отдельную папку класса.
Для kNN путь к `val` передаётся отдельно в `scripts/evaluate_knn.py`.

## Дополнительно: CIFAR-10

Для основного эксперимента на ImageNet100 этот раздел не нужен.

Разовое скачивание штатными средствами torchvision:

```sh
python -c "from torchvision.datasets import CIFAR10; CIFAR10(root='data/cifar10', train=True, download=True)"
```

Затем короткая проверка обучения:

```sh
python scripts/train.py --dataset cifar10 --data data/cifar10 --batch-size 4 --max-steps 10 --output runs/cifar10-check
```

`data/cifar10` должна содержать `cifar-10-batches-py`. CIFAR-10 здесь служит
для проверки пайплайна; изображения приводятся к 224×224. Текущий скрипт kNN
работает с ImageFolder train/val, а не с архивным форматом CIFAR-10.
