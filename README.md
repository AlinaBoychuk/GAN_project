# Генератор реалистичных изображений на основе GAN

Проект реализует генеративно-состязательную сеть (GAN) для создания реалистичных изображений лиц с возможностью интерактивной генерации через веб-интерфейс.

## 🔥 Особенности
- **Архитектура**: DCGAN с градиентным штрафом (WGAN-GP)
- **Размер изображений**: 64×64 пикселя
- **Интерфейс**: Интерактивное веб-приложение на Streamlit
- **Поддержка**: Сохранение и просмотр сгенерированных изображений

## 🚀 Быстрый старт

### 1. Клонирование репозитория
```bash
git clone https://github.com/AlinaBoychuk/GAN_project.git
cd GAN_project
```

### 2. Установка зависимостей
```bash
pip install -r requirements.txt
```
Файл requirements.txt должен содержать:
```bash
text
streamlit
torch
torchvision
numpy
pillow
tensorflow
matplotlib
tqdm
```
### 3. Подготовка модели
Поместите обученные веса генератора в файл generator.pth в корне проекта

(Опционально) Для обучения с нуля выполните:

```bash
python train_gan.py
```
### 4. Запуск интерфейса
```bash
streamlit run app.py
```
Приложение будет доступно по адресу: http://localhost:8501

🖥️ Интерфейс приложения
https://interface_screenshot.png

Интерфейс предоставляет:

- Кнопку для генерации новых изображений
- Отображение сгенерированных изображений
- Возможность сохранения результатов

🧠 Архитектура модели
```python
# Генератор
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(4*4*512, use_bias=False, input_shape=(LATENT_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Reshape((4, 4, 512)),
        # ... остальные слои ...
    ])
    return model
```

📂 Структура проекта
```text
GAN_project/
├── app.py                # Streamlit-интерфейс
├── gan.py               # Архитектура GAN
├── generator.pth        # Веса обученной модели
├── img_align_celeba/    # Датесет CelebA
├── requirements.txt     # Зависимости
├── README.md
└── images/            # Примеры сгенерированных изображений
```
👨‍💻 Разработчики
@AlinaBoychuk
