import streamlit as st
import torch
from gan import build_generator as Generator  # Импорт твоей модели GAN
import numpy as np
from PIL import Image

# Загрузка модели
@st.cache_resource
def load_model():
    generator = Generator()  # Инициализация генератора
    generator.load_state_dict(torch.load('generator.pth', map_location='cpu'))
    generator.eval()
    return generator

def generate_image(generator, noise_dim=100):
    noise = torch.randn(1, noise_dim, 1, 1)  # Шум на вход GAN
    with torch.no_grad():
        generated_img = generator(noise).squeeze().permute(1, 2, 0).numpy()
    generated_img = (generated_img * 255).astype(np.uint8)
    return Image.fromarray(generated_img)

# Интерфейс
st.title("Генератор изображений с GAN")
if st.button("Сгенерировать изображение"):
    generator = load_model()
    img = generate_image(generator)
    st.image(img, caption="Сгенерированное изображение", use_column_width=True)