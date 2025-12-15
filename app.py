import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import pandas as pd

# --- CONFIGURAÇÃO DA PÁGINA (Deve ser a primeira linha) ---
st.set_page_config(
    page_title="DermoScan AI",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CSS PERSONALIZADO (ESTILO MINIMALISTA AZUL) ---
st.markdown("""
    <style>
    /* Esconder menus padrão do Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Fundo geral (Azul Suave definido por ti) */
    .stApp {
        background-color: #5d8aa8;
    }

    /* Estilo do Título */
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 300;
        color: white; /* Mudei para branco para melhor contraste no fundo azul */
        text-align: center;
        padding-bottom: 20px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }

    p {
        color: #f0f0f0; /* Texto secundário mais claro */
    }

    /* Estilo dos Botões */
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #2E86C1;
        color: white;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1B4F72;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    /* Cartão de Resultados (Métricas) */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        color: #333;
    }
    [data-testid="stMetricLabel"] {
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CLASSES DINÂMICAS ---
# Ajusta este caminho se mudares de computador
dataset_path = r"C:\Users\Daniel\Desktop\3_ano\IC\TP\Dataset\Skin_Diseases\kaggle\train"


def get_classes_from_folder(path):
    if os.path.exists(path):
        # 1. Listar tudo na pasta
        items = os.listdir(path)
        # 2. Filtrar apenas o que são pastas e ORDENAR ALFABETICAMENTE
        # A ordenação é fundamental para corresponder aos índices do modelo (0, 1, 2...)
        return sorted([d for d in items if os.path.isdir(os.path.join(path, d))])
    else:
        # Fallback de segurança caso a pasta não seja encontrada
        return ['Acne', 'Eczema', 'Melanoma', 'Psoriasis', 'Seborrheic Keratosis']


# Executar a função ao iniciar a app
class_names = get_classes_from_folder(dataset_path)


# --- FUNÇÕES ---
@st.cache_resource
def load_model():
    model_path = r'C:\Users\Daniel\Desktop\3_ano\IC\TP\Source\Fase 3\modelo_final_mobilenet.h5'
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Erro ao ler o ficheiro do modelo: {e}")
            return None
    return None


def predict(image, model):
    # 1. Redimensionar para 224x224 (MobileNetV2 standard)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # 2. Converter para array e normalizar (0-1)
    img = np.asarray(image)
    img = img / 255.0
    img_reshape = np.expand_dims(img, axis=0)

    if model:
        prediction = model.predict(img_reshape)
    else:
        # Simulação para UI (caso o modelo não carregue)
        prediction = np.random.rand(1, len(class_names))
        prediction = prediction / np.sum(prediction)

    return prediction


# --- LAYOUT DA APP ---

# 1. Cabeçalho
st.markdown("<h1>🩺 DermoScan AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Diagnóstico assistido por Inteligência Artificial</p>",
            unsafe_allow_html=True)
st.write("")

# Carregar modelo
model = load_model()

# 2. Área de Input
tab1, tab2 = st.tabs(["📁 Carregar Foto", "📸 Usar Câmara"])
file = None

with tab1:
    file_upload = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if file_upload: file = file_upload

with tab2:
    camera_upload = st.camera_input("Tire uma foto")
    if camera_upload: file = camera_upload

# 3. Área de Visualização e Resultados
if file is not None:
    st.write("---")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        image = Image.open(file)
        st.image(image, use_container_width=True)
        analyze_btn = st.button("🔍 Analisar Imagem")

    if analyze_btn:
        with st.spinner('A processar imagem...'):
            predictions = predict(image, model)

            # Processar dados
            class_index = np.argmax(predictions)
            class_name = class_names[class_index]
            confidence = np.max(predictions) * 100

            # --- RESULTADOS ---
            st.markdown("---")

            # Usar containers brancos para destacar os resultados do fundo azul
            with st.container():
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric(label="Diagnóstico Previsto", value=class_name)
                with m_col2:
                    st.metric(label="Grau de Confiança", value=f"{confidence:.1f}%")

            st.write("")
            st.caption("Detalhe das Probabilidades:")

            # Criar DataFrame com os nomes das classes corretos
            probs_df = pd.DataFrame(
                predictions[0],
                index=class_names,  # Usa a lista dinâmica aqui
                columns=['Probabilidade']
            )

            # Gráfico de barras
            st.bar_chart(probs_df, color="#2E86C1")

elif model is None:
    st.warning("⚠️ Modo Demonstração (Modelo 'modelo_final_mobilenet.h5' não encontrado)")
    st.info(f"Classes configuradas: {', '.join(class_names)}")
else:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #E0E0E0;'>Carregue uma imagem para começar</p>",
                unsafe_allow_html=True)

# Rodapé
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 12px; color: #E0E0E0;'>Projeto Inteligência Computacional • Fase III</p>",
    unsafe_allow_html=True)