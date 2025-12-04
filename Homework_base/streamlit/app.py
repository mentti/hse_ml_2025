import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
import plotly.express as px

st.set_page_config(
    page_title="Car price predicting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кэшируем модель (загружается только один раз)
@st.cache_resource 
def load_model():
    with open("streamlit/models/cars_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

#Поскольку у нас требования в задании непонятно как сформулированы сделаем функции для загрузки всего подряд, чтобы на всем подряд поделать EDA. >:|
#Это основной датасет. Поскольку деплоится на стримлит не от папки где лежит app.py, а от домашней папки нужно поменять пути ниже для локального запуска!
@st.cache_data
def load_train_raw():
    return pd.read_csv("streamlit/data/cars_train.csv")

#Это предобработанный датасет
@st.cache_data
def load_train_prep():
    return pd.read_csv("streamlit/data/cars_train_prep_OHE.csv")

#А это 
@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)
    
#Добавляем функцию, с помощью которой предобрабатывали числовые признаки в пайплайне
def parse_numbers(data: pd.DataFrame):
    df = data.copy()
    cols = df.select_dtypes(include=['object']).columns.to_list()
    for col in cols:
        df[col] = df[col].str.split(' ').str[0].replace(["", " "], np.nan).astype(float) 
    return df

#Функция для получения целевой переменной, если она есть в данных
def take_selling_price(df):
    X = df.copy()
    y_target = None
    if 'selling_price' in df.columns:
        y_target = df['selling_price']
        X = X.drop(columns='selling_price', axis = 1)
    return X, y_target

#Загружаем модель
model = load_model()

#Сделаем меню слева, чтобы в многообразии требований мы смогли найти что нужно. Немного заморочимся с кнопками, чтобы они сохраняли состояния и позволили нам сделать полноценную менюшку.
st.title("Модель предсказания цен на автомобили") 
st.sidebar.title("Сегодня в Меню")

if st.sidebar.button("Сырая EDA (train)"):
    st.session_state.page = "EDA_raw"

if st.sidebar.button("Готовая EDA (train clean)"):
    st.session_state.page = "EDA_clean"

if st.sidebar.button("Приготовим вашу CSV"):
    st.session_state.page = "Predict"

if st.sidebar.button("Посмотреть что на кухне (коэффициенты)"):
    st.session_state.page = "Coefs"

#Обрабатываем, если ни одна кнопка не нажата
if "page" not in st.session_state:
    st.session_state.page = "EDA_raw"

page = st.session_state["page"]

#Идем уже в нажатые кнопки из нажатий выше в разделы
if page == 'EDA_raw':
    st.header('Исследование данных (EDA) до обработки для модели')

    #Загружаем сырые данные 
    df_tr = load_train_raw()

    st.subheader("Случайные 20 строк тренировочного датасета")
    st.dataframe(df_tr.sample(30))
    
    st.subheader("Первые 5 и последние 5 строк тренировочного датасета")
    st.dataframe(df_tr.head(5))
    st.dataframe(df_tr.tail(5))

    st.subheader("Описание числовых признаков")
    st.table(df_tr.describe())

    #Посчитаем пропуски, чтобы было красиво, а не как мы делали в ноутбуке (подсмотрел у LLM)
    st.subheader("Пропуски в данных")
    missing = df_tr.isna().sum().to_frame('missing')
    missing = missing[missing['missing'] > 0].sort_values('missing', ascending=False)
    st.dataframe(missing)

    #Распределение таргета
    if "selling_price" in df_tr.columns:
        st.subheader("Распределение целевой переменной (selling_price)")
        fig_price = px.histogram(df_tr, x='selling_price', nbins=50)
        st.plotly_chart(fig_price, use_container_width=True)

    #Распределение числовых признаков
    num_cols = df_tr.select_dtypes(exclude=['object']).drop(columns='selling_price', axis = 1).columns.tolist()
    st.subheader("Распределение числовых признаков")
    col_num = st.selectbox("Выберите числовой признак", num_cols, index=0)
    fig_num = px.histogram(df_tr, x=col_num, nbins=50)
    st.plotly_chart(fig_num, use_container_width=True)
    
    #Распределение категориальных признаков
    cat_cols = df_tr.select_dtypes(include=["object"]).columns.tolist()
    st.subheader("Распределение категориальных признаков")
    col_cat = st.selectbox("Выберите категориальный признак", cat_cols, index=0)
    fig_cat = px.histogram(df_tr, x=col_cat)
    st.plotly_chart(fig_cat, use_container_width=True)
    
    #Корреляции числовых признаков c таргетом
    st.subheader("Корреляционная матрица числовых признаков")
    num_cols_target = df_tr.select_dtypes(exclude=['object']).columns.tolist()
    corr = df_tr[num_cols_target].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu"    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
elif page == "EDA_clean":
    st.header('Исследование данных (EDA) после обработки для модели')

    #Повторяем тоже самое, но для очищенных данных
    df_trpr = load_train_prep()

    st.subheader("Случайные 20 строк тренировочного датасета")
    st.dataframe(df_trpr.sample(30))
    
    st.subheader("Первые 5 и последние 5 строк тренировочного датасета")
    st.dataframe(df_trpr.head(5))
    st.dataframe(df_trpr.tail(5))

    st.subheader("Описание числовых признаков")
    st.dataframe(df_trpr.describe())

    #Посчитаем пропуски, чтобы было красиво, а не как мы делали в ноутбуке (подсмотрел у LLM)
    st.subheader("Пропуски в данных")
    missing = df_trpr.isna().sum().to_frame('missing')
    missing = missing[missing['missing'] > 0].sort_values('missing', ascending=False)
    st.dataframe(missing)

    #Распределение таргета тут нет

    #Распределение признаков
    num_cols = df_trpr.select_dtypes(exclude=['object']).columns.tolist()
    st.subheader("Распределение числовых признаков")
    col_num = st.selectbox("Выберите числовой признак", num_cols, index=0)
    fig_num = px.histogram(df_trpr, x=col_num, nbins=50)
    st.plotly_chart(fig_num, use_container_width=True)
    
    #Распределения категориальных тоже нет после OHE
        
    #Корреляции числовых признаков c таргетом
    st.subheader("Корреляционная матрица числовых признаков")
    num_cols_target = df_trpr.select_dtypes(exclude=['object']).columns.tolist()
    corr = df_trpr[num_cols_target].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu"    )
    st.plotly_chart(fig_corr, use_container_width=True)

elif page == "Predict":
    st.header('Результаты модели по вашим данным (загрузка CSV)')

    uploaded_file = st.file_uploader("Загрузите CSV с данными автомобилей", type=["csv"])

    if uploaded_file is not None:
        df_user = load_csv(uploaded_file)

        st.subheader("Вы загрузили следующие данные (первые 5 и последние 5):")
        st.dataframe(df_user.head(5))
        st.dataframe(df_user.tail(5))
    
        #Определяем сеты обучения и тренинга (если есть)
        X, y_true = take_selling_price(df_user)
    
        #Получаем предсказания (без обработки, мы в пайп уже все зашили)
        y_pred = model.predict(X)
    
        #Соберем все вместе, добавим к датасету колонку с предсказаниями
        df_result = df_user.copy()
        df_result["predicted_selling_price"] = y_pred
        st.subheader("Получились такие предсказания (первые 10):")
        st.dataframe(df_result.head(10))
    
        # Если в данных была настоящая цена, покажем метрики
        if y_true is not None:
            st.subheader("Оценка качества модели на загруженном датасете")
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            
            #Вывод результатов в колонках (подсмотрено у LLM)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R2", f"{r2:.3f}")
            with col2:
                st.metric("MSE", f"{mse:,.0f}")
            
            #Вывод результатов в графике (подсмотрено у LLM)
            st.subheader("Факт vs прогноз (первые 200 точек)")
            n = min(len(y_true), 200)
            comp = pd.DataFrame({
                "y_true": y_true.iloc[:n].values,
                "y_pred": y_pred[:n],
            })
            st.line_chart(comp)
    
            #Скачать результаты
            csv_out = df_result.to_csv(index=False).encode("utf-8")
            st.download_button("Скачать результат", data=csv_out, file_name="predictions.csv", mime="text/csv", )
    else:
        st.write("Загрузите CSV-файл для получения предсказаний.")

elif page == "Coefs":
    st.header('Коэффициенты обученной модели')

    #Тут получаем признаки / веса модели и выводим их в dataframe и bar_chart (частично подсмотрено у LLM)
    prep = model.named_steps["prep"]
    ridge: Ridge = model.named_steps["model"]

    feature_names = prep.get_feature_names_out()
    coefs = ridge.coef_

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)

    st.subheader("Таблица коэффициентов (по модулю)")
    st.dataframe(coef_df.head(50))

    st.subheader("График коэффициентов (топ-30 по модулю)")
    top = coef_df.head(30).set_index("feature")["coef"]
    st.bar_chart(top)

    st.caption(f"Свободный член (intercept): {ridge.intercept_:.2f}")








    
