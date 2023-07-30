# from sqlalchemy import column
import streamlit as st
import plotly_express as px
import pandas as pd
# from torch import layout
from pathlib import Path
import plotly.graph_objects as go

# configuration
st.set_option('deprecation.showfileUploaderEncoding', False)

# Judul Halaman
st.set_page_config(
    page_title="Regresi Data ISPU ELM",
    page_icon="📈",
)
# Add a sidebar
st.sidebar.subheader("PENGATURAN VISUALISASI")
#-------------------------------------------------------------VISUALISASI METODE Extreme Learning Machine-------------------------------------------------------------------------

metode = st.sidebar.radio("Pilih Metode",['Extreme Learning Machine', 'Kernel Extreme Learning Machine'])

if metode == "Extreme Learning Machine":
    # title of the app
    st.title("Website Visualisasi Prediksi dan Forecasting Data ISPU dengan metode Extreme Learning Machine")

    # Setup selecting csv file
    file_select = st.sidebar.selectbox(
        label="Pilih file",
        options=[

                    "Prediksi DKI1 PM10", "Prediksi DKI1 SO2", "Prediksi DKI1 CO", "Prediksi DKI1 O3", "Prediksi DKI1 NO2", "Forecasting DKI1",
                    "Prediksi DKI2 PM10", "Prediksi DKI2 SO2", "Prediksi DKI2 CO", "Prediksi DKI2 O3", "Prediksi DKI2 NO2", "Forecasting DKI2",
                    "Prediksi DKI3 PM10", "Prediksi DKI3 SO2", "Prediksi DKI3 CO", "Prediksi DKI3 O3", "Prediksi DKI3 NO2", "Forecasting DKI3",
                    "Prediksi DKI4 PM10", "Prediksi DKI4 SO2", "Prediksi DKI4 CO", "Prediksi DKI4 O3", "Prediksi DKI4 NO2", "Forecasting DKI4",
                    "Prediksi DKI5 PM10", "Prediksi DKI5 SO2", "Prediksi DKI5 CO", "Prediksi DKI5 O3", "Prediksi DKI5 NO2", "Forecasting DKI5",
                ]
    )

    global df

    # Pemilihan file prediksi atau forecasting
    if file_select == "Prediksi DKI1 PM10":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 1/DKI1_PM10_pred.xlsx'
    elif file_select == "Prediksi DKI1 SO2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 1/DKI1_SO2_pred.xlsx'
    elif file_select == "Prediksi DKI1 CO":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 1/DKI1_CO_pred.xlsx'
    elif file_select == "Prediksi DKI1 O3":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 1/DKI1_O3_pred.xlsx'
    elif file_select == "Prediksi DKI1 NO2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 1/DKI1_NO2_pred.xlsx'
    elif file_select == "Forecasting DKI1":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM forecasting/ALL DKI FORECAST/DKI1_FORECAST.xlsx'
    if file_select == "Prediksi DKI2 PM10":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 2/DKI2_PM10_pred.xlsx'
    elif file_select == "Prediksi DKI2 SO2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 2/DKI2_SO2_pred.xlsx'
    elif file_select == "Prediksi DKI2 CO":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 2/DKI2_CO_pred.xlsx'
    elif file_select == "Prediksi DKI2 O3":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 2/DKI2_O3_pred.xlsx'
    elif file_select == "Prediksi DKI2 NO2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 2/DKI2_NO2_pred.xlsx'
    elif file_select == "Forecasting DKI2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM forecasting/ALL DKI FORECAST/DKI2_FORECAST.xlsx'
    if file_select == "Prediksi DKI3 PM10":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 3/DKI3_PM10_pred.xlsx'
    elif file_select == "Prediksi DKI3 SO2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 3/DKI3_SO2_pred.xlsx'
    elif file_select == "Prediksi DKI3 CO":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 3/DKI3_CO_pred.xlsx'
    elif file_select == "Prediksi DKI3 O3":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 3/DKI3_O3_pred.xlsx'
    elif file_select == "Prediksi DKI3 NO2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 3/DKI3_NO2_pred.xlsx'
    elif file_select == "Forecasting DKI3":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM forecasting/ALL DKI FORECAST/DKI3_Forecast.xlsx'
    if file_select == "Prediksi DKI4 PM10":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 4/DKI4_PM10_pred.xlsx'
    elif file_select == "Prediksi DKI4 SO2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 4/DKI4_SO2_pred.xlsx'
    elif file_select == "Prediksi DKI4 CO":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 4/DKI4_CO_pred.xlsx'
    elif file_select == "Prediksi DKI4 O3":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 4/DKI4_O3_pred.xlsx'
    elif file_select == "Prediksi DKI4 NO2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 4/DKI4_NO2_pred.xlsx'
    elif file_select == "Forecasting DKI4":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM forecasting/ALL DKI FORECAST/DKI4_Forecast.xlsx'
    if file_select == "Prediksi DKI5 PM10":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 5/DKI5_PM10_pred.xlsx'
    elif file_select == "Prediksi DKI5 SO2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 5/DKI5_SO2_pred.xlsx'
    elif file_select == "Prediksi DKI5 CO":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 5/DKI5_CO_pred.xlsx'
    elif file_select == "Prediksi DKI5 O3":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 5/DKI5_O3_pred.xlsx'
    elif file_select == "Prediksi DKI5 NO2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM prediction/DKI 5/DKI5_NO2_pred.xlsx'
    elif file_select == "Forecasting DKI5":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/ELM forecasting/ALL DKI FORECAST/DKI5_Forecast.xlsx'

    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        print(e)
        df = pd.read_csv(uploaded_file)

    global numeric_columns
    global non_numeric_columns
    st.write(df)
    date_column = list(df.select_dtypes(['datetime']).columns)
    numeric_columns = list(df.select_dtypes(['float', 'int']).columns)
    non_numeric_columns = list(df.select_dtypes(['object']).columns)
    non_numeric_columns.append(None)
    # print(non_numeric_columns)

    # Plotting
    st.sidebar.subheader("Pengaturan Plot Garis")
    # Select Box dan Multiselect untuk pemilihan data atau warna yang ingin ditampilkan
    x_values = st.sidebar.selectbox('X axis', options=date_column)
    y_values = st.sidebar.multiselect('Y axis', options=numeric_columns, default=numeric_columns)
    color_values = st.sidebar.multiselect("Warna Kategori ISPU", options=["Baik", "Sedang", "Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"], default=["Baik", "Sedang"])
        
    # Keterangan DKI1 sampai DKI5
    st.sidebar.markdown(
        """
            Keterangan:\n
            DKI1 :  Bundaran HI, DKI Jakarta\n
            DKI2 :  Kelapa Gading, DKI Jakarta\n
            DKI3 :  Jagakarsa, DKI Jakarta\n
            DKI4 :  Lubang Buaya, DKI Jakarta\n
            DKI5 :  Kebon Jeruk, DKI Jakarta
        """
    )

    # Pembuatan Data Frame Area Kategori ISPU
    area = {"Tanggal":df["Tanggal"], "Baik":50, "Sedang":50, "Tidak Sehat":100, "Sangat Tidak Sehat":100, "Berbahaya":100}
    df_area = pd.DataFrame(area)

    # Membuat skema warna 
    color_sequence = ["#4FF04F", "#55A6E4", "#FFFF54", "#FF5454", "#322F2F"]
    # membuat plot area ISPU
    plot1 = px.area(data_frame=df_area, x="Tanggal", y=color_values, color_discrete_sequence=color_sequence)

    # Membuat skema warna 
    color_plot_sequence = ["#00007D", "#B30000", "#B2B200", "#000000", "#510077"]
    # membuat plot garis
    plot2= px.line(data_frame=df, x=x_values, y=y_values, color_discrete_sequence=color_plot_sequence)

    # Penggabungan Plot Line dan Area
    plot3 = go.Figure(data=plot1.data + plot2.data)
    st.plotly_chart(plot3)

#-------------------------------------------------------------VISUALISASI METODE K-ELM-------------------------------------------------------------------------

if metode == "Kernel Extreme Learning Machine":
    # title of the app
    st.title("Website Visualisasi Prediksi dan Forecasting Data ISPU dengan metode Kernel-Extreme Learning Machine")

    # Add a sidebar
    st.sidebar.subheader("Pengaturan Visualisasi")

    # Setup selecting csv file
    file_select = st.sidebar.selectbox(
        label="Pilih file",
        options=[

                    "Prediksi DKI1 PM10", "Prediksi DKI1 SO2", "Prediksi DKI1 CO", "Prediksi DKI1 O3", "Prediksi DKI1 NO2", "Forecasting DKI1",
                    "Prediksi DKI2 PM10", "Prediksi DKI2 SO2", "Prediksi DKI2 CO", "Prediksi DKI2 O3", "Prediksi DKI2 NO2", "Forecasting DKI2",
                    "Prediksi DKI3 PM10", "Prediksi DKI3 SO2", "Prediksi DKI3 CO", "Prediksi DKI3 O3", "Prediksi DKI3 NO2", "Forecasting DKI3",
                    "Prediksi DKI4 PM10", "Prediksi DKI4 SO2", "Prediksi DKI4 CO", "Prediksi DKI4 O3", "Prediksi DKI4 NO2", "Forecasting DKI4",
                    "Prediksi DKI5 PM10", "Prediksi DKI5 SO2", "Prediksi DKI5 CO", "Prediksi DKI5 O3", "Prediksi DKI5 NO2", "Forecasting DKI5",
                ]
    )

    global df1

    # Pemilihan file prediksi atau forecasting
    if file_select == "Prediksi DKI1 PM10":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI1/DKI1_PM10_prediksi.xlsx'
    elif file_select == "Prediksi DKI1 SO2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI1/DKI1_SO2_prediksi.xlsx'
    elif file_select == "Prediksi DKI1 CO":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI1/DKI1_CO_prediksi.xlsx'
    elif file_select == "Prediksi DKI1 O3":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI1/DKI1_O3_prediksi.xlsx'
    elif file_select == "Prediksi DKI1 NO2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI1/DKI1_NO2_prediksi.xlsx'
    elif file_select == "Forecasting DKI1":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K- ELM forecasting/ALL_DKI_FORECAST/DKI1_Forecast.xlsx'
    if file_select == "Prediksi DKI2 PM10":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI2/DKI2_PM10_prediksi.xlsx'
    elif file_select == "Prediksi DKI2 SO2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI2/DKI2_SO2_prediksi.xlsx'
    elif file_select == "Prediksi DKI2 CO":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI2/DKI2_CO_prediksi.xlsx'
    elif file_select == "Prediksi DKI2 O3":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI2/DKI2_O3_prediksi.xlsx'
    elif file_select == "Prediksi DKI2 NO2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI2/DKI2_NO2_prediksi.xlsx'
    elif file_select == "Forecasting DKI2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K- ELM forecasting/ALL_DKI_FORECAST/DKI2_Forecast.xlsx'
    if file_select == "Prediksi DKI3 PM10":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI3/DKI3_PM10_prediksi.xlsx'
    elif file_select == "Prediksi DKI3 SO2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI3/DKI3_SO2_prediksi.xlsx'
    elif file_select == "Prediksi DKI3 CO":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI3/DKI3_CO_prediksi.xlsx'
    elif file_select == "Prediksi DKI3 O3":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI3/DKI3_O3_prediksi.xlsx'
    elif file_select == "Prediksi DKI3 NO2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI3/DKI3_NO2_prediksi.xlsx'
    elif file_select == "Forecasting DKI3":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K- ELM forecasting/ALL_DKI_FORECAST/DKI3_Forecast.xlsx'
    if file_select == "Prediksi DKI4 PM10":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI4/DKI4_PM10_prediksi.xlsx'
    elif file_select == "Prediksi DKI4 SO2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI4/DKI4_SO2_prediksi.xlsx'
    elif file_select == "Prediksi DKI4 CO":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI4/DKI4_CO_prediksi.xlsx'
    elif file_select == "Prediksi DKI4 O3":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI4/DKI4_O3_prediksi.xlsx'
    elif file_select == "Prediksi DKI4 NO2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI4/DKI4_NO2_prediksi.xlsx'
    elif file_select == "Forecasting DKI4":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K- ELM forecasting/ALL_DKI_FORECAST/DKI4_Forecast.xlsx'
    if file_select == "Prediksi DKI5 PM10":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI5/DKI5_PM10_prediksi.xlsx'
    elif file_select == "Prediksi DKI5 SO2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI5/DKI5_SO2_prediksi.xlsx'
    elif file_select == "Prediksi DKI5 CO":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI5/DKI5_CO_prediksi.xlsx'
    elif file_select == "Prediksi DKI5 O3":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI5/DKI5_O3_prediksi.xlsx'
    elif file_select == "Prediksi DKI5 NO2":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K-ELM prediction/DKI5/DKI5_NO2_prediksi.xlsx'
    elif file_select == "Forecasting DKI5":
        uploaded_file = Path(__file__).parents[2] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/files/K- ELM forecasting/ALL_DKI_FORECAST/DKI5_Forecast.xlsx'

    try:
        df1 = pd.read_excel(uploaded_file)
    except Exception as e:
        print(e)
        df1 = pd.read_csv(uploaded_file)

    global numeric_columns1
    global non_numeric_columns1
    st.write(df1)
    date_column = list(df1.select_dtypes(['datetime']).columns)
    numeric_columns1 = list(df1.select_dtypes(['float', 'int']).columns)
    non_numeric_columns1 = list(df1.select_dtypes(['object']).columns)
    non_numeric_columns1.append(None)
    # print(non_numeric_columns)

    # Plotting
    st.sidebar.subheader("Pengaturan Plot Garis")
    # Select Box dan Multiselect untuk pemilihan data atau warna yang ingin ditampilkan
    x_values = st.sidebar.selectbox('X axis', options=date_column)
    y_values = st.sidebar.multiselect('Y axis', options=numeric_columns1, default=numeric_columns1)
    color_values = st.sidebar.multiselect("Warna Kategori ISPU", options=["Baik", "Sedang", "Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"], default=["Baik", "Sedang"])
        
    # Keterangan DKI1 sampai DKI5
    st.sidebar.markdown(
        """
            Keterangan:\n
            DKI1 :  Bundaran HI, DKI Jakarta\n
            DKI2 :  Kelapa Gading, DKI Jakarta\n
            DKI3 :  Jagakarsa, DKI Jakarta\n
            DKI4 :  Lubang Buaya, DKI Jakarta\n
            DKI5 :  Kebon Jeruk, DKI Jakarta
        """
    )

    # Pembuatan Data Frame Area Kategori ISPU
    area = {"Tanggal":df1["Tanggal"], "Baik":50, "Sedang":50, "Tidak Sehat":100, "Sangat Tidak Sehat":100, "Berbahaya":100}
    df1_area = pd.DataFrame(area)

    # Membuat skema warna 
    color_sequence = ["#4FF04F", "#55A6E4", "#FFFF54", "#FF5454", "#322F2F"]
    # membuat plot area ISPU
    plot1 = px.area(data_frame=df1_area, x="Tanggal", y=color_values, color_discrete_sequence=color_sequence)

    # Membuat skema warna 
    color_plot_sequence = ["#00007D", "#B30000", "#B2B200", "#000000", "#510077"]
    # membuat plot garis
    plot2= px.line(data_frame=df1, x=x_values, y=y_values, color_discrete_sequence=color_plot_sequence)

    # Penggabungan Plot Line dan Area
    plot3 = go.Figure(data=plot1.data + plot2.data)
    st.plotly_chart(plot3)