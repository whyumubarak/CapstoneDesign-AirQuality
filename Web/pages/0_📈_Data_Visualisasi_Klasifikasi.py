import pandas as pd
import streamlit as st
import folium
from geopy.geocoders import Nominatim
from streamlit_folium import folium_static
from datetime import date, timedelta

st.set_page_config(
    page_title="Klasifikasi",
    page_icon="📈",
)
# Add a sidebar
st.sidebar.subheader("PENGATURAN VISUALISASI")
visualisasi = st.sidebar.radio("Pilih Visualisasi", ['Dataset ISPU', 'Metode Extreme Learning Machine', 'Metode Kernel Extreme Learning Machine'])

#-------------------------------------------------------------VISUALISASI METODE Extreme Learning Machine-------------------------------------------------------------------------

if visualisasi == "Dataset ISPU":
    st.title("Visualisasi Dataset ISPU")
    # Load the Excel file
    def load_excel():
        file_path = "DATA/Imputer/DATA ISPU - Impute.xlsx"
        xls = pd.ExcelFile(file_path)
        sheets = xls.sheet_names
        data = {}
        data_mapping = {
            'DKI1': 'Bunderan HI',
            'DKI2': 'Kelapa Gading',
            'DKI3': 'Jagakarsa',
            'DKI4': 'Lubang Buaya',
            'DKI5': 'Kebon Jeruk'
        }
        for sheet in sheets:
            location = data_mapping.get(sheet)
            if location:
                df = xls.parse(sheet)
                df['Location'] = location
                data[sheet] = df
        return data

    # Geocode function to get latitude and longitude
    def geocode_location(location):
        geolocator = Nominatim(user_agent="my_geocoder", timeout=10)
        location_data = geolocator.geocode(location)
        return location_data.latitude, location_data.longitude

    # Filter data by selected location, date, and nextday
    def filter_data(data, selected_location, selected_date, selected_nextday):
        filtered_data = {}
        for sheet, df in data.items():
            if selected_location == 'Semua Lokasi' or selected_location == sheet:
                selected_date = pd.to_datetime(selected_date)  # Convert selected_date to pandas datetime object
                next_dates = [selected_date + timedelta(days=i) for i in range(selected_nextday)]  # Get the next 'selected_nextday' dates
                
                # Filter the data for selected_date and next_dates
                filtered_df = df[df['Tanggal'].isin(next_dates)]
                filtered_data[sheet] = filtered_df
        return filtered_data

    # Main function
    def main():
        
        # Load data from Excel
        data = load_excel()
        
        # Get list of dates
        dates = list(data['DKI1']['Tanggal'].unique())
        
        # Sidebar location selection
        locations = ['Semua Lokasi'] + list(data.keys())
        selected_location = st.sidebar.selectbox("Pilih Lokasi", locations)
        
        # Sidebar date selection
        selected_date = st.sidebar.date_input("Pilih Tanggal", dates[0], min_value=dates[0], max_value=dates[-1])

        # Sidebar nextday selection
        selected_nextday = st.sidebar.selectbox("Pilih Jumlah Hari Yang Ditampilkan", [1, 2, 3, 4])
        
        # Filter data by selected location, date, and nextday
        filtered_data = filter_data(data, selected_location, selected_date.strftime("%Y-%m-%d"), selected_nextday)
        st.sidebar.markdown(
        """
            Keterangan:\n
            DKI1 :  Bundaran HI, DKI Jakarta\n
            DKI2 :  Kelapa Gading, DKI Jakarta\n
            DKI3 :  Jagakarsa, DKI Jakarta\n
            DKI4 :  Lubang Buaya, DKI Jakarta\n
            DKI5 :  Kebon Jeruk, DKI Jakarta\n
        """
        )
        tab1, tab2, tab3, tab4, tab5, tab6   = st.tabs(["Maps Visualisasi","DKI1 Bunderan HI","DKI2 Kelapa Gading", "DKI3 Jagakarsa", "DKI4 Lubang Buaya", "DKI5 Kebon Jeruk"])
        with tab1:
            # Get location coordinates
            if selected_location != 'Semua Lokasi':
                location = data[selected_location]['Location'].iloc[0]
                location_lat, location_lon = geocode_location(location)
                map_center = [location_lat, location_lon]
                zoom = 12
            else:
                map_center = [-6.200000, 106.816666]  # Jakarta coordinates
                zoom = 10
            
            # Create map
            # Create map
            m = folium.Map(location=map_center, zoom_start=zoom)

            # Lists to store markers and popups
            markers = []
            popups = []

            # Define colors for each location
            location_colors = {
                'DKI1': 'darkblue',
                'DKI2': 'red',
                'DKI3': 'purple',
                'DKI4': 'green',
                'DKI5': 'orange'
            }
            # Define tooltip for each location
            location_tooltip = {
                'DKI1': "Bunderan HI",
                'DKI2': "Kelapa Gading",
                'DKI3': "Jagakarsa",
                'DKI4': "Lubang Buaya",
                'DKI5': "Kebon Jeruk"
            }
            # Add markers and popups to the lists
            for sheet, df in filtered_data.items():
                location = df['Location'].iloc[0]
                location_lat, location_lon = geocode_location(location)
                marker = folium.Marker(
                    location=[location_lat, location_lon],
                    tooltip=location_tooltip[sheet],
                    icon=folium.Icon(color=location_colors[sheet]),
                )
                markers.append(marker)

                # Popup text for markers
                popup_text = f"<h2 style='font-size: 20px; font-weight:bold; color:black; text-align:center;'>{location}</h2>"
                for _, next_date_row in df.iterrows():
                    next_date = next_date_row['Tanggal']
                    next_date_PM10 = next_date_row['PM10']
                    next_date_SO2 = next_date_row['SO2']
                    next_date_CO = next_date_row['CO']
                    next_date_O3 = next_date_row['O3']
                    next_date_NO2 = next_date_row['NO2']
                    next_date_Kategori = next_date_row['Kategori']

                    # Set the color based on the category
                    if next_date_Kategori == 'BAIK':
                        popup_color = 'green'
                    elif next_date_Kategori == 'SEDANG':
                        popup_color = 'blue'
                    elif next_date_Kategori == 'TIDAK SEHAT':
                        popup_color = 'orange'
                    elif next_date_Kategori == 'SANGAT TIDAK SEHAT':
                        popup_color = 'red'
                    elif next_date_Kategori == 'BERBAHAYA':
                        popup_color = 'black'
                    else:
                        popup_color = 'gray'  # Default color if the category is not recognized

                    popup_text += f"<h2 style='font-size: 16px; font-weight:bold; color:white; text-align:center; background-color:{popup_color};'>{next_date_Kategori}</h2>"
                    popup_text += f"====== Tanggal {next_date.strftime('%Y-%m-%d')} ======<br>"
                    popup_text += f"PM10: {next_date_PM10}<br>"
                    popup_text += f"SO2: {next_date_SO2}<br>"
                    popup_text += f"CO: {next_date_CO}<br>"
                    popup_text += f"O3: {next_date_O3}<br>"
                    popup_text += f"NO2: {next_date_NO2}<br>"
                    popup_text += f"Kualitas Udara: {next_date_Kategori}<br>"

                popup = folium.Popup(popup_text, max_width=500)
                popups.append(popup)

            # Add markers and popups to the map
            for marker, popup in zip(markers, popups):
                marker.add_child(popup)
                marker.add_to(m)

            # Display the map
            folium_static(m)

            #keterangan Kualitas ISPU
            data = [
                {
                    'title': 'Baik',
                    'subtitle': 'Tingkat kualitas udara yang sangat baik, tidak memberikan efek  negatif terhadap manusia, hewan, tumbuhan.',
                    'action': 'Sangat baik melakukan kegiatan diluar.',
                    'action2': '',
                    'action3': '',
                    'color': 'green'
                },
                {
                    'title': 'Sedang',
                    'subtitle': 'Tingkat kualitas udara masih dapat diterima pada kesehatan manusia, hewan dan tumbuhan.',
                    'action': 'Kelompok Sensitif: Kurangi aktivitas fisik yang terlalu lama atau berat.',
                    'action2': 'Setiap Orang: Masih dapat beraktivitas di luar.',
                    'action3': '',
                    'color': 'blue'
                },
                {
                    'title': 'Tidak Sehat',
                    'subtitle': 'Tingkat kualitas udara yang bersifat merugikan paga manusia, hewan, dan tumbuhan.',
                    'action': 'Kelompok Sensitif : Boleh melakukan aktivitas di luar, tetapi mengambil rehat lebih sering dan melakukan aktivitas ringan. Amati gejala berupa batuk atau nafas sesak.',
                    'action2': 'Penderita Asma : harus mengikuti petunjuk kesehatan untuk asma dan menyimpan obat asma.',
                    'action3': 'Setiap Orang : Mengurangi aktivitas fisik yang terlalu lama di luar ruangan.',
                    'color': 'orange'
                },
                {
                    'title': 'Sangat Tidak Sehat',
                    'subtitle': 'Tingkat kualitas udara yang dapat meningkatkan risiko kesehatan pada sejumlah segmen populasi yang terpapar.',
                    'action': 'Kelompok Sensitif : Hindari semua aktivitas di luar. Perbanyak aktivitas di dalam ruangan atau lakukan penjadwalan ulang pada waktu dengan kualitas udara yang baik.',
                    'action2': 'Setiap Orang : Hindari aktivitas fisik yang terlalu lama di luar ruangan, pertimbangkan untuk melakukan aktivitas di dalam ruangan.',
                    'action3': '',
                    'color': 'red'
                },
                {
                    'title': 'Berbahaya',
                    'subtitle': 'Tingkat kualitas udara yang dapat merugikan kesehatan serius pada populasi dan perlu penanganan cepat.',
                    'action': 'Kelompok Sensitif : Tetap di dalam ruangan dan hanya melakukan sedikit aktivitas',
                    'action2': 'Setiap Orang : Hindari semua aktivitas di luar',
                    'action3': '',
                    'color': 'black'
                }
            ]
            col1, col2, col3, col4, col5 = st.columns(5)

            for i, col in enumerate([col1, col2, col3, col4, col5]):
                with col:
                    st.write(f'<h2 style="font-size: 20px; text-align:center; background-color:{data[i]["color"]}">{data[i]["title"]}</h2>', unsafe_allow_html=True)
                    st.write(f'<h2 style="font-size: 12px; color:black; text-align:center; background-color:white;">{data[i]["subtitle"]}</h2>', unsafe_allow_html=True)
                    st.write('<h2 style="font-size: 16px; font-weight:bold; color:black; text-align:center; background-color:white;">Apa yang harus dilakukan:</h2>', unsafe_allow_html=True)
                    st.write(f'<h2 style="font-size: 12px; color:black; text-align:center; background-color:white;">{data[i]["action"]}</h2>', unsafe_allow_html=True)
                    st.write(f'<h2 style="font-size: 12px; color:black; text-align:center; background-color:white;">{data[i]["action2"]}</h2>', unsafe_allow_html=True)
                    st.write(f'<h2 style="font-size: 12px; color:black; text-align:center; background-color:white;">{data[i]["action3"]}</h2>', unsafe_allow_html=True)

        with tab2:
            st.subheader("DKI1 Bunderan HI")
            Dki1 = pd.read_excel("DATA/Imputer/DATA ISPU - Impute.xlsx", sheet_name="DKI1")
            st.write(Dki1)

        with tab3:
            st.subheader("DKI2 Kelapa Gading")
            Dki2 = pd.read_excel("DATA/Imputer/DATA ISPU - Impute.xlsx", sheet_name="DKI2")
            st.write(Dki2)

        with tab4:
            st.subheader("DKI3 Jagakarsa")
            Dki3 = pd.read_excel("DATA/Imputer/DATA ISPU - Impute.xlsx", sheet_name="DKI3")
            st.write(Dki3)
            
        with tab5:
            st.subheader("DKI4 Lubang Buaya")
            Dki4 = pd.read_excel("DATA/Imputer/DATA ISPU - Impute.xlsx", sheet_name="DKI4")
            st.write(Dki4)

        with tab6:
            st.subheader("DKI5 Kebon Jeruk")
            Dki5 = pd.read_excel("DATA/Imputer/DATA ISPU - Impute.xlsx", sheet_name="DKI5")
            st.write(Dki5)

    # Run the app
    if __name__ == "__main__":
        main()


#-------------------------------------------------------------Visualisasi Forecasting Metode Extreme Learning Machine-------------------------------------------------------------------------

if visualisasi == "Metode Extreme Learning Machine":

    import streamlit as st
    import pandas as pd
    
    st.title("Visualisasi Data ISPU Hasil Forecasting Dengan Metode Extreme Learning Machine")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["DKI1 Bunderan HI","DKI2 Kelapa Gading", "DKI3 Jagakarsa", "DKI4 Lubang Buaya", "DKI5 Kebon Jeruk", "Data Uji Website"])
    with tab1:
        st.subheader("DKI1 Bunderan HI")
        Dki1 = pd.read_excel("Web/files/ELM forecasting/FORECAST ELM.xlsx", sheet_name="DKI1")
        st.write(Dki1)
    with tab2:
        st.subheader("DKI2 Kelapa Gading")
        Dki2 = pd.read_excel("Web/files/ELM forecasting/FORECAST ELM.xlsx", sheet_name="DKI2")
        st.write(Dki2)
    with tab3:
        st.subheader("DKI3 Jagakarsa")
        Dki3 = pd.read_excel("Web/files/ELM forecasting/FORECAST ELM.xlsx", sheet_name="DKI3")
        st.write(Dki3)
    with tab4:
        st.subheader("DKI4 Lubang Buaya")
        Dki4 = pd.read_excel("Web/files/ELM forecasting/FORECAST ELM.xlsx", sheet_name="DKI4")
        st.write(Dki4)
    with tab5:
        st.subheader("DKI5 Kebon Jeruk")
        Dki5 = pd.read_excel("Web/files/ELM forecasting/FORECAST ELM.xlsx", sheet_name="DKI5")
        st.write(Dki5)
    with tab6:
        st.subheader("Data Uji Website")
        Dki5 = pd.read_excel("DATA/Preprocessing/balance data testing 20%.xlsx")
        st.write(Dki5)

    import numpy as np
    from numpy.linalg import pinv, inv
    import time

    class elm():
        def __init__(self, hidden_units, activation_function, x, y, C, elm_type, one_hot=True, random_type='normal'):
            self.hidden_units = hidden_units
            self.activation_function = activation_function
            self.random_type = random_type
            self.x = x
            self.y = y
            self.C = C
            self.class_num = np.unique(self.y).shape[0]
            self.beta = np.zeros((self.hidden_units, self.class_num))
            self.elm_type = elm_type
            self.one_hot = one_hot

            if elm_type == 'clf' and self.one_hot:
                self.one_hot_label = np.zeros((self.y.shape[0], self.class_num + 1))
                for i in range(self.y.shape[0]):
                    self.one_hot_label[i, int(self.y[i])] = 1

            if self.random_type == 'uniform':
                self.W = np.random.uniform(low=0, high=1, size=(self.hidden_units, self.x.shape[1]))
                self.b = np.random.uniform(low=0, high=1, size=(self.hidden_units, 1))
            if self.random_type == 'normal':
                self.W = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, self.x.shape[1]))
                self.b = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, 1))

        def __input2hidden(self, x):
            #x = np.array(x, dtype=np.float64)  # Mengubah tipe data x menjadi float64
            self.temH = np.dot(self.W, x.T) + self.b

            if self.activation_function == 'sigmoid':
                self.H = 1 / (1 + np.exp(-self.temH))

            if self.activation_function == 'relu':
                self.H = self.temH * (self.temH > 0)

            if self.activation_function == 'sin':
                self.H = np.sin(self.temH)

            if self.activation_function == 'tanh':
                self.H = (np.exp(self.temH) - np.exp(-self.temH)) / (np.exp(self.temH) + np.exp(-self.temH))

            if self.activation_function == 'leaky_relu':
                self.H = np.maximum(0, self.temH) + 0.1 * np.minimum(0, self.temH)

            return self.H

        def __hidden2output(self, H):
            self.output = np.dot(H.T, self.beta)
            return self.output

        def fit(self, algorithm):
            self.time1 = time.perf_counter()
            self.H = self.__input2hidden(self.x)
            if self.elm_type == 'clf':
                if self.one_hot:
                    self.y_temp = self.one_hot_label
                else:
                    self.y_temp = self.y
            if self.elm_type == 'reg':
                self.y_temp = self.y

            if algorithm == 'no_re':
                self.beta = np.dot(pinv(self.H.T), self.y_temp)

            if algorithm == 'solution1':
                self.tmp1 = inv(np.eye(self.H.shape[0]) / self.C + np.dot(self.H, self.H.T))
                self.tmp2 = np.dot(self.tmp1, self.H)
                self.beta = np.dot(self.tmp2, self.y_temp)

            if algorithm == 'solution2':
                self.tmp1 = inv(np.eye(self.H.shape[0]) / self.C + np.dot(self.H, self.H.T))
                self.tmp2 = np.dot(self.H.T, self.tmp1)
                self.beta = np.dot(self.tmp2.T, self.y_temp)
            self.time2 = time.perf_counter()

            self.result = self.__hidden2output(self.H)

            if self.elm_type == 'clf':
                self.result = np.exp(self.result) / np.sum(np.exp(self.result), axis=1).reshape(-1, 1)

            if self.elm_type == 'clf':
                self.y_ = np.where(self.result == np.max(self.result, axis=1).reshape(-1, 1))[1]
                self.correct = 0
                for i in range(self.y.shape[0]):
                    if self.y_[i] == self.y[i]:
                        self.correct += 1
                self.train_score = self.correct / self.y.shape[0]
            if self.elm_type == 'reg':
                self.train_score = np.sqrt(np.sum((self.result - self.y) * (self.result - self.y)) / self.y.shape[0])
            train_time = str(self.time2 - self.time1)
            return self.beta, self.train_score, train_time

        def predict(self, x):
            #x = np.array(x, dtype=np.float64)  # Mengubah tipe data x menjadi float64
            self.H = self.__input2hidden(x)
            self.y_ = self.__hidden2output(self.H)
            if self.elm_type == 'clf':
                self.y_ = np.where(self.y_ == np.max(self.y_, axis=1).reshape(-1, 1))[1]

            return self.y_

        def predict_proba(self, x):
            #x = np.array(x, dtype=np.float64)  # Mengubah tipe data x menjadi float64
            self.H = self.__input2hidden(x)
            self.y_ = self.__hidden2output(self.H)
            if self.elm_type == 'clf':
                self.proba = np.exp(self.y_) / np.sum(np.exp(self.y_), axis=1).reshape(-1, 1)
            return self.proba

        def score(self, x, y):
            self.prediction = self.predict(x)
            if self.elm_type == 'clf':
                self.correct = 0
                for i in range(y.shape[0]):
                    if self.prediction[i] == y[i]:
                        self.correct += 1
                self.test_score = self.correct / y.shape[0]
            if self.elm_type == 'reg':
                self.test_score = np.sqrt(np.sum((self.result - self.y) * (self.result - self.y)) / self.y.shape[0])
            return self.test_score

    import pickle
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # Load the ELM model from the .sav file
    with open('Web/ELM_M.sav', 'rb') as file:
        model_data = pickle.load(file)

    kelayakan_model = model_data

    # Function to make predictions using the loaded model
    def predict_air_quality(data):
        # Perform the prediction using the Extreme Learning Machine model
        air_prediction = kelayakan_model.predict(data)

        return air_prediction

    def main():
        st.title("Penilaian Kualitas Udara")
        
        # Membaca file Excel
        excel_file = "Web/files/ELM forecasting/FORECAST ELM.xlsx"
        xls = pd.ExcelFile(excel_file)
        
        # Mendapatkan daftar sheet dalam file Excel
        sheet_names = xls.sheet_names
        
        # Pemilihan lokasi di sidebar
        selected_sheet = st.sidebar.selectbox("Pilih Lokasi", sheet_names)
        
        # Membaca data dari sheet yang dipilih
        df = pd.read_excel(excel_file, sheet_name=selected_sheet)
        
        # Mendapatkan tanggal unik dari kolom Tanggal
        unique_dates = pd.to_datetime(df["Tanggal"]).dt.date.unique()
        min_date = min(unique_dates)
        max_date = max(unique_dates)
        
        # Menentukan nilai default berdasarkan batasan tanggal
        default_date = min_date
        
        # Pemilihan tanggal di sidebar
        selected_date = st.sidebar.date_input("Pilih Tanggal", value=default_date, min_value=min_date, max_value=max_date)
        
        # Filtering data berdasarkan tanggal
        selected_data = df[pd.to_datetime(df["Tanggal"]).dt.date == selected_date]

        if not selected_data.empty:
            st.subheader(selected_sheet)
            st.write(selected_data)
            
            # Mendapatkan nilai partikel berdasarkan tanggal yang dipilih
            selected_row = selected_data.iloc[0]  # Mengambil baris pertama
            
        # Mengisi nilai PM10, SO2, CO, O3, NO2 secara otomatis
        PM10 = st.number_input("Masukkan nilai PM10", value=int(selected_row["PM10"]))
        SO2 = st.number_input("Masukkan nilai SO2", value=int(selected_row["SO2"]))
        CO = st.number_input("Masukkan nilai CO", value=int(selected_row["CO"]))
        O3 = st.number_input("Masukkan nilai O3", value=int(selected_row["O3"]))
        NO2 = st.number_input("Masukkan nilai NO2", value=int(selected_row["NO2"]))

        if st.button("Proses"):
            # Periksa apakah ada masukan yang kosong atau tidak valid
            if any(pd.isnull([PM10, SO2, CO, O3, NO2])):
                st.error("Harap masukkan nilai numerik yang valid untuk semua input.")
            else:
                # Konversi data masukan pengguna menjadi array numpy dan ubah bentuknya
                input_data = np.array([PM10, SO2, CO, O3, NO2]).reshape(1, -1)

                # Memuat data pelatihan dari file Excel
                training_data = pd.read_excel('DATA/Preprocessing/balance data train 80%.xlsx')  

                # Ekstrak fitur-fitur masukan dari data pelatihan dan sesuaikan Standar Scaler
                X_train = training_data.drop('Kategori', axis=1)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)

                # Transformasikan data masukan pengguna menggunakan Standar Scaler yang telah di sesuaikan
                input_data_scaled = scaler.transform(input_data)

                # Lakukan prediksi menggunakan model Extreme Learning Machine
                air_prediction = predict_air_quality(input_data_scaled)
                
                # Hubungkan prediksi numerik dengan label yang sesuai
                if air_prediction == 1:
                    air_quality_label = "BAIK"
                elif air_prediction == 2:
                    air_quality_label = "SEDANG"
                elif air_prediction == 3:
                    air_quality_label = "TIDAK SEHAT"
                elif air_prediction == 4:
                    air_quality_label = "SANGAT TIDAK SEHAT"
                else:
                    air_quality_label = "Prediksi tidak valid"

                # Tampilkan hasil prediksi atau gunakan untuk proses lebih lanjut
                st.write("Prediksi Kualitas Udara: ", air_quality_label)
                st.write("Data Setelah Scaled:", input_data_scaled)

    # Menjalankan aplikasi Streamlit
    if __name__ == '__main__':
        main()
#-------------------------------------------------------------Visualisasi Forecasting Metode Kernel Extreme Learning Machine-------------------------------------------------------------------------

if visualisasi == "Metode Kernel Extreme Learning Machine":
    import streamlit as st
    import pandas as pd
    
    st.title("Visualisasi Data ISPU Hasil Forecasting Dengan Metode Kernel Extreme Learning Machine")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["DKI1 Bunderan HI","DKI2 Kelapa Gading", "DKI3 Jagakarsa", "DKI4 Lubang Buaya", "DKI5 Kebon Jeruk", "Data Uji Website"])
    with tab1:
        st.subheader("DKI1 Bunderan HI")
        Dki1 = pd.read_excel("Web/files/K- ELM forecasting/FORECAST KELM.xlsx", sheet_name="DKI1")
        st.write(Dki1)
    with tab2:
        st.subheader("DKI2 Kelapa Gading")
        Dki2 = pd.read_excel("Web/files/K- ELM forecasting/FORECAST KELM.xlsx", sheet_name="DKI2")
        st.write(Dki2)
    with tab3:
        st.subheader("DKI3 Jagakarsa")
        Dki3 = pd.read_excel("Web/files/K- ELM forecasting/FORECAST KELM.xlsx", sheet_name="DKI3")
        st.write(Dki3)
    with tab4:
        st.subheader("DKI4 Lubang Buaya")
        Dki4 = pd.read_excel("Web/files/K- ELM forecasting/FORECAST KELM.xlsx", sheet_name="DKI4")
        st.write(Dki4)
    with tab5:
        st.subheader("DKI5 Kebon Jeruk")
        Dki5 = pd.read_excel("Web/files/K- ELM forecasting/FORECAST KELM.xlsx", sheet_name="DKI5")
        st.write(Dki5)
    with tab6:
        st.subheader("Data Uji Website")
        Dki5 = pd.read_excel("DATA/Preprocessing/balance data testing 20%.xlsx")
        st.write(Dki5)

    import numpy as np
    from numpy.linalg import pinv, inv
    import time

    class elm():
        def __init__(self, hidden_units, activation_function, x, y, C, elm_type, one_hot=True, random_type='normal'):
            self.hidden_units = hidden_units
            self.activation_function = activation_function
            self.random_type = random_type
            self.x = x
            self.y = y
            self.C = C
            self.class_num = np.unique(self.y).shape[0]
            self.beta = np.zeros((self.hidden_units, self.class_num))
            self.elm_type = elm_type
            self.one_hot = one_hot

            if elm_type == 'clf' and self.one_hot:
                self.one_hot_label = np.zeros((self.y.shape[0], self.class_num + 1))
                for i in range(self.y.shape[0]):
                    self.one_hot_label[i, int(self.y[i])] = 1

            if self.random_type == 'uniform':
                self.W = np.random.uniform(low=0, high=1, size=(self.hidden_units, self.x.shape[1]))
                self.b = np.random.uniform(low=0, high=1, size=(self.hidden_units, 1))
            if self.random_type == 'normal':
                self.W = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, self.x.shape[1]))
                self.b = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, 1))

        def __input2hidden(self, x):
            #x = np.array(x, dtype=np.float64)  # Mengubah tipe data x menjadi float64
            self.temH = np.dot(self.W, x.T) + self.b

            if self.activation_function == 'sigmoid':
                self.H = 1 / (1 + np.exp(-self.temH))

            if self.activation_function == 'relu':
                self.H = self.temH * (self.temH > 0)

            if self.activation_function == 'sin':
                self.H = np.sin(self.temH)

            if self.activation_function == 'tanh':
                self.H = (np.exp(self.temH) - np.exp(-self.temH)) / (np.exp(self.temH) + np.exp(-self.temH))

            if self.activation_function == 'leaky_relu':
                self.H = np.maximum(0, self.temH) + 0.1 * np.minimum(0, self.temH)

            return self.H

        def __hidden2output(self, H):
            self.output = np.dot(H.T, self.beta)
            return self.output

        def fit(self, algorithm):
            self.time1 = time.perf_counter()
            self.H = self.__input2hidden(self.x)
            if self.elm_type == 'clf':
                if self.one_hot:
                    self.y_temp = self.one_hot_label
                else:
                    self.y_temp = self.y
            if self.elm_type == 'reg':
                self.y_temp = self.y

            if algorithm == 'no_re':
                self.beta = np.dot(pinv(self.H.T), self.y_temp)

            if algorithm == 'solution1':
                self.tmp1 = inv(np.eye(self.H.shape[0]) / self.C + np.dot(self.H, self.H.T))
                self.tmp2 = np.dot(self.tmp1, self.H)
                self.beta = np.dot(self.tmp2, self.y_temp)

            if algorithm == 'solution2':
                self.tmp1 = inv(np.eye(self.H.shape[0]) / self.C + np.dot(self.H, self.H.T))
                self.tmp2 = np.dot(self.H.T, self.tmp1)
                self.beta = np.dot(self.tmp2.T, self.y_temp)
            self.time2 = time.perf_counter()

            self.result = self.__hidden2output(self.H)

            if self.elm_type == 'clf':
                self.result = np.exp(self.result) / np.sum(np.exp(self.result), axis=1).reshape(-1, 1)

            if self.elm_type == 'clf':
                self.y_ = np.where(self.result == np.max(self.result, axis=1).reshape(-1, 1))[1]
                self.correct = 0
                for i in range(self.y.shape[0]):
                    if self.y_[i] == self.y[i]:
                        self.correct += 1
                self.train_score = self.correct / self.y.shape[0]
            if self.elm_type == 'reg':
                self.train_score = np.sqrt(np.sum((self.result - self.y) * (self.result - self.y)) / self.y.shape[0])
            train_time = str(self.time2 - self.time1)
            return self.beta, self.train_score, train_time

        def predict(self, x):
            #x = np.array(x, dtype=np.float64)  # Mengubah tipe data x menjadi float64
            self.H = self.__input2hidden(x)
            self.y_ = self.__hidden2output(self.H)
            if self.elm_type == 'clf':
                self.y_ = np.where(self.y_ == np.max(self.y_, axis=1).reshape(-1, 1))[1]

            return self.y_

        def predict_proba(self, x):
            #x = np.array(x, dtype=np.float64)  # Mengubah tipe data x menjadi float64
            self.H = self.__input2hidden(x)
            self.y_ = self.__hidden2output(self.H)
            if self.elm_type == 'clf':
                self.proba = np.exp(self.y_) / np.sum(np.exp(self.y_), axis=1).reshape(-1, 1)
            return self.proba

        def score(self, x, y):
            self.prediction = self.predict(x)
            if self.elm_type == 'clf':
                self.correct = 0
                for i in range(y.shape[0]):
                    if self.prediction[i] == y[i]:
                        self.correct += 1
                self.test_score = self.correct / y.shape[0]
            if self.elm_type == 'reg':
                self.test_score = np.sqrt(np.sum((self.result - self.y) * (self.result - self.y)) / self.y.shape[0])
            return self.test_score

    import pickle
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.preprocessing import StandardScaler

    # Load the ELM model from the .sav file
    with open('Web/ELM_M.sav', 'rb') as file:
        model_data = pickle.load(file)

    kelayakan_model = model_data

    # Function to make predictions using the loaded model
    def predict_air_quality(data):
        # Perform the prediction using the Extreme Learning Machine model
        air_prediction = kelayakan_model.predict(data)

        return air_prediction

    def main():
        st.title("Penilaian Kualitas Udara")
        
        # Membaca file Excel
        excel_file = "Web/files/K- ELM forecasting/FORECAST KELM.xlsx"
        xls = pd.ExcelFile(excel_file)
        
        # Mendapatkan daftar sheet dalam file Excel
        sheet_names = xls.sheet_names
        
        # Pemilihan lokasi di sidebar
        selected_sheet = st.sidebar.selectbox("Pilih Lokasi", sheet_names)
        
        # Membaca data dari sheet yang dipilih
        df = pd.read_excel(excel_file, sheet_name=selected_sheet)
        
        # Mendapatkan tanggal unik dari kolom Tanggal
        unique_dates = pd.to_datetime(df["Tanggal"]).dt.date.unique()
        min_date = min(unique_dates)
        max_date = max(unique_dates)
        
        # Menentukan nilai default berdasarkan batasan tanggal
        default_date = min_date
        
        # Pemilihan tanggal di sidebar
        selected_date = st.sidebar.date_input("Pilih Tanggal", value=default_date, min_value=min_date, max_value=max_date)
        
        # Filtering data berdasarkan tanggal
        selected_data = df[pd.to_datetime(df["Tanggal"]).dt.date == selected_date]

        if not selected_data.empty:
            st.subheader(selected_sheet)
            st.write(selected_data)
            
            # Mendapatkan nilai partikel berdasarkan tanggal yang dipilih
            selected_row = selected_data.iloc[0]  # Mengambil baris pertama
            
        # Mengisi nilai PM10, SO2, CO, O3, NO2 secara otomatis
        PM10 = st.number_input("Masukkan nilai PM10", value=int(selected_row["PM10"]))
        SO2 = st.number_input("Masukkan nilai SO2", value=int(selected_row["SO2"]))
        CO = st.number_input("Masukkan nilai CO", value=int(selected_row["CO"]))
        O3 = st.number_input("Masukkan nilai O3", value=int(selected_row["O3"]))
        NO2 = st.number_input("Masukkan nilai NO2", value=int(selected_row["NO2"]))

        if st.button("Proses"):
            # Periksa apakah ada masukan yang kosong atau tidak valid
            if any(pd.isnull([PM10, SO2, CO, O3, NO2])):
                st.error("Harap masukkan nilai numerik yang valid untuk semua input.")
            else:
                # Konversi data masukan pengguna menjadi array numpy dan ubah bentuknya
                input_data = np.array([PM10, SO2, CO, O3, NO2]).reshape(1, -1)

                # Memuat data pelatihan dari file Excel
                training_data = pd.read_excel('DATA/Preprocessing/balance data train 80%.xlsx')  

                # Ekstrak fitur-fitur masukan dari data pelatihan dan sesuaikan Standar Scaler
                X_train = training_data.drop('Kategori', axis=1)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)

                # Transformasikan data masukan pengguna menggunakan Standar Scaler yang telah di sesuaikan
                input_data_scaled = scaler.transform(input_data)

                # Lakukan prediksi menggunakan model Extreme Learning Machine
                air_prediction = predict_air_quality(input_data_scaled)
                
                # Hubungkan prediksi numerik dengan label yang sesuai
                if air_prediction == 1:
                    air_quality_label = "BAIK"
                elif air_prediction == 2:
                    air_quality_label = "SEDANG"
                elif air_prediction == 3:
                    air_quality_label = "TIDAK SEHAT"
                elif air_prediction == 4:
                    air_quality_label = "SANGAT TIDAK SEHAT"
                else:
                    air_quality_label = "Prediksi tidak valid"

                # Tampilkan hasil prediksi atau gunakan untuk proses lebih lanjut
                st.write("Prediksi Kualitas Udara: ", air_quality_label)
                st.write("Data Setelah Scaled:", input_data_scaled)

    # Menjalankan aplikasi Streamlit
    if __name__ == '__main__':
        main()
