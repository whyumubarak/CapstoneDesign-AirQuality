import streamlit as st
from PIL import Image
# from xarray import align
from pathlib import Path

st.set_page_config(
    page_title="Home Page",
    page_icon="🏠",
)

st.title("Selamat datang")

st.markdown(
    """
    <div style='text-align: justify;'>
    Aplikasi ini dibuat dengan menggunakan streamlit untuk melakukan visualisasi regresi dan klasifikasi Data ISPU.
    
    Data ISPU juga bisa diakses di [halaman resmi.](https://data.jakarta.go.id/dataset?q=ispu&sort=1)
    
    Data yang digunakan diambil dari 5 titik di daerah DKI Jakarta. Berikut adalah peta dari 5 titik pengambilan data ISPU.
    </div>
""", unsafe_allow_html=True
)

st.subheader('Peta DKI Jakarta')

colu1, colu2 = st.columns([3.7, 1.3])
with colu1:
    import folium
    from streamlit_folium import st_folium
    m = folium.Map(location=[-6.214508, 106.820635], zoom_start=11)
    icon1 = folium.Icon(color="darkblue")
    icon2 = folium.Icon(color="red")
    icon3 = folium.Icon(color="purple")
    icon4 = folium.Icon(color="green")
    icon5 = folium.Icon(color="orange")
    folium.Marker(
        [-6.195322117029913, 106.82310195807386], popup="UDARA : TIDAK SEHAT", 
        tooltip="Bundaran HI",
        icon=icon1
    ).add_to(m)
    folium.Marker(
        [-6.1528630832227655, 106.89343818072786], popup="UDARA : TIDAK SEHAT", 
        tooltip="Kelapa gading",
        icon=icon2
    ).add_to(m)
    folium.Marker(
        [-6.331951542862599, 106.81439116568198], popup="UDARA : TIDAK SEHAT", 
        tooltip="Jagakarsa",
        icon=icon3
    ).add_to(m)
    folium.Marker(
        [-6.291030482700883, 106.89962883811985], popup="UDARA : TIDAK SEHAT", 
        tooltip="Lubang Buaya",
        icon=icon4
    ).add_to(m)
    folium.Marker(
        [-6.195942, 106.773595], popup="UDARA : TIDAK SEHAT", 
        tooltip="Kebon Jeruk",
        icon=icon5
    ).add_to(m)
    
    # call to render Folium map in Streamlit
    st_data = st_folium(m, width=550, height=400)

with colu2:
   st.subheader('Keterangan Lokasi Tower')
   colo1, colo2 = st.columns([1.1, 3.9])
   with colo1:
      from PIL import Image
      #
      image = Image.open('D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/images/b.png')
      resized_image = image.resize((35, 35))
      st.image(resized_image)
      #
      image = Image.open('D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/images/m.png')
      resized_image = image.resize((35, 35))
      st.image(resized_image)
      #
      image = Image.open('D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/images/u.png')
      resized_image = image.resize((35, 35))
      st.image(resized_image)
      #
      image = Image.open('D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/images/h.png')
      resized_image = image.resize((35, 35))
      st.image(resized_image)
      #
      image = Image.open('D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/images/o.png')
      resized_image = image.resize((35, 35))
      st.image(resized_image)
   with colo2:
    with st.container():
            st.write("""
    <div style="font-size: 15px;">
        DKI 1 Bunderan HI<br><br>
        DKI 2 Kelapa Gading<br><br>
        DKI 3 Jagakarsa<br><br>
        DKI 4 Lubang Buaya<br><br>
        DKI 5 Kebon Jeruk<br>
    </div>
    """,
    unsafe_allow_html=True)
      
st.header("Apa itu ISPU?")

st.markdown(
    """
        <div style='text-align: justify;'>
        Menurut Peraturan Menteri Lingkungan Hidup dan Kehutanan Nomor P.14/Menlhk/Setjen/Kum.1/7/2020 tentang Indeks Standar Pencemar Udara, 
        <b>ISPU</b> merupakan angka tanpa satuan yang menggambarkan kondisi kualitas udara ambien di lokasi tertentu, yang didasarkan pada dampak terhadap 
        kesehatan manusia, nilai estetik, dan makhluk hidup lainnya. Adapun parameter ISPU meliputi Hidrokarbon (HC), Karbon monoksida (CO), Sulfur dioksida (SO2), 
        Nitrogen dioksida (NO2), Ozon (O3), dan Partikulat (PM10 dan PM2,5).
        </div>
    """, unsafe_allow_html=True
)

# tabel_kategori_ispu = Image.open('images/Tabel_kategori_indeks_ISPU.png')
tabel_kategori_ispu = Image.open(Path(__file__).parents[1] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/images/Tabel_kategori_indeks_ISPU.png')

col1, col2, col3 = st.columns([1,6,1])

with col1:
    st.write("")

with col2:
    st.image(tabel_kategori_ispu, caption='Tabel Kategori Indeks ISPU')

with col3:
    st.write("")

st.markdown(
    """
        <div style='text-align: justify;'>
        Hasil perhitungan ISPU parameter PM2.5 disampaikan kepada publik tiap jam selama 24 jam. Sedangkan hasil perhitungan ISPU parameter PM10, NO2, SO2, CO, O3, 
        dan HC disampaikan kepada publik paling sedikit 2 (dua) kali dalam 1 (satu) hari pada pukul 09.00 dan 15.00. Tabel konversi nilai konsentrasi parameter ISPU 
        dan cara perhitungan sebagai berikut:
        </div>
    """, unsafe_allow_html=True
)

# konversi_nilai_konsentrasi = Image.open('./images/Konversi_nilai_konsentrasi.png')
konversi_nilai_konsentrasi = Image.open(Path(__file__).parents[1] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/images/Konversi_nilai_konsentrasi.png')

col1, col2, col3 = st.columns([1,6,1])

with col1:
    st.write("")

with col2:
    st.image(konversi_nilai_konsentrasi, caption='Konversi Nilai Konsentrasi')

with col3:
    st.write("")

st.markdown(
    """
        <div style='text-align: justify;'>
        Perhitungan ISPU dilakukan berdasarkan nilai ISPU batas atas, ISPU batas bawah, ambien batas atas, ambien batas bawah, dan konsentrasi ambien hasil pengukuran. 
        Persamaan matematika perhitungan ISPU sebagai berikut:
        </div>
    """, unsafe_allow_html=True
)

# rumus_ISPU = Image.open('./images/rumus_ISPU.png')
rumus_ISPU = Image.open(Path(__file__).parents[1] / 'D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/images/rumus_ISPU.png')

col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.write("")

with col2:
    st.image(rumus_ISPU)

with col3:
    st.write("")

st.markdown(
    """
        Keterangan:

        I = ISPU terhitung
        
        Ia = ISPU batas atas

        Ib = ISPU batas bawah

        Xa = Konsentrasi ambien batas atas (µg/m3)

        Xb = Konsentrasi ambien batas bawah (µg/m3)

        Xx = Konsentrasi ambien nyata hasil pengukuran (µg/m3)
    """
)
