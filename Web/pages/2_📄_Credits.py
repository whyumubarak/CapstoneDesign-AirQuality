import streamlit as st
# from PIL import Image
# from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="Credits",
    page_icon="📄",
)

st.subheader("Website Projek TA")
st.markdown(
    """
    <style>
    .center {
        font-size: 20px;
        display: flex;
        justify-content: center;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<h1 class="center">PEMBUATAN SISTEM KLASIFIKASI DAN PREDIKSI KUALITAS UDARA MENGGUNAKAN METODE ELM & K-ELM BERDASARKAN JAKARTA OPEN DATA (INDEKS STANDAR PENCEMARAN UDARA)</h1>', unsafe_allow_html=True)
st.write("dibuat oleh:")

col1, col2, col3 = st.columns(3)
with col1:
    image = Image.open('D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/images/x.jpg')
    scale_factor = 0.5 
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    resized_image = image.resize((new_width, new_height))
    st.image(resized_image, caption='WAHYU MUBARAK SUKIMAN')
with col2:
    image = Image.open('D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/images/z.jpg')
    scale_factor = 0.5 
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    resized_image = image.resize((new_width, new_height))
    st.image(resized_image, caption='AGUNG SULAKSONO RAMDHANI')
with col3:
    image = Image.open('D:/Data/Kuliah/TA/Projek-Capstone-Design/Web/images/y.jpg')
    scale_factor = 0.5 
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    resized_image = image.resize((new_width, new_height))
    st.image(resized_image, caption='SULTAN CHISSON OBIE')


st.markdown(
    """
        ### Terima kasih untuk dosen pembimbing yang telah membimbing dan memberikan saran atas projek TA ini:\n
        > Dr. Meta Kallista, S.Si., M.Si.\n
        > IG. Prasetya Dwi Wibawa, S.T., M.T.\n
        > Umar Ali Ahmad, Ph.D.\n\n
    """
)
