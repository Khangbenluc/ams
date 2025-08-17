# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import pytz
from paddleocr import PaddleOCR
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import re
import os
import matplotlib.pyplot as plt
import io

# --- Khởi tạo OCR ---
@st.cache_resource
def get_reader():
    """Khởi tạo PaddleOCR một lần và lưu vào cache."""
    return PaddleOCR(lang="vi", use_angle_cls=False)

ocr = get_reader()

# --- Kết nối SQLite ---
conn = sqlite3.connect("lich_su_giao_dich.db", check_same_thread=False)
c = conn.cursor()

# Tạo bảng lịch sử giao dịch nếu chưa tồn tại
c.execute('''
CREATE TABLE IF NOT EXISTS lich_su (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thoi_gian TEXT,
    ho_va_ten TEXT,
    so_cccd TEXT,
    que_quan TEXT,
    khoi_luong REAL,
    don_gia REAL,
    thanh_tien REAL
)
''')
conn.commit()

# --- Chuyển số sang chữ ---
def doc_so_thanh_chu(number):
    if not isinstance(number, (int, float)) or number < 0:
        return "Số không hợp lệ"
    number = int(number)
    
    chu_so = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
    don_vi = ["", "nghìn", "triệu", "tỷ", "nghìn tỷ", "triệu tỷ", "tỷ tỷ"]
    
    def doc_ba_so(so):
        if so == 0: return ""
        tram = so // 100
        chuc = (so % 100) // 10
        don_vi_le = so % 10
        
        chuoi = ""
        if tram > 0:
            chuoi += chu_so[tram] + " trăm "
        
        if chuc == 0 and don_vi_le > 0 and tram > 0:
            chuoi += "linh "
        elif chuc == 1:
            chuoi += "mười "
        elif chuc > 1:
            chuoi += chu_so[chuc] + " mươi "
        
        if don_vi_le > 0:
            chuoi += chu_so[don_vi_le]
        
        return chuoi.strip()

    s = str(number)
    parts = []
    
    while len(s) > 0:
        if len(s) >= 3:
            part = s[-3:]
            s = s[:-3]
        else:
            part = s
            s = ""
        parts.insert(0, int(part))
    
    ket_qua = ""
    for i, p in enumerate(parts):
        if p != 0:
            ket_qua += doc_ba_so(p) + " " + don_vi[len(parts) - 1 - i] + " "
    
    return ket_qua.strip().capitalize() + " đồng"

# --- Tiền xử lý ảnh cho OCR ---
def preprocess_image(img_bytes):
    """Tiền xử lý ảnh để cải thiện chất lượng OCR."""
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    return equalized

# --- Hàm OCR CCCD ---
def trich_xuat_cccd(image_bytes):
    try:
        if image_bytes is None: return "", "", ""
        preprocessed_img = preprocess_image(image_bytes)
        result = ocr.ocr(preprocessed_img, cls=False)
        
        ho_ten, so_cccd, que_quan = "", "", ""
        if result and result[0]:
            for line in result[0]:
                text = line[1][0].upper()
                if "HỌ VÀ TÊN" in text:
                    idx = result[0].index(line)
                    if idx + 1 < len(result[0]):
                        ho_ten = result[0][idx + 1][1][0]
                elif "SỐ" in text and len(text.split()[-1]) == 12 and text.split()[-1].isdigit():
                    so_cccd = text.split()[-1]
                elif "QUÊ QUÁN" in text:
                    idx = result[0].index(line)
                    if idx + 1 < len(result[0]):
                        que_quan = result[0][idx + 1][1][0]
        return ho_ten, so_cccd, que_quan
    except Exception as e:
        st.error(f"Lỗi khi xử lý OCR: {e}")
        return "", "", ""

# --- Hàm OCR cân ---
def trich_xuat_can(image_bytes):
    try:
        if image_bytes is None: return ""
        preprocessed_img = preprocess_image(image_bytes)
        result = ocr.ocr(preprocessed_img, cls=False)
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                cleaned_text = ''.join(c for c in text if c.isdigit() or c == '.')
                if cleaned_text:
                    return cleaned_text
        return ""
    except Exception as e:
        st.error(f"Lỗi khi xử lý OCR cân: {e}")
        return ""

# --- Hàm tính tiền và lưu SQLite ---
def xu_ly_giao_dich(ho_va_ten, so_cccd, que_quan, so_luong_str, don_gia_str):
    try:
        so_luong = float(so_luong_str.replace(',', ''))
        don_gia = float(don_gia_str.replace(',', ''))
        thanh_tien = so_luong * don_gia

        vn_timezone = pytz.timezone('Asia/Ho_Chi_Minh')
        current_time = datetime.now(vn_timezone)
        thoi_gian_luu = current_time.strftime("%Y-%m-%d %H:%M:%S")

        c.execute('''
            INSERT INTO lich_su (thoi_gian, ho_va_ten, so_cccd, que_quan, khoi_luong, don_gia, thanh_tien)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (thoi_gian_luu, ho_va_ten, so_cccd, que_quan, so_luong, don_gia, thanh_tien))
        conn.commit()

        return {
            "ho_va_ten": ho_va_ten,
            "so_cccd": so_cccd,
            "que_quan": que_quan,
            "so_luong": so_luong,
            "don_gia": don_gia,
            "thanh_tien": thanh_tien,
            "ngay_tao": current_time.strftime("%d/%m/%Y")
        }
    except (ValueError, TypeError) as e:
        st.error(f"Lỗi: Dữ liệu nhập không hợp lệ. {e}")
        return None

# --- Hàm tạo PDF ---
try:
    pdfmetrics.registerFont(TTFont('TimesNewRoman', 'Times New Roman.ttf'))
except:
    st.warning("Không tìm thấy font 'Times New Roman.ttf'. PDF có thể hiển thị lỗi font.")
    pdfmetrics.registerFont(TTFont('Vera', 'Vera.ttf'))

def tao_pdf(data):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    pdf.setFont("TimesNewRoman", 12)

    pdf.drawCentredString(width/2, height-50, "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM")
    pdf.drawCentredString(width/2, height-70, "Độc lập - Tự do - Hạnh phúc")
    pdf.drawCentredString(width/2, height-90, "BẢNG KÊ THU MUA HÀNG HÓA, DỊCH VỤ MUA VÀO")
    
    pdf.drawString(50, height-130, f"Họ và tên: {data['ho_va_ten']}")
    pdf.drawString(50, height-150, f"Số CCCD: {data['so_cccd']}")
    pdf.drawString(50, height-170, f"Quê quán: {data['que_quan']}")
    
    y = height-210
    pdf.drawString(50, y, "STT")
    pdf.drawString(100, y, "Tên hàng/dịch vụ")
    pdf.drawString(300, y, "Số lượng")
    pdf.drawString(380, y, "Đơn giá")
    pdf.drawString(480, y, "Thành tiền")
    y -= 20
    pdf.drawString(50, y, "1")
    pdf.drawString(100, y, "Hàng hóa")
    pdf.drawString(300, y, str(data['so_luong']))
    pdf.drawString(380, y, f"{data['don_gia']:,.0f}")
    pdf.drawString(480, y, f"{data['thanh_tien']:,.0f}")

    y -= 40
    pdf.drawString(50, y, f"Tổng cộng: {data['thanh_tien']:,.0f} VNĐ")
    y -= 20
    pdf.drawString(50, y, f"Bằng chữ: {doc_so_thanh_chu(data['thanh_tien'])}")

    pdf.save()
    buffer.seek(0)
    return buffer

# --- Giao diện Streamlit ---
def main_app():
    st.set_page_config(layout="wide")
    st.title("ỨNG DỤNG TẠO BẢN KÊ MUA HÀNG - 01/TNDN")
    st.markdown("---")

    tab1, tab2 = st.tabs(["Tạo giao dịch", "Lịch sử & Thống kê"])

    with tab1:
        create_new_transaction_page()

    with tab2:
        history_and_stats_page()

def create_new_transaction_page():
    if 'ho_ten' not in st.session_state:
        st.session_state.ho_ten = ""
    if 'so_cccd' not in st.session_state:
        st.session_state.so_cccd = ""
    if 'que_quan' not in st.session_state:
        st.session_state.que_quan = ""
    if 'so_luong' not in st.session_state:
        st.session_state.so_luong = ""

    st.subheader("1. Trích xuất thông tin từ ảnh 🖼️")
    col_cccd, col_can = st.columns(2)

    with col_cccd:
        st.subheader("Chụp ảnh hoặc tải ảnh CCCD")
        anh_cccd = st.camera_input("Chụp ảnh CCCD")
        uploaded_cccd = st.file_uploader("Hoặc tải ảnh CCCD", type=["jpg", "jpeg", "png"], key="cccd_uploader")
        
        if anh_cccd:
            with st.spinner('Đang xử lý OCR...'):
                ho_ten, so_cccd, que_quan = trich_xuat_cccd(anh_cccd.read())
                st.session_state.ho_ten = ho_ten
                st.session_state.so_cccd = so_cccd
                st.session_state.que_quan = que_quan
            st.success("Trích xuất thành công!")
        elif uploaded_cccd:
            with st.spinner('Đang xử lý OCR...'):
                ho_ten, so_cccd, que_quan = trich_xuat_cccd(uploaded_cccd.read())
                st.session_state.ho_ten = ho_ten
                st.session_state.so_cccd = so_cccd
                st.session_state.que_quan = que_quan
            st.success("Trích xuất thành công!")
    
    with col_can:
        st.subheader("Chụp ảnh hoặc tải ảnh cân")
        anh_can = st.camera_input("Chụp ảnh màn hình cân")
        uploaded_can = st.file_uploader("Hoặc tải ảnh cân", type=["jpg", "jpeg", "png"], key="can_uploader")
        
        if anh_can:
            with st.spinner('Đang xử lý OCR...'):
                so_luong = trich_xuat_can(anh_can.read())
                st.session_state.so_luong = so_luong
            st.success("Trích xuất thành công!")
        elif uploaded_can:
            with st.spinner('Đang xử lý OCR...'):
                so_luong = trich_xuat_can(uploaded_can.read())
                st.session_state.so_luong = so_luong
            st.success("Trích xuất thành công!")

    st.markdown("---")

    st.subheader("2. Tạo bản kê và lưu giao dịch 📝")
    with st.form("form_giao_dich"):
        ho_ten_input = st.text_input("Họ và Tên", value=st.session_state.ho_ten)
        so_cccd_input = st.text_input("Số Căn cước công dân", value=st.session_state.so_cccd)
        que_quan_input = st.text_input("Quê quán", value=st.session_state.que_quan)
        so_luong_input = st.text_input("Khối lượng (chỉ)", value=st.session_state.so_luong)
        don_gia_input = st.text_input("Đơn giá (VNĐ/chỉ)")
        
        col_submit, col_download = st.columns(2)
        with col_submit:
            submitted = st.form_submit_button("Lưu giao dịch")
        
        if submitted:
            if not ho_ten_input or not so_luong_input or not don_gia_input:
                st.error("Vui lòng nhập đầy đủ thông tin.")
            else:
                giao_dich_data = xu_ly_giao_dich(ho_ten_input, so_cccd_input, que_quan_input, so_luong_input, don_gia_input)
                if giao_dich_data:
                    st.success("Giao dịch đã được lưu thành công!")
                    st.session_state['last_giao_dich'] = giao_dich_data
                    
                    st.metric(label="Thành Tiền", value=f"{giao_dich_data['thanh_tien']:,.0f} VNĐ")
                    st.write(f"Bằng chữ: {doc_so_thanh_chu(giao_dich_data['thanh_tien'])}")
                    
                    pdf_bytes = tao_pdf(giao_dich_data)
                    with col_download:
                        st.download_button(
                            "Tải bản kê PDF",
                            data=pdf_bytes,
                            file_name=f"bang_ke_{giao_dich_data['ho_va_ten']}.pdf",
                            mime="application/pdf"
                        )
    st.markdown("---")
    if st.button("Làm mới trang"):
        st.experimental_rerun()

def history_and_stats_page():
    st.header("Lịch sử và Thống kê 📈")
    
    df = pd.read_sql_query("SELECT * FROM lich_su ORDER BY thoi_gian DESC", conn)
    
    if df.empty:
        st.info("Chưa có giao dịch nào được ghi lại.")
        return

    # --- Bộ lọc ---
    st.subheader("Bộ lọc")
    col1, col2 = st.columns(2)
    with col1:
        ho_ten_search = st.text_input("Tìm kiếm theo tên khách hàng")
    with col2:
        cccd_search = st.text_input("Tìm kiếm theo CCCD")
        
    df_filtered = df.copy()
    if ho_ten_search:
        df_filtered = df_filtered[df_filtered['ho_va_ten'].str.contains(ho_ten_search, case=False, na=False)]
    if cccd_search:
        df_filtered = df_filtered[df_filtered['so_cccd'].str.contains(cccd_search, case=False, na=False)]

    st.markdown("---")

    # --- Thống kê ---
    st.subheader("Thống kê")
    
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    with col_stats1:
        tong_giao_dich = len(df_filtered)
        st.metric("Tổng giao dịch", value=f"{tong_giao_dich}")
    with col_stats2:
        tong_thanh_tien = df_filtered['thanh_tien'].sum()
        st.metric("Tổng thành tiền", value=f"{tong_thanh_tien:,.0f} VNĐ")
    with col_stats3:
        tong_khoi_luong = df_filtered['khoi_luong'].sum()
        st.metric("Tổng khối lượng", value=f"{tong_khoi_luong} chỉ")

    st.markdown("---")
    
    # --- Biểu đồ ---
    st.subheader("Biểu đồ doanh thu")
    df_filtered['thoi_gian'] = pd.to_datetime(df_filtered['thoi_gian'])
    df_filtered['ngay'] = df_filtered['thoi_gian'].dt.date
    daily_revenue = df_filtered.groupby('ngay')['thanh_tien'].sum()
    
    fig, ax = plt.subplots()
    ax.bar(daily_revenue.index.astype(str), daily_revenue.values)
    ax.set_title("Doanh thu hàng ngày")
    ax.set_ylabel("Thành tiền (VNĐ)")
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")

    # --- Lịch sử giao dịch chi tiết ---
    st.subheader("Lịch sử giao dịch")
    st.dataframe(df_filtered)
    
    # Nút tải xuống CSV
    csv_file = df_filtered.to_csv(index=False)
    st.download_button(
        label="Tải xuống CSV",
        data=csv_file,
        file_name='lich_su_giao_dich.csv',
        mime='text/csv'
    )

# --- Chạy ứng dụng ---
if __name__ == "__main__":
    main_app()
