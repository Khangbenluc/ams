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
from reportlab.lib.units import mm
import re
import os
import matplotlib.pyplot as plt

# --- Quản lý người dùng (đơn giản, dùng cho demo) ---
users = {
    "admin": "admin123",
    "user1": "user123"
}

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
        
        if don_vi_le == 5 and chuc != 0:
            chuoi += "lăm"
        elif don_vi_le == 1 and chuc != 0 and chuc != 1:
            chuoi += "mốt"
        elif don_vi_le > 0:
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
        ho_ten, so_cccd, que_quan = "", "", ""
        if image_bytes is None: 
            return ho_ten, so_cccd, que_quan
        
        preprocessed_img = preprocess_image(image_bytes)
        result = ocr.ocr(preprocessed_img) 
        
        if result and result[0]:
            for idx, line in enumerate(result[0]):
                text = line[1][0].upper()
                # Thêm kiểm tra 'and idx + 1 < len(result[0])' để đảm bảo index không bị lỗi
                if "HỌ VÀ TÊN" in text and idx + 1 < len(result[0]):
                    ho_ten = result[0][idx + 1][1][0]
                elif "SỐ" in text and len(text.split()[-1]) == 12 and text.split()[-1].isdigit():
                    so_cccd = text.split()[-1]
                elif "QUÊ QUÁN" in text and idx + 1 < len(result[0]):
                    que_quan = result[0][idx + 1][1][0]
        return ho_ten, so_cccd, que_quan
    except Exception as e:
        st.error(f"Lỗi khi xử lý OCR: {e}")
        return "", "", ""

# --- Hàm OCR cân ---
def trich_xuat_can(image_bytes):
    try:
        if image_bytes is None: 
            return ""
        
        preprocessed_img = preprocess_image(image_bytes)
        result = ocr.ocr(preprocessed_img)
        
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

# --- Hàm tạo PDF theo mẫu 01/TNDN ---
# Cố gắng đăng ký font Arial, nếu không được thì dùng font mặc định
FONT_FILE = "Arial.ttf"
FONT_NAME = "Arial"
try:
    if os.path.exists(FONT_FILE):
        pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_FILE))
    else:
        st.warning(f"Không tìm thấy font '{FONT_FILE}'. Vui lòng đặt font vào cùng thư mục với file app.")
except Exception as e:
    st.error(f"Lỗi khi đăng ký font: {e}")
    st.warning("Ứng dụng sẽ sử dụng font mặc định, có thể không hiển thị được tiếng Việt.")
    FONT_NAME = "Helvetica" # Font mặc định của reportlab

def tao_pdf_mau_01(data, ten_don_vi=""):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Đảm bảo font được sử dụng đã được đăng ký
    pdf.setFont(FONT_NAME, 12)

    # Tiêu đề
    if ten_don_vi:
        pdf.drawString(20*mm, height - 15*mm, ten_don_vi.upper())
    pdf.drawCentredString(width/2, height - 20*mm, "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM")
    pdf.drawCentredString(width/2, height - 25*mm, "Độc lập - Tự do - Hạnh phúc")
    pdf.drawCentredString(width/2, height - 30*mm, "--------------------------")
    pdf.drawRightString(width - 20*mm, height - 35*mm, "Mẫu số: 01/TNDN")
    
    pdf.setFont(FONT_NAME, 14)
    pdf.drawCentredString(width/2, height - 50*mm, "BẢNG KÊ THU MUA HÀNG HÓA, DỊCH VỤ")
    pdf.drawCentredString(width/2, height - 55*mm, "KHÔNG CÓ HÓA ĐƠN")
    pdf.setFont(FONT_NAME, 12)

    # Thông tin chung
    pdf.drawString(20*mm, height - 70*mm, f"Họ và tên người bán: {data['ho_va_ten']}")
    pdf.drawString(20*mm, height - 75*mm, f"Số CCCD: {data['so_cccd']}")
    pdf.drawString(20*mm, height - 80*mm, f"Quê quán: {data['que_quan']}")
    pdf.drawString(20*mm, height - 85*mm, f"Ngày lập: {data['ngay_tao']}")

    # Bảng chi tiết
    y = height - 100*mm
    pdf.rect(20*mm, y-20*mm, 170*mm, 20*mm) # Khung bảng
    
    # Header
    pdf.drawString(22*mm, y - 5*mm, "STT")
    pdf.drawString(35*mm, y - 5*mm, "Tên hàng hóa, dịch vụ")
    pdf.drawString(100*mm, y - 5*mm, "Đơn vị tính")
    pdf.drawString(120*mm, y - 5*mm, "Số lượng")
    pdf.drawString(140*mm, y - 5*mm, "Đơn giá")
    pdf.drawString(170*mm, y - 5*mm, "Thành tiền")

    # Dòng dữ liệu
    pdf.drawString(22*mm, y - 15*mm, "1")
    pdf.drawString(35*mm, y - 15*mm, "Hàng hóa")
    pdf.drawString(100*mm, y - 15*mm, "chỉ")
    pdf.drawString(120*mm, y - 15*mm, f"{data['so_luong']:,.2f}")
    pdf.drawString(140*mm, y - 15*mm, f"{data['don_gia']:,.0f}")
    pdf.drawString(170*mm, y - 15*mm, f"{data['thanh_tien']:,.0f}")

    # Tổng cộng
    y -= 30*mm
    pdf.drawString(20*mm, y, f"Tổng cộng: {data['thanh_tien']:,.0f} VNĐ")
    y -= 5*mm
    pdf.drawString(20*mm, y, f"Bằng chữ: {doc_so_thanh_chu(data['thanh_tien'])}")
    
    pdf.save()
    buffer.seek(0)
    return buffer

# --- Giao diện Streamlit ---
def login_page():
    st.title("Đăng nhập/Đăng ký")
    menu = ["Đăng nhập", "Đăng ký"]
    choice = st.selectbox("Chọn", menu)

    if choice == "Đăng nhập":
        st.subheader("Đăng nhập")
        username = st.text_input("Tên đăng nhập")
        password = st.text_input("Mật khẩu", type="password")
        if st.button("Đăng nhập"):
            if username in users and users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Chào mừng, {username}!")
                st.rerun()
            else:
                st.error("Tên đăng nhập hoặc mật khẩu không đúng.")

    elif choice == "Đăng ký":
        st.subheader("Đăng ký tài khoản mới")
        new_user = st.text_input("Tên đăng nhập mới")
        new_password = st.text_input("Mật khẩu mới", type="password")
        if st.button("Đăng ký"):
            if new_user in users:
                st.warning("Tên đăng nhập đã tồn tại.")
            else:
                users[new_user] = new_password
                st.success("Đăng ký thành công! Vui lòng đăng nhập.")
                st.balloons()

def main_app():
    st.set_page_config(layout="wide")
    st.title("ỨNG DỤNG TẠO BẢN KÊ MUA HÀNG - 01/TNDN")
    st.markdown("---")

    if st.button("Đăng xuất"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

    tab1, tab2 = st.tabs(["Tạo giao dịch", "Lịch sử & Thống kê"])

    with tab1:
        create_new_transaction_page()

    with tab2:
        history_and_stats_page()

def create_new_transaction_page():
    # Khởi tạo các giá trị trong session_state để lưu trữ trạng thái của form
    if 'ho_ten' not in st.session_state:
        st.session_state.ho_ten = ""
    if 'so_cccd' not in st.session_state:
        st.session_state.so_cccd = ""
    if 'que_quan' not in st.session_state:
        st.session_state.que_quan = ""
    if 'so_luong' not in st.session_state:
        st.session_state.so_luong = ""
    if 'pdf_for_download' not in st.session_state:
        st.session_state.pdf_for_download = None

    st.subheader("1. Trích xuất thông tin từ ảnh 🖼️")
    col_cccd, col_can = st.columns(2)

    with col_cccd:
        st.subheader("Chụp ảnh hoặc tải ảnh CCCD")
        anh_cccd = st.camera_input("Chụp ảnh CCCD")
        uploaded_cccd = st.file_uploader("Hoặc tải ảnh CCCD", type=["jpg", "jpeg", "png"], key="cccd_uploader")
        
        # Logic trích xuất và lưu vào session state
        if anh_cccd:
            with st.spinner('Đang xử lý OCR...'):
                ho_ten, so_cccd, que_quan = trich_xuat_cccd(anh_cccd.read())
                # Cập nhật session_state để điền vào các ô nhập liệu
                st.session_state.ho_ten = ho_ten
                st.session_state.so_cccd = so_cccd
                st.session_state.que_quan = que_quan
            st.success("Trích xuất thành công! Dữ liệu đã được điền vào form.")
            st.rerun() # Thêm lệnh này để làm mới giao diện ngay lập tức
        elif uploaded_cccd:
            with st.spinner('Đang xử lý OCR...'):
                ho_ten, so_cccd, que_quan = trich_xuat_cccd(uploaded_cccd.read())
                # Cập nhật session_state để điền vào các ô nhập liệu
                st.session_state.ho_ten = ho_ten
                st.session_state.so_cccd = so_cccd
                st.session_state.que_quan = que_quan
            st.success("Trích xuất thành công! Dữ liệu đã được điền vào form.")
            st.rerun() # Thêm lệnh này để làm mới giao diện ngay lập tức
    
    with col_can:
        st.subheader("Chụp ảnh hoặc tải ảnh cân")
        anh_can = st.camera_input("Chụp ảnh màn hình cân")
        uploaded_can = st.file_uploader("Hoặc tải ảnh cân", type=["jpg", "jpeg", "png"], key="can_uploader")
        
        # Logic trích xuất và lưu vào session state
        if anh_can:
            with st.spinner('Đang xử lý OCR...'):
                so_luong = trich_xuat_can(anh_can.read())
                # Cập nhật session_state để điền vào ô nhập liệu
                st.session_state.so_luong = so_luong
            st.success("Trích xuất thành công! Khối lượng đã được điền vào form.")
            st.rerun() # Thêm lệnh này để làm mới giao diện ngay lập tức
        elif uploaded_can:
            with st.spinner('Đang xử lý OCR...'):
                so_luong = trich_xuat_can(uploaded_can.read())
                # Cập nhật session_state để điền vào ô nhập liệu
                st.session_state.so_luong = so_luong
            st.success("Trích xuất thành công! Khối lượng đã được điền vào form.")
            st.rerun() # Thêm lệnh này để làm mới giao diện ngay lập tức

    st.markdown("---")

    st.subheader("2. Tạo bản kê và lưu giao dịch 📝")
    ho_ten_input = st.text_input("Họ và Tên", value=st.session_state.ho_ten)
    so_cccd_input = st.text_input("Số Căn cước công dân", value=st.session_state.so_cccd)
    que_quan_input = st.text_input("Quê quán", value=st.session_state.que_quan)
    so_luong_input = st.text_input("Khối lượng (chỉ)", value=st.session_state.so_luong)
    don_gia_input = st.text_input("Đơn giá (VNĐ/chỉ)")
    ten_don_vi = st.text_input("Tên đơn vị (không bắt buộc)")

    if st.button("Lưu giao dịch"):
        if not ho_ten_input or not so_luong_input or not don_gia_input:
            st.error("Vui lòng nhập đầy đủ thông tin.")
        else:
            giao_dich_data = xu_ly_giao_dich(ho_ten_input, so_cccd_input, que_quan_input, so_luong_input, don_gia_input)
            if giao_dich_data:
                st.success("Giao dịch đã được lưu thành công!")
                
                st.metric(label="Thành Tiền", value=f"{giao_dich_data['thanh_tien']:,.0f} VNĐ")
                st.write(f"Bằng chữ: {doc_so_thanh_chu(giao_dich_data['thanh_tien'])}")
                
                pdf_bytes = tao_pdf_mau_01(giao_dich_data, ten_don_vi)
                st.session_state.pdf_for_download = pdf_bytes
                st.session_state.giao_dich_data = giao_dich_data
                
    if st.session_state.pdf_for_download:
        st.download_button(
            "Tải bản kê PDF (Mẫu 01/TNDN)",
            data=st.session_state.pdf_for_download,
            file_name=f"bang_ke_{st.session_state.giao_dich_data['ho_va_ten']}.pdf",
            mime="application/pdf"
        )

    st.markdown("---")
    if st.button("Làm mới trang"):
        st.session_state.ho_ten = ""
        st.session_state.so_cccd = ""
        st.session_state.que_quan = ""
        st.session_state.so_luong = ""
        st.session_state.pdf_for_download = None
        st.rerun()

def history_and_stats_page():
    st.header("Lịch sử và Thống kê 📈")
    
    df = pd.read_sql_query("SELECT * FROM lich_su ORDER BY thoi_gian DESC", conn)
    
    if df.empty:
        st.info("Chưa có giao dịch nào được ghi lại.")
        return

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

    st.subheader("Lịch sử giao dịch")
    st.dataframe(df_filtered)
    
    csv_file = df_filtered.to_csv(index=False)
    st.download_button(
        label="Tải xuống CSV",
        data=csv_file,
        file_name='lich_su_giao_dich.csv',
        mime='text/csv'
    )

# --- Chạy ứng dụng ---
if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        main_app()
    else:
        login_page()
