import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pytz
from paddleocr import PaddleOCR

# --- Khởi tạo OCR ---
@st.cache_resource
def get_reader():
    return PaddleOCR(lang="vi", use_angle_cls=False)

ocr = get_reader()
lich_su_file = 'lich_su_giao_dich.csv'

if not os.path.exists(lich_su_file):
    df = pd.DataFrame(columns=['Thời gian', 'Họ và Tên', 'Số CCCD', 'Quê quán', 'Khối lượng', 'Đơn giá', 'Thành tiền'])
    df.to_csv(lich_su_file, index=False)

# --- Hàm OCR trực tiếp ---
def ocr_image(file):
    if file is None:
        return None
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    return img

def trich_xuat_cccd(file):
    if file is None:
        return "", "", ""
    img = ocr_image(file)
    if img is None:
        return "", "", ""

    result = ocr.ocr(img, cls=False)
    if not result or not result[0]:
        return "", "", ""

    ho_ten, so_cccd, que_quan = "", "", ""

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

def trich_xuat_can(file):
    if file is None:
        return ""
    img = ocr_image(file)
    if img is None:
        return ""

    result = ocr.ocr(img, cls=False)
    if not result or not result[0]:
        return ""

    for line in result[0]:
        text = line[1][0]
        cleaned_text = ''.join(c for c in text if c.isdigit() or c == '.')
        if cleaned_text:
            return cleaned_text
    return ""

def xu_ly_giao_dich(ho_va_ten, so_cccd, que_quan, so_luong_str, don_gia_str):
    try:
        so_luong = float(so_luong_str.replace(',', ''))
        don_gia = float(don_gia_str.replace(',', ''))
        thanh_tien = so_luong * don_gia

        vn_timezone = pytz.timezone('Asia/Ho_Chi_Minh')
        current_time = datetime.now(vn_timezone)
        ngay_tao = current_time.strftime("%d/%m/%Y")
        thoi_gian_luu = current_time.strftime("%Y-%m-%d %H:%M:%S")

        noi_dung_ban_ke = f"""
  CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
              Độc lập - Tự do - Hạnh phúc
  ----------------------------------------
                BẢN KÊ MUA HÀNG
                 (Mẫu số 01/TNDN)

  Thông tin khách hàng:
  - Họ và Tên: {ho_va_ten}
  - Số CCCD: {so_cccd}
  - Quê quán: {que_quan}

  Chi tiết giao dịch:
  - Khối lượng: {so_luong}
  - Đơn giá: {don_gia:,.0f} VNĐ
  - Thành tiền: {thanh_tien:,.0f} VNĐ
  ----------------------------------------
  Bến Lức, ngày {ngay_tao}
      NGƯỜI BÁN                   KHÁCH HÀNG
      (Ký, ghi rõ họ tên)          (Ký, ghi rõ họ tên)
    """

        df_moi = pd.DataFrame([{
            'Thời gian': thoi_gian_luu,
            'Họ và Tên': ho_va_ten,
            'Số CCCD': so_cccd,
            'Quê quán': que_quan,
            'Khối lượng': so_luong,
            'Đơn giá': don_gia,
            'Thành tiền': thanh_tien
        }])
        df_moi.to_csv(lich_su_file, mode='a', header=False, index=False)
        return noi_dung_ban_ke, thanh_tien
    except ValueError:
        return "Lỗi: Khối lượng và Đơn giá phải là số.", 0.0

# --- Giao diện Streamlit ---
st.set_page_config(layout="wide")
st.title("ỨNG DỤNG TẠO BẢN KÊ MUA HÀNG (MẪU 01/TNDN)")
st.markdown("---")

if 'ho_ten' not in st.session_state:
    st.session_state.ho_ten = ""
if 'so_cccd' not in st.session_state:
    st.session_state.so_cccd = ""
if 'que_quan' not in st.session_state:
    st.session_state.que_quan = ""
if 'so_luong' not in st.session_state:
    st.session_state.so_luong = ""

# --- Nhập thông tin khách hàng ---
st.header("1. Nhập thông tin khách hàng")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Trích xuất từ CCCD")
    anh_cccd = st.camera_input("Chụp hoặc tải ảnh CCCD")
    if anh_cccd:
        st.session_state.ho_ten, st.session_state.so_cccd, st.session_state.que_quan = trich_xuat_cccd(anh_cccd)

with col2:
    st.subheader("Nhập liệu thủ công")
    ho_ten_input = st.text_input("Họ và Tên", value=st.session_state.ho_ten)
    so_cccd_input = st.text_input("Số Căn cước công dân", value=st.session_state.so_cccd)
    que_quan_input = st.text_input("Quê quán", value=st.session_state.que_quan)

st.markdown("---")

# --- Nhập thông tin giao dịch ---
st.header("2. Nhập thông tin giao dịch")
col3, col4 = st.columns(2)

with col3:
    st.subheader("Trích xuất từ cân")
    anh_can = st.camera_input("Chụp hoặc tải ảnh màn hình cân")
    if anh_can:
        st.session_state.so_luong = trich_xuat_can(anh_can)

with col4:
    st.subheader("Nhập liệu thủ công")
    so_luong_input = st.text_input("Khối lượng (chỉ)", value=st.session_state.so_luong)
    don_gia_input = st.text_input("Đơn giá (VNĐ/chỉ)")

st.markdown("---")

# --- Tạo bản kê ---
st.header("3. Tạo bản kê")
if st.button("Tính tiền và Tạo bản kê"):
    if not ho_ten_input or not so_luong_input or not don_gia_input:
        st.error("Vui lòng nhập đầy đủ thông tin.")
    else:
        noi_dung_ban_ke, thanh_tien = xu_ly_giao_dich(
            ho_ten_input, so_cccd_input, que_quan_input, so_luong_input, don_gia_input
        )
        st.text_area("Bản Kê Mua Hàng", value=noi_dung_ban_ke, height=400)
        st.metric(label="Thành Tiền", value=f"{thanh_tien:,.0f} VNĐ")

st.markdown("---")

# --- Lịch sử giao dịch ---
st.header("4. Lịch sử giao dịch")
if os.path.exists(lich_su_file):
    df_lich_su = pd.read_csv(lich_su_file)
    st.dataframe(df_lich_su)
else:
    st.info("Chưa có giao dịch nào.")
