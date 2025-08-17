import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pytz
import re

# --- Khởi tạo và Thiết lập ---
# Cấu hình đường dẫn Tesseract (nếu chạy cục bộ trên máy tính)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

lich_su_file = 'lich_su_giao_dich.csv'

if not os.path.exists(lich_su_file):
    df = pd.DataFrame(columns=['Thời gian', 'Họ và Tên', 'Số CCCD', 'Quê quán', 'Khối lượng', 'Đơn giá', 'Thành tiền'])
    df.to_csv(lich_su_file, index=False)

# --- Các hàm xử lý logic ---
def trich_xuat_cccd(image):
    if image is None: return "", "", ""
    try:
        img_pil = Image.open(image)
        text = pytesseract.image_to_string(img_pil, lang='vie+eng')
        lines = text.split('\n')
        
        ho_ten, so_cccd, que_quan = "", "", ""
        
        for i, line in enumerate(lines):
            line_upper = line.strip().upper()
            if "HỌ VÀ TÊN" in line_upper:
                # Tìm dòng tiếp theo sau "Họ và Tên"
                if i + 1 < len(lines):
                    ho_ten = lines[i+1].strip()
            elif "SỐ" in line_upper:
                # Sử dụng biểu thức chính quy để tìm 12 chữ số
                match = re.search(r'\d{12}', line)
                if match:
                    so_cccd = match.group(0)
            elif "QUÊ QUÁN" in line_upper:
                # Tìm dòng tiếp theo sau "Quê quán"
                if i + 1 < len(lines):
                    que_quan = lines[i+1].strip()
        
        return ho_ten, so_cccd, que_quan
    except Exception as e:
        st.error(f"Lỗi khi trích xuất CCCD: {e}")
        return "", "", ""

def trich_xuat_can(image):
    if image is None: return ""
    try:
        img_pil = Image.open(image)
        text = pytesseract.image_to_string(img_pil, config='--psm 6 outputbase digits')
        cleaned_text = ''.join(c for c in text if c.isdigit() or c == '.')
        return cleaned_text
    except Exception as e:
        st.error(f"Lỗi khi trích xuất từ cân: {e}")
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
  BẢN KÊ MUA HÀNG

  ----------------------------------------
  Thông tin khách hàng:
  - Họ và Tên: {ho_va_ten}
  - Số CCCD: {so_cccd}
  - Quê quán: {que_quan}
  ----------------------------------------
  Chi tiết giao dịch:
  - Khối lượng: {so_luong}
  - Đơn giá: {don_gia:,.0f} VNĐ
  - Thành tiền: {thanh_tien:,.0f} VNĐ
  ----------------------------------------
  TP. HCM, ngày {ngay_tao}
      NGƯỜI BÁN                   KHÁCH HÀNG
      (Ký tên)                     (Ký tên)
    """
        
        df_moi = pd.DataFrame([{'Thời gian': thoi_gian_luu, 'Họ và Tên': ho_va_ten, 'Số CCCD': so_cccd, 'Quê quán': que_quan, 'Khối lượng': so_luong, 'Đơn giá': don_gia, 'Thành tiền': thanh_tien}])
        df_moi.to_csv(lich_su_file, mode='a', header=False, index=False)
        return noi_dung_ban_ke, thanh_tien
    except ValueError:
        return "Lỗi: Khối lượng và Đơn giá phải là số.", 0.0

# --- Giao diện Streamlit ---
st.set_page_config(layout="wide")
st.title("ỨNG DỤNG TẠO BẢN KÊ MUA HÀNG")
st.markdown("---")

if 'ho_ten' not in st.session_state:
    st.session_state.ho_ten = ""
if 'so_cccd' not in st.session_state:
    st.session_state.so_cccd = ""
if 'que_quan' not in st.session_state:
    st.session_state.que_quan = ""
if 'so_luong' not in st.session_state:
    st.session_state.so_luong = ""

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

st.header("3. Tạo bản kê")
if st.button("Tính tiền và Tạo bản kê"):
    if not ho_ten_input or not so_luong_input or not don_gia_input:
        st.error("Vui lòng nhập đầy đủ thông tin.")
    else:
        noi_dung_ban_ke, thanh_tien = xu_ly_giao_dich(ho_ten_input, so_cccd_input, que_quan_input, so_luong_input, don_gia_input)
        st.text_area("Bản Kê Mua Hàng", value=noi_dung_ban_ke, height=300)
        st.metric(label="Thành Tiền", value=f"{thanh_tien:,.0f} VNĐ")
        
st.markdown("---")

st.header("4. Lịch sử giao dịch")
if os.path.exists(lich_su_file):
    df_lich_su = pd.read_csv(lich_su_file)
    st.dataframe(df_lich_su)
else:
    st.info("Chưa có giao dịch nào.")
