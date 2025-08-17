# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import pytz
import easyocr
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import mm
import re
import os
import matplotlib.pyplot as plt
import tempfile

# ========== CẤU HÌNH =============
st.set_page_config(layout="wide")

# --- Quản lý người dùng (đơn giản, demo) ---
users = {
    "admin": "admin123",
    "user1": "user123"
}

# --- Khởi tạo EasyOCR (cache) ---
@st.cache_resource
def get_reader():
    return easyocr.Reader(['vi', 'en'], gpu=False)

reader = get_reader()

# --- Kết nối SQLite ---
conn = sqlite3.connect("lich_su_giao_dich.db", check_same_thread=False)
c = conn.cursor()
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

# ========== HỖ TRỢ NHIỀU HÀM =============
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

# --- Image helpers ---
def _bytes_to_bgr(image_bytes):
    return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

def preprocess_image_for_ocr(image_bytes):
    img = _bytes_to_bgr(image_bytes)
    if img is None:
        return None
    # cải thiện: grayscale -> bilateral -> adaptive threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 9)
    # trả về màu BGR vì easyocr chấp nhận cả ảnh màu/ngang
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

# --- EasyOCR extract helper (dùng detail=0 -> list text) ---
def _easyocr_texts_from_bytes(image_bytes):
    img = _bytes_to_bgr(image_bytes)
    if img is None:
        return []
    try:
        texts = reader.readtext(img, detail=0)
        return [str(t).strip() for t in texts if t is not None]
    except Exception:
        # fallback: dùng preprocessed
        proc = preprocess_image_for_ocr(image_bytes)
        if proc is None:
            return []
        try:
            texts = reader.readtext(proc, detail=0)
            return [str(t).strip() for t in texts if t is not None]
        except Exception:
            return []

# --- Hàm OCR CCCD bằng EasyOCR ---
def trich_xuat_cccd_easy(image_bytes):
    ho_ten, so_cccd, que_quan = "", "", ""
    try:
        texts = _easyocr_texts_from_bytes(image_bytes)
        if not texts:
            return "", "", ""
        texts_upper = [t.upper() for t in texts]

        # Tìm "HỌ VÀ TÊN" hoặc "HỌ TÊN" hoặc "HỌ & TÊN"
        for i, t in enumerate(texts_upper):
            if "HỌ VÀ TÊN" in t or "HỌ TÊN" in t or "HỌ VÀ TÊN:" in t or "HỌ & TÊN" in t:
                if i + 1 < len(texts):
                    ho_ten = texts[i + 1]
                break
        # fallback: tìm dòng chứa "Họ" + dấu ví dụ "Họ tên: NGUYEN VAN A"
        if not ho_ten:
            for t in texts:
                m = re.search(r"Họ( và)? tên[:\s\-]*([A-Za-zÀ-ỹ\s]+)", t, re.IGNORECASE)
                if m:
                    ho_ten = m.group(2).strip()
                    break

        # Số CCCD (12 chữ số)
        pat_cccd = re.compile(r"\d{12}")
        for t in texts:
            m = pat_cccd.search(t.replace(" ", ""))
            if m:
                so_cccd = m.group(0)
                break

        # Quê quán
        for i, t in enumerate(texts_upper):
            if "QUÊ QUÁN" in t or "QUE QUAN" in t:
                if i + 1 < len(texts):
                    que_quan = texts[i + 1]
                break
        # fallback: nếu vẫn rỗng, tìm dòng chứa từ "QUÊ" hoặc "QUÊ QUÁN"
        if not que_quan:
            for t in texts:
                if "QUÊ" in t.upper():
                    que_quan = t
                    break

        return ho_ten, so_cccd, que_quan
    except Exception:
        return "", "", ""

# --- Hàm OCR cân bằng EasyOCR ---
def trich_xuat_can_easy(image_bytes):
    try:
        # dùng preprocessed ảnh cân để tăng độ chính xác số
        proc = preprocess_image_for_ocr(image_bytes)
        texts = []
        if proc is not None:
            try:
                texts = reader.readtext(proc, detail=0)
            except Exception:
                pass
        if not texts:
            texts = _easyocr_texts_from_bytes(image_bytes)
        if not texts:
            return ""
        candidates = []
        for t in texts:
            for m in re.findall(r"[0-9]+(?:[.,][0-9]+)?", str(t)):
                val = m.replace(",", ".")
                try:
                    candidates.append((m, float(val)))
                except:
                    pass
        if not candidates:
            return ""
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0].replace(",", ".")
    except Exception:
        return ""

# ========== Hàm tính tiền & PDF (giữ nguyên chức năng) ==========
def xu_ly_giao_dich(ho_va_ten, so_cccd, que_quan, so_luong_str, don_gia_str):
    try:
        so_luong = float(str(so_luong_str).replace(',', ''))
        don_gia = float(str(don_gia_str).replace(',', ''))
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

FONT_FILE = "arial.ttf"
FONT_NAME = "Arial"
try:
    if os.path.exists(FONT_FILE):
        pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_FILE))
    else:
        st.warning(f"Không tìm thấy font '{FONT_FILE}'. Vui lòng đặt font vào cùng thư mục với file app.")
except Exception as e:
    st.error(f"Lỗi khi đăng ký font: {e}")
    FONT_NAME = "Helvetica"

def tao_pdf_mau_01(data, ten_don_vi=""):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    pdf.setFont(FONT_NAME, 12)

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
    pdf.drawString(20*mm, height - 70*mm, f"Họ và tên người bán: {data['ho_va_ten']}")
    pdf.drawString(20*mm, height - 75*mm, f"Số CCCD: {data['so_cccd']}")
    pdf.drawString(20*mm, height - 80*mm, f"Quê quán: {data['que_quan']}")
    pdf.drawString(20*mm, height - 85*mm, f"Ngày lập: {data['ngay_tao']}")

    # --- Bảng hàng hóa ---
    y = height - 100*mm
    pdf.rect(20*mm, y-20*mm, 170*mm, 20*mm)
    pdf.drawString(22*mm, y - 5*mm, "STT")
    pdf.drawString(35*mm, y - 5*mm, "Tên hàng hóa, dịch vụ")
    pdf.drawString(100*mm, y - 5*mm, "Đơn vị tính")
    pdf.drawString(120*mm, y - 5*mm, "Số lượng")
    pdf.drawString(140*mm, y - 5*mm, "Đơn giá")
    pdf.drawString(170*mm, y - 5*mm, "Thành tiền")

    pdf.drawString(22*mm, y - 15*mm, "1")
    pdf.drawString(35*mm, y - 15*mm, "Hàng hóa")
    pdf.drawString(100*mm, y - 15*mm, "chỉ")
    pdf.drawString(120*mm, y - 15*mm, f"{data['so_luong']:,.2f}")
    pdf.drawString(140*mm, y - 15*mm, f"{data['don_gia']:,.0f}")
    pdf.drawString(170*mm, y - 15*mm, f"{data['thanh_tien']:,.0f}")

    # --- Tổng cộng ---
    y -= 30*mm
    pdf.drawString(20*mm, y, f"Tổng cộng: {data['thanh_tien']:,.0f} VNĐ")

    y -= 5*mm
    pdf.drawString(20*mm, y, f"Bằng chữ: {doc_so_thanh_chu(data['thanh_tien'])}")

    # --- Xuống ngay dưới để thêm ngày tháng và chữ ký ---
    y -= 20*mm
    pdf.setFont(FONT_NAME, 11)
    # Thay thế dòng này để hiển thị Bến Lức và ngày tháng năm hiện tại
    vn_timezone = pytz.timezone('Asia/Ho_Chi_Minh')
    current_time = datetime.now(vn_timezone)
    current_date_str = f"Bến Lức, ngày {current_time.day} tháng {current_time.month} năm {current_time.year}"
    pdf.drawRightString(width - 30*mm, y, current_date_str)

    # Người mua (bên trái)
    pdf.drawString(30*mm, y - 20, "Người mua")
    pdf.drawString(30*mm, y - 30, "(Ký, ghi rõ họ tên)")

    # Giám đốc (bên phải)
    pdf.drawRightString(width - 30*mm, y - 20, "Giám đốc")
    pdf.drawRightString(width - 30*mm, y - 30, "(Ký, đóng dấu)")

    pdf.save()
    buffer.seek(0)
    return buffer


# ========== GIAO DIỆN ==========
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
    # Khởi tạo session_state mặc định
    defaults = {
        "ho_ten": "",
        "so_cccd": "",
        "que_quan": "",
        "so_luong": "",
        "pdf_for_download": None,
        "giao_dich_data": None,
        "don_gia_input": "",
        "ten_don_vi": ""
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    st.subheader("1. Trích xuất thông tin từ ảnh 🖼️")
    col_cccd, col_can = st.columns([1,1])

    with col_cccd:
        st.subheader("Chụp ảnh hoặc tải ảnh CCCD")
        anh_cccd = st.camera_input("Chụp ảnh CCCD")
        uploaded_cccd = st.file_uploader("Hoặc tải ảnh CCCD", type=["jpg", "jpeg", "png"], key="cccd_uploader")
        if anh_cccd:
            with st.spinner("Đang xử lý OCR CCCD..."):
                ho_ten, so_cccd, que_quan = trich_xuat_cccd_easy(anh_cccd.read())
            if ho_ten:
                st.session_state.ho_ten = ho_ten
            if so_cccd:
                st.session_state.so_cccd = so_cccd
            if que_quan:
                st.session_state.que_quan = que_quan
            st.success("Đã trích xuất thông tin CCCD!")
            st.image(anh_cccd, use_container_width=True)
        elif uploaded_cccd:
            with st.spinner("Đang xử lý OCR CCCD..."):
                ho_ten, so_cccd, que_quan = trich_xuat_cccd_easy(uploaded_cccd.read())
            if ho_ten:
                st.session_state.ho_ten = ho_ten
            if so_cccd:
                st.session_state.so_cccd = so_cccd
            if que_quan:
                st.session_state.que_quan = que_quan
            st.success("Đã trích xuất thông tin CCCD!")
            st.image(uploaded_cccd, use_container_width=True)

    with col_can:
        st.subheader("Chụp ảnh hoặc tải ảnh cân")
        anh_can = st.camera_input("Chụp ảnh màn hình cân")
        uploaded_can = st.file_uploader("Hoặc tải ảnh cân", type=["jpg", "jpeg", "png"], key="can_uploader")
        if anh_can:
            with st.spinner("Đang xử lý OCR cân..."):
                so_luong = trich_xuat_can_easy(anh_can.read())
            if so_luong:
                st.session_state.so_luong = so_luong
            st.success("Đã trích xuất khối lượng!")
            st.image(anh_can, use_contai
