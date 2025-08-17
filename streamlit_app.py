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
import tempfile

# ========== CẤU HÌNH =============
st.set_page_config(layout="wide")

# --- Quản lý người dùng (đơn giản, dùng cho demo) ---
users = {
    "admin": "admin123",
    "user1": "user123"
}

# --- Khởi tạo OCR (cache để nhanh) ---
@st.cache_resource
def get_reader():
    return PaddleOCR(lang="vi", use_angle_cls=False)

ocr = get_reader()

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

# --- OCR helpers (an toàn với nhiều dạng output) ---
def _bytes_to_bgr(image_bytes):
    return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

def _safe_ocr_call(img_bgr):
    """Gọi ocr.ocr với fallback: BGR -> RGB -> tệp tạm."""
    try:
        return ocr.ocr(img_bgr)
    except Exception:
        try:
            img_rgb = img_bgr[:, :, ::-1]
            return ocr.ocr(img_rgb)
        except Exception:
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                cv2.imwrite(tmp_path, img_bgr)
                res = ocr.ocr(tmp_path)
                os.remove(tmp_path)
                return res
            except Exception:
                return None

def _extract_texts_from_ocr_result(result):
    texts = []
    try:
        if not result:
            return texts
        batch = result[0] if isinstance(result, (list, tuple)) else result
        if not batch:
            return texts
        for line in batch:
            if isinstance(line, (list, tuple)):
                if len(line) >= 2:
                    item = line[1]
                    if isinstance(item, (list, tuple)) and len(item) >= 1 and isinstance(item[0], str):
                        texts.append(item[0])
                    elif isinstance(item, dict) and isinstance(item.get("text"), str):
                        texts.append(item["text"])
                    elif isinstance(item, str):
                        texts.append(item)
                elif len(line) >= 1 and isinstance(line[0], str):
                    texts.append(line[0])
            elif isinstance(line, dict) and isinstance(line.get("text"), str):
                texts.append(line["text"])
    except Exception:
        pass
    return texts

def preprocess_image_for_ocr(image_bytes):
    img = _bytes_to_bgr(image_bytes)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    equalized = cv2.equalizeHist(blurred)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

# --- Hàm OCR CCCD (an toàn) ---
def trich_xuat_cccd(image_bytes):
    ho_ten, so_cccd, que_quan = "", "", ""
    try:
        proc_bgr = preprocess_image_for_ocr(image_bytes)
        if proc_bgr is None:
            return ho_ten, so_cccd, que_quan
        result = _safe_ocr_call(proc_bgr)
        all_text_raw = _extract_texts_from_ocr_result(result)
        if not all_text_raw:
            return ho_ten, so_cccd, que_quan
        all_text_upper = [str(t).upper() for t in all_text_raw]

        # Họ và tên (dòng sau "HỌ VÀ TÊN")
        for i, t in enumerate(all_text_upper):
            if "HỌ VÀ TÊN" in t:
                if i + 1 < len(all_text_raw):
                    ho_ten = str(all_text_raw[i + 1]).strip()
                break

        # Số CCCD (12 chữ số)
        pat_cccd = re.compile(r"\d{12}")
        for t in all_text_raw:
            m = pat_cccd.search(str(t).replace(" ", ""))
            if m:
                so_cccd = m.group(0)
                break

        # Quê quán (dòng sau "QUÊ QUÁN")
        for i, t in enumerate(all_text_upper):
            if "QUÊ QUÁN" in t:
                if i + 1 < len(all_text_raw):
                    que_quan = str(all_text_raw[i + 1]).strip()
                break

        return ho_ten, so_cccd, que_quan
    except Exception:
        return "", "", ""

# --- Hàm OCR cân (an toàn) ---
def trich_xuat_can(image_bytes):
    try:
        proc_bgr = preprocess_image_for_ocr(image_bytes)
        if proc_bgr is None:
            return ""
        gray = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10)
        proc_bgr2 = cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)
        result = _safe_ocr_call(proc_bgr2)
        texts = _extract_texts_from_ocr_result(result)
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
        best = candidates[0][0].replace(",", ".")
        return best
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

FONT_FILE = "Arial.ttf"
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
    y -= 30*mm
    pdf.drawString(20*mm, y, f"Tổng cộng: {data['thanh_tien']:,.0f} VNĐ")
    y -= 5*mm
    pdf.drawString(20*mm, y, f"Bằng chữ: {doc_so_thanh_chu(data['thanh_tien'])}")
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
                ho_ten, so_cccd, que_quan = trich_xuat_cccd(anh_cccd.read())
                # Ghi vào session_state (nếu trống thì giữ giá trị rỗng)
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
                ho_ten, so_cccd, que_quan = trich_xuat_cccd(uploaded_cccd.read())
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
                so_luong = trich_xuat_can(anh_can.read())
                if so_luong:
                    st.session_state.so_luong = so_luong
            st.success("Đã trích xuất khối lượng!")
            st.image(anh_can, use_container_width=True)
        elif uploaded_can:
            with st.spinner("Đang xử lý OCR cân..."):
                so_luong = trich_xuat_can(uploaded_can.read())
                if so_luong:
                    st.session_state.so_luong = so_luong
            st.success("Đã trích xuất khối lượng!")
            st.image(uploaded_can, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Nhập đơn giá và lưu giao dịch 📝")

    # Hiển thị tóm tắt (thông tin sẵn có)
    st.info(f"Họ và Tên: **{st.session_state.ho_ten}**")
    st.info(f"Số CCCD: **{st.session_state.so_cccd}**")
    st.info(f"Quê quán: **{st.session_state.que_quan}**")
    st.info(f"Khối lượng: **{st.session_state.so_luong}** chỉ")

    st.write("**(Nếu OCR đã trích xuất được, ô tương ứng sẽ bị khóa — không thể nhập lại. Nếu chưa có, bạn có thể nhập thủ công.)**")

    # Họ tên => khóa nếu OCR có, else cho nhập
    if st.session_state.get("ho_ten"):
        st.text_input("Họ và tên người bán", value=st.session_state.ho_ten, disabled=True, key="ho_ten_disabled")
    else:
        st.text_input("Họ và tên người bán", key="ho_ten")

    # Số CCCD
    if st.session_state.get("so_cccd"):
        st.text_input("Số CCCD", value=st.session_state.so_cccd, disabled=True, key="so_cccd_disabled")
    else:
        st.text_input("Số CCCD", key="so_cccd")

    # Quê quán
    if st.session_state.get("que_quan"):
        st.text_area("Quê quán", value=st.session_state.que_quan, disabled=True, key="que_quan_disabled")
    else:
        st.text_area("Quê quán", key="que_quan")

    # Khối lượng (chỉ)
    if st.session_state.get("so_luong"):
        st.text_input("Khối lượng (chỉ)", value=st.session_state.so_luong, disabled=True, key="so_luong_disabled")
    else:
        st.text_input("Khối lượng (chỉ)", key="so_luong")

    # Don gia and ten don vi luôn để nhập (người dùng cung cấp)
    st.text_input("Đơn giá (VNĐ/chỉ)", key="don_gia_input")
    st.text_input("Tên đơn vị (không bắt buộc)", key="ten_don_vi")

    # Khi lưu: lấy value ưu tiên từ các key editable (nếu có), else từ disabled key
    def _get_value(field):
        # field: "ho_ten", "so_cccd", "que_quan", "so_luong"
        # ưu tiên key (editable) tồn tại -> st.session_state[field]
        if st.session_state.get(field):
            return st.session_state.get(field)
        # else check disabled key
        disabled_key = field + "_disabled"
        return st.session_state.get(disabled_key, "")

    if st.button("Lưu giao dịch"):
        ho_va_ten = _get_value("ho_ten")
        so_cccd_val = _get_value("so_cccd")
        que_quan_val = _get_value("que_quan")
        so_luong_val = _get_value("so_luong")
        don_gia_val = st.session_state.get("don_gia_input", "")

        if not ho_va_ten or not so_luong_val or not don_gia_val:
            st.error("Vui lòng đảm bảo đã trích xuất/nhập Họ tên, Khối lượng và nhập đơn giá trước khi lưu.")
        else:
            giao_dich_data = xu_ly_giao_dich(ho_va_ten, so_cccd_val, que_quan_val, so_luong_val, don_gia_val)
            if giao_dich_data:
                st.success("Giao dịch đã được lưu thành công!")
                st.metric(label="Thành Tiền", value=f"{giao_dich_data['thanh_tien']:,.0f} VNĐ")
                st.write(f"Bằng chữ: {doc_so_thanh_chu(giao_dich_data['thanh_tien'])}")

                pdf_buffer = tao_pdf_mau_01(giao_dich_data, st.session_state.get("ten_don_vi", ""))
                st.session_state.pdf_for_download = pdf_buffer
                st.session_state.giao_dich_data = giao_dich_data

    # Hiển thị download PDF nếu có
    if st.session_state.pdf_for_download:
        st.download_button(
            "Tải bản kê PDF (Mẫu 01/TNDN)",
            data=st.session_state.pdf_for_download.getvalue(),
            file_name=f"bang_ke_{(st.session_state.giao_dich_data['ho_va_ten']).replace(' ', '_')}.pdf",
            mime="application/pdf"
        )

    st.markdown("---")
    if st.button("Làm mới trang"):
        # reset keys (giữ login)
        for k in ["ho_ten", "so_cccd", "que_quan", "so_luong", "pdf_for_download", "giao_dich_data", "don_gia_input", "ten_don_vi",
                  "ho_ten_disabled", "so_cccd_disabled", "que_quan_disabled", "so_luong_disabled"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

def history_and_stats_page():
    st.header("Lịch sử và Thống kê")
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
    st.download_button(label="Tải xuống CSV", data=csv_file, file_name='lich_su_giao_dich.csv', mime='text/csv')

# --- Chạy ứng dụng ---
if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if st.session_state.logged_in:
        main_app()
    else:
        login_page()
