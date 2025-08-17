# app.py
# Streamlit app: BẢNG KÊ 01/TNDN
# - OCR trực tiếp (camera/file) cho CCCD và cân (PaddleOCR)
# - Giữ CCCD là chuỗi (không mất số 0)
# - Xuất PDF (ReportLab) với font Unicode (DejaVu nếu có), bảng kẻ, in-ready
# - Fallback HTML in-ready
# - Preview PDF/HTML và tải xuống
# - Lưu lịch sử tự động (CSV)
#
# NOTE: Place DejaVuSans.ttf in same folder for best Vietnamese support in PDF.
# Requirements example:
# streamlit
# opencv-python-headless
# numpy
# pandas
# paddleocr
# paddlepaddle
# pytz
# reportlab

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io
import os
import sys
import logging
from datetime import datetime
import pytz
from paddleocr import PaddleOCR
import base64
from typing import Tuple, Dict, Any, Optional, List

# ReportLab imports (optional)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# ---------------- Config ----------------
APP_TITLE = "BẢNG KÊ 01/TNDN — OCR CCCD & Cân — PDF/HTML đẹp"
LICH_SU_FILE = "lich_su_giao_dich.csv"
CSV_COLUMNS = [
    "Thời gian", "Đơn vị bán hàng", "MST", "Địa chỉ đơn vị",
    "Địa điểm thu mua", "Người phụ trách",
    "Họ và Tên", "Số CCCD", "Quê quán",
    "Khối lượng", "Đơn vị tính", "Đơn giá", "Thành tiền"
]
DEFAULT_DIA_DIEM = "Bến Lức"
TIMEZONE = "Asia/Ho_Chi_Minh"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# Streamlit config
st.set_page_config(page_title="Bảng kê 01/TNDN", layout="wide", page_icon="📄")

# Ensure history csv exists
if not os.path.exists(LICH_SU_FILE):
    df_empty = pd.DataFrame(columns=CSV_COLUMNS)
    df_empty.to_csv(LICH_SU_FILE, index=False, encoding="utf-8")
    logger.info("Created history CSV: %s", LICH_SU_FILE)

# ---------------- OCR init ----------------
@st.cache_resource
def init_ocr():
    try:
        reader = PaddleOCR(lang="vi", use_angle_cls=False)
        logger.info("Initialized PaddleOCR.")
        return reader
    except Exception as e:
        logger.exception("Failed to init PaddleOCR: %s", e)
        raise

try:
    ocr = init_ocr()
except Exception:
    ocr = None

# ---------------- Helpers ----------------
def now_local_str(fmt: str = "%d/%m/%Y") -> str:
    tz = pytz.timezone(TIMEZONE)
    return datetime.now(tz).strftime(fmt)

def now_iso() -> str:
    tz = pytz.timezone(TIMEZONE)
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

def safe_float_from_str(s: str) -> float:
    if s is None:
        return 0.0
    s0 = str(s).strip()
    if s0 == "":
        return 0.0
    s0 = s0.replace(" ", "")
    candidates = []
    if "." in s0 and "," in s0:
        candidates.append(s0.replace(".", "").replace(",", "."))
        candidates.append(s0.replace(",", ""))
    else:
        if "," in s0 and "." not in s0:
            candidates.append(s0.replace(",", ""))
        else:
            candidates.append(s0)
    for cand in candidates:
        try:
            return float(cand)
        except:
            continue
    cleaned = "".join(ch for ch in s0 if ch.isdigit() or ch == ".")
    try:
        return float(cleaned) if cleaned else 0.0
    except:
        return 0.0

def fmt_money(v: float) -> str:
    try:
        n = int(round(v))
        return f"{n:,}".replace(",", ".")
    except:
        return "0"

dv_words = ['không','một','hai','ba','bốn','năm','sáu','bảy','tám','chín']
def read3(n: int) -> str:
    s = ""
    tr = n // 100
    ch = (n % 100) // 10
    d = n % 10
    if tr > 0:
        s += dv_words[tr] + " trăm"
        if ch == 0 and d > 0:
            s += " linh"
    if ch > 1:
        s += (" " if s else "") + dv_words[ch] + " mươi"
        if d == 1:
            s += " mốt"
        elif d == 5:
            s += " lăm"
        elif d > 0:
            s += " " + dv_words[d]
    elif ch == 1:
        s += (" " if s else "") + "mười"
        if d == 5:
            s += " lăm"
        elif d > 0:
            s += " " + dv_words[d]
    elif ch == 0 and d > 0:
        s += (" " if s else "") + dv_words[d]
    return s.strip()

def to_words_vnd(num: float) -> str:
    try:
        num = int(round(num))
    except:
        return "Không đồng"
    if num <= 0:
        return "Không đồng"
    units = ['',' nghìn',' triệu',' tỷ',' nghìn tỷ',' triệu tỷ']
    out = []
    i = 0
    while num > 0 and i < len(units):
        chunk = num % 1000
        if chunk > 0:
            out.insert(0, (read3(chunk) + units[i]).strip())
        num //= 1000
        i += 1
    s = ' '.join(out).strip()
    return s[0].upper() + s[1:] + " đồng"

def normalize_cccd_candidate(candidate: str) -> str:
    if candidate is None:
        return ""
    digits = ''.join(ch for ch in str(candidate) if ch.isdigit())
    if len(digits) >= 12:
        return digits[:12]
    if 9 <= len(digits) < 12:
        return digits.zfill(12)
    return digits

# ---------------- OCR helpers ----------------
def img_from_uploaded_file(uploaded) -> Optional[np.ndarray]:
    if uploaded is None:
        return None
    try:
        data = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
        file_bytes = np.asarray(bytearray(data), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.exception("img_from_uploaded_file error: %s", e)
        return None

def ocr_image_to_raw_lines(img: np.ndarray) -> List[Tuple]:
    if img is None:
        return []
    if ocr is None:
        logger.warning("OCR engine not initialized.")
        return []
    try:
        result = ocr.ocr(img, cls=False)
        if not result:
            return []
        return result[0]
    except Exception as e:
        logger.exception("ocr_image_to_raw_lines error: %s", e)
        return []

def extract_text_lines_from_uploaded(uploaded) -> List[str]:
    img = img_from_uploaded_file(uploaded)
    if img is None:
        return []
    raw = ocr_image_to_raw_lines(img)
    lines = []
    for ln in raw:
        try:
            lines.append(ln[1][0].strip())
        except:
            continue
    return lines

# ---------------- Extract CCCD & scale ----------------
def trich_xuat_cccd_from_uploaded(uploaded) -> Tuple[str, str, str]:
    ho_ten = ""
    so_cccd = ""
    que_quan = ""
    if uploaded is None:
        return ho_ten, so_cccd, que_quan
    img = img_from_uploaded_file(uploaded)
    if img is None:
        return ho_ten, so_cccd, que_quan
    raw = ocr_image_to_raw_lines(img)
    texts = [ln[1][0].strip() for ln in raw if ln and ln[1] and ln[1][0]]
    for idx, txt in enumerate(texts):
        up = txt.upper()
        if "HỌ VÀ TÊN" in up or "HO VA TEN" in up or "HỌ TÊN" in up:
            if idx + 1 < len(texts):
                candidate = texts[idx + 1].strip()
                if candidate:
                    ho_ten = candidate
        digits = ''.join(ch for ch in txt if ch.isdigit())
        if len(digits) >= 9:
            c = normalize_cccd_candidate(digits)
            if c:
                so_cccd = c
        if "QUÊ QUÁN" in up or "QUE QUAN" in up:
            if idx + 1 < len(texts):
                candidate = texts[idx + 1].strip()
                if candidate:
                    que_quan = candidate
    if not so_cccd:
        for txt in texts:
            digits = ''.join(ch for ch in txt if ch.isdigit())
            if len(digits) >= 12:
                so_cccd = normalize_cccd_candidate(digits)
                break
    return ho_ten, so_cccd, que_quan

def trich_xuat_can_from_uploaded(uploaded) -> str:
    if uploaded is None:
        return ""
    img = img_from_uploaded_file(uploaded)
    if img is None:
        return ""
    raw = ocr_image_to_raw_lines(img)
    for ln in raw:
        txt = ln[1][0]
        cleaned = ''.join(ch for ch in txt if ch.isdigit() or ch in '.,')
        if any(ch.isdigit() for ch in cleaned):
            cleaned2 = cleaned.replace(',', '.')
            parts = cleaned2.split('.')
            if len(parts) > 2:
                cleaned2 = ''.join(parts[:-1]) + '.' + parts[-1]
            cleaned2 = cleaned2.strip('.')
            return cleaned2
    return ""

# ---------------- CSV functions ----------------
def append_history_row(row: Dict[str, Any]) -> None:
    try:
        df = pd.DataFrame([row])
        df = df.reindex(columns=CSV_COLUMNS)
        df.to_csv(LICH_SU_FILE, mode='a', header=False, index=False, encoding='utf-8')
        logger.info("Appended history row.")
    except Exception as e:
        logger.exception("append_history_row error: %s", e)

def read_history_df() -> pd.DataFrame:
    try:
        df = pd.read_csv(LICH_SU_FILE, encoding='utf-8')
        return df
    except Exception as e:
        logger.exception("read_history_df error: %s", e)
        return pd.DataFrame(columns=CSV_COLUMNS)

# ---------------- PDF builder (fixed font + table) ----------------
def build_pdf_bytes_from_row(row: Dict[str, Any]) -> Optional[bytes]:
    if not REPORTLAB_OK:
        logger.warning("ReportLab not installed; cannot create PDF.")
        return None
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                leftMargin=18*mm, rightMargin=18*mm,
                                topMargin=18*mm, bottomMargin=18*mm)
        # Register DejaVu if present; else register CID font fallback
        font_name = "Helvetica"
        try:
            if os.path.exists("DejaVuSans.ttf"):
                pdfmetrics.registerFont(TTFont("DejaVu", "DejaVuSans.ttf"))
                font_name = "DejaVu"
                logger.info("Using DejaVu font for PDF.")
            else:
                # register CID fallback
                pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
                font_name = "STSong-Light"
                logger.info("Using STSong-Light (CID) as fallback font.")
        except Exception as e:
            logger.exception("Font registration failed: %s", e)
            font_name = "Helvetica"

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='VNTitle', fontName=font_name, fontSize=13, alignment=1, leading=15))
        styles.add(ParagraphStyle(name='VNNormal', fontName=font_name, fontSize=10, leading=12))
        styles.add(ParagraphStyle(name='VNSmall', fontName=font_name, fontSize=9, leading=11))

        elements = []
        elements.append(Paragraph("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", styles['VNSmall']))
        elements.append(Paragraph("Độc lập - Tự do - Hạnh phúc", styles['VNSmall']))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("MẪU SỐ: 01/TNDN", styles['VNSmall']))
        elements.append(Spacer(1, 8))
        elements.append(Paragraph("BẢNG KÊ THU MUA HÀNG HÓA, DỊCH VỤ MUA VÀO KHÔNG CÓ HÓA ĐƠN", styles['VNTitle']))
        elements.append(Spacer(1, 10))

        if row.get('don_vi'):
            elements.append(Paragraph(f"Đơn vị: {row.get('don_vi')}", styles['VNSmall']))
        if row.get('mst'):
            elements.append(Paragraph(f"MST: {row.get('mst')}", styles['VNSmall']))
        if row.get('dia_chi'):
            elements.append(Paragraph(f"Địa chỉ: {row.get('dia_chi')}", styles['VNSmall']))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(f"Địa điểm thu mua: {row.get('dia_diem','')}", styles['VNSmall']))
        elements.append(Paragraph(f"Người phụ trách: {row.get('phu_trach','')}", styles['VNSmall']))
        elements.append(Paragraph(f"Ngày lập: {row.get('ngay_lap','')}", styles['VNSmall']))
        elements.append(Spacer(1, 8))

        elements.append(Paragraph("<b>Thông tin người bán</b>", styles['VNNormal']))
        elements.append(Paragraph(f"Họ và tên: {row.get('ho_va_ten','')}", styles['VNNormal']))
        elements.append(Paragraph(f"Số CCCD/CMND: {row.get('so_cccd','')}", styles['VNNormal']))
        elements.append(Paragraph(f"Quê quán: {row.get('que_quan','')}", styles['VNNormal']))
        elements.append(Spacer(1, 8))

        table_data = [
            ["STT", "Tên hàng/dịch vụ", "ĐVT", "Số lượng", "Đơn giá (VNĐ)", "Thành tiền (VNĐ)"],
            [
                "1",
                str(row.get('mieu_ta', 'Hàng hóa')),
                str(row.get('don_vi_unit', '')),
                str(row.get('so_luong', '')),
                fmt_money(row.get('don_gia', 0)),
                fmt_money(row.get('thanh_tien', 0))
            ]
        ]
        col_widths = [18*mm, 80*mm, 22*mm, 30*mm, 35*mm, 40*mm]
        t = Table(table_data, colWidths=col_widths, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), font_name),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f3f6fb")),
            ('ALIGN', (0,0), (-1,0), 'CENTER'),
            ('ALIGN', (3,1), (5, -1), 'RIGHT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#bfc9d9")),
            ('LEFTPADDING', (1,1), (-1,-1), 6),
            ('RIGHTPADDING', (1,1), (-1,-1), 6),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(f"Tổng cộng: {fmt_money(row.get('thanh_tien', 0))} VNĐ", styles['VNNormal']))
        elements.append(Paragraph(f"Số tiền bằng chữ: {to_words_vnd(row.get('thanh_tien', 0))}", styles['VNNormal']))
        elements.append(Spacer(1, 20))

        sign_table = Table([
            ["Người lập bảng kê\n(Ký, ghi rõ họ tên)", "", "Thủ trưởng đơn vị\n(Ký, đóng dấu)"]
        ], colWidths=[70*mm, 30*mm, 70*mm])
        sign_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        elements.append(sign_table)

        doc.build(elements)
        buffer.seek(0)
        pdf_bytes = buffer.getvalue()
        logger.info("PDF built, size=%d bytes", len(pdf_bytes))
        return pdf_bytes
    except Exception as e:
        logger.exception("build_pdf_bytes_from_row error: %s", e)
        return None

# ---------------- HTML builder ----------------
def build_html_bytes_from_row(row: Dict[str, Any]) -> bytes:
    style = """
<style>
@page { size: A4; margin:20mm; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; color:#111; margin:0; padding:0; }
.container { width:210mm; padding:12mm; box-sizing:border-box; }
.header { display:flex; justify-content:space-between; align-items:flex-start; }
.h-title { text-align:center; margin-top:6px; }
.small { color:#555; font-size:12px; }
.table { width:100%; border-collapse:collapse; margin-top:12px; }
.table th, .table td { border:1px solid #ddd; padding:8px; font-size:13px; }
.table thead th { background:#f4f7fb; font-weight:600; text-align:center; }
.right { text-align:right; }
.sig { display:flex; justify-content:space-between; margin-top:40px; }
.signbox { width:30%; text-align:center; }
</style>
"""
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Bảng kê 01/TNDN</title>
{style}
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <div><strong>CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM</strong></div>
      <div class="small">Độc lập - Tự do - Hạnh phúc</div>
    </div>
    <div class="small">Mẫu số: 01/TNDN<br><em>(Ban hành kèm theo Thông tư 78/2014/TT-BTC)</em></div>
  </div>

  <h2 class="h-title">BẢNG KÊ THU MUA HÀNG HÓA, DỊCH VỤ MUA VÀO KHÔNG CÓ HÓA ĐƠN</h2>

  {"<p><strong>Đơn vị:</strong> " + row.get('don_vi','') + " &nbsp;&nbsp; <strong>MST:</strong> " + row.get('mst','') + "</p>" if row.get('don_vi') else ""}
  <p class="small"><strong>Địa điểm thu mua:</strong> {row.get('dia_diem','')} &nbsp;&nbsp; <strong>Người phụ trách:</strong> {row.get('phu_trach','')}</p>
  <p class="small"><strong>Ngày lập:</strong> {row.get('ngay_lap','')}</p>

  <h4>Thông tin người bán</h4>
  <p><strong>Họ và tên:</strong> {row.get('ho_va_ten','')}<br>
  <strong>Số CCCD:</strong> {row.get('so_cccd','')}<br>
  <strong>Quê quán:</strong> {row.get('que_quan','')}</p>

  <table class="table" role="table" aria-label="Chi tiết giao dịch">
    <thead>
      <tr>
        <th>STT</th><th>Tên hàng/dịch vụ</th><th>ĐVT</th><th>Số lượng</th><th>Đơn giá (VNĐ)</th><th>Thành tiền (VNĐ)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="right">1</td>
        <td>{row.get('mieu_ta','Hàng hóa')}</td>
        <td class="right">{row.get('don_vi_unit','')}</td>
        <td class="right">{row.get('so_luong','')}</td>
        <td class="right">{fmt_money(row.get('don_gia',0))}</td>
        <td class="right">{fmt_money(row.get('thanh_tien',0))}</td>
      </tr>
    </tbody>
  </table>

  <p class="right"><strong>Tổng cộng: {fmt_money(row.get('thanh_tien',0))} VNĐ</strong></p>
  <p><strong>Số tiền bằng chữ:</strong> {to_words_vnd(row.get('thanh_tien',0))}</p>

  <div class="sig">
    <div class="signbox">Người lập bảng kê<br/>(Ký, ghi rõ họ tên)</div>
    <div class="signbox">Người bán<br/>(Ký, ghi rõ họ tên)</div>
    <div class="signbox">Thủ trưởng đơn vị<br/>(Ký, đóng dấu)</div>
  </div>

</div>
</body>
</html>
"""
    return html.encode('utf-8')

# ---------------- Core process & save ----------------
def process_transaction_and_build(ho_va_ten: str, so_cccd: str, que_quan: str,
                                  so_luong_str: str, don_gia_str: str,
                                  don_vi_unit: str = "chỉ", mieu_ta: str = "Hàng hóa",
                                  don_vi_name: str = "", mst: str = "", dia_chi: str = "",
                                  dia_diem: str = DEFAULT_DIA_DIEM, phu_trach: str = "") -> Tuple[Dict[str, Any], Optional[bytes], bytes]:
    if not ho_va_ten or ho_va_ten.strip() == "":
        raise ValueError("Họ và tên không được để trống.")
    if not so_luong_str or so_luong_str.strip() == "":
        raise ValueError("Khối lượng không được để trống.")
    if not don_gia_str or don_gia_str.strip() == "":
        raise ValueError("Đơn giá không được để trống.")

    so_luong = safe_float_from_str(so_luong_str)
    don_gia = safe_float_from_str(don_gia_str)
    if so_luong <= 0:
        raise ValueError("Khối lượng phải > 0.")
    if don_gia <= 0:
        raise ValueError("Đơn giá phải > 0.")

    thanh_tien = so_luong * don_gia
    ngay_display = now_local_str("%d/%m/%Y")
    time_iso = now_iso()

    row_csv = {
        "Thời gian": time_iso,
        "Đơn vị bán hàng": don_vi_name,
        "MST": mst,
        "Địa chỉ đơn vị": dia_chi,
        "Địa điểm thu mua": dia_diem,
        "Người phụ trách": phu_trach,
        "Họ và Tên": ho_va_ten,
        "Số CCCD": str(so_cccd),
        "Quê quán": que_quan,
        "Khối lượng": so_luong,
        "Đơn vị tính": don_vi_unit,
        "Đơn giá": don_gia,
        "Thành tiền": thanh_tien
    }
    append_history_row(row_csv)

    row = {
        "don_vi": don_vi_name,
        "mst": mst,
        "dia_chi": dia_chi,
        "dia_diem": dia_diem,
        "phu_trach": phu_trach,
        "ngay_lap": ngay_display,
        "ho_va_ten": ho_va_ten,
        "so_cccd": str(so_cccd),
        "que_quan": que_quan,
        "so_luong": so_luong,
        "don_gia": don_gia,
        "thanh_tien": thanh_tien,
        "don_vi_unit": don_vi_unit,
        "mieu_ta": mieu_ta
    }

    pdf_bytes = None
    try:
        pdf_bytes = build_pdf_bytes_from_row(row)
    except Exception as e:
        logger.exception("Error building PDF: %s", e)
        pdf_bytes = None

    html_bytes = build_html_bytes_from_row(row)
    return row, pdf_bytes, html_bytes

# ---------------- UI ----------------
st.title(APP_TITLE)
st.markdown("Chụp/tải ảnh CCCD và cân → app tự OCR → tạo bản kê mẫu 01/TNDN → preview & tải PDF/HTML. Kiểm tra OCR trước khi in.")

top_cols = st.columns([1,3,1])
with top_cols[0]:
    if st.button("Làm mới"):
        st.rerun()
with top_cols[1]:
    st.markdown("**Hướng dẫn:** Chụp CCCD bằng camera hoặc tải ảnh; chụp màn hình cân; chỉnh nếu OCR chưa chính xác; bấm `Tạo bản kê`.")
with top_cols[2]:
    if REPORTLAB_OK:
        st.success("ReportLab: available — PDF sẽ được tạo")
    else:
        st.warning("ReportLab: not available — PDF không thể tạo, sử dụng HTML")

with st.expander("Thông tin đơn vị (tuỳ chọn)"):
    don_vi_name = st.text_input("Tên đơn vị")
    mst = st.text_input("Mã số thuế (MST)")
    dia_chi = st.text_input("Địa chỉ đơn vị")
    dia_diem = st.text_input("Địa điểm thu mua", value=DEFAULT_DIA_DIEM)
    phu_trach = st.text_input("Người phụ trách thu mua")

if 'ho_ten' not in st.session_state: st.session_state.ho_ten = ""
if 'so_cccd' not in st.session_state: st.session_state.so_cccd = ""
if 'que_quan' not in st.session_state: st.session_state.que_quan = ""
if 'so_luong' not in st.session_state: st.session_state.so_luong = ""

st.header("1) Thông tin người bán (CCCD)")
col1, col2 = st.columns(2)
with col1:
    st.subheader("OCR CCCD — chụp hoặc tải ảnh")
    up_cccd = st.file_uploader("Tải ảnh CCCD (JPG/PNG)", type=['jpg','jpeg','png'])
    cam_cccd = st.camera_input("Hoặc chụp bằng camera")
    chosen_cccd = cam_cccd if cam_cccd is not None else up_cccd
    if chosen_cccd:
        try:
            ho, so, que = trich_xuat_cccd_from_uploaded(chosen_cccd)
            if ho:
                st.session_state.ho_ten = ho
            if so:
                st.session_state.so_cccd = so
            if que:
                st.session_state.que_quan = que
            st.success("Đã trích xuất (kiểm tra và chỉnh nếu cần).")
        except Exception as e:
            logger.exception("OCR CCCD error: %s", e)
            st.error("Lỗi OCR CCCD: " + str(e))
with col2:
    st.subheader("Nhập/chỉnh tay")
    ho_va_ten = st.text_input("Họ và tên", value=st.session_state.ho_ten)
    so_cccd = st.text_input("Số CCCD/CMND (chuỗi)", value=st.session_state.so_cccd)
    que_quan = st.text_input("Quê quán", value=st.session_state.que_quan)

st.markdown("---")
st.header("2) Thông tin giao dịch (cân)")
col3, col4 = st.columns(2)
with col3:
    st.subheader("OCR cân — chụp/tải ảnh màn hình cân")
    up_can = st.file_uploader("Tải ảnh cân (JPG/PNG)", type=['jpg','jpeg','png'], key="up_can")
    cam_can = st.camera_input("Hoặc chụp cân bằng camera", key="cam_can")
    chosen_can = cam_can if cam_can is not None else up_can
    if chosen_can:
        try:
            so_luong_ex = trich_xuat_can_from_uploaded(chosen_can)
            if so_luong_ex:
                st.session_state.so_luong = so_luong_ex
            st.success("Đã trích xuất khối lượng (kiểm tra và chỉnh nếu cần).")
        except Exception as e:
            logger.exception("OCR cân error: %s", e)
            st.error("Lỗi OCR cân: " + str(e))
with col4:
    st.subheader("Nhập/chỉnh tay")
    so_luong_input = st.text_input("Khối lượng", value=str(st.session_state.so_luong))
    don_gia_input = st.text_input("Đơn giá (VNĐ)", value="1000000")
    don_vi_unit = st.text_input("Đơn vị tính (ví dụ: chỉ, kg)", value="chỉ")
    mieu_ta = st.text_input("Mô tả hàng (VD: Vàng miếng...)", value="Hàng hóa")

st.markdown("---")
st.header("3) Tạo bản kê, Preview & Tải")
colA, colB = st.columns([2,1])
with colA:
    if st.button("Tạo bản kê (Tính & Lưu)"):
        try:
            ho_final = ho_va_ten.strip() if ho_va_ten and ho_va_ten.strip() != "" else st.session_state.ho_ten
            so_cccd_final = so_cccd.strip() if so_cccd and so_cccd.strip() != "" else st.session_state.so_cccd
            que_quan_final = que_quan.strip() if que_quan and que_quan.strip() != "" else st.session_state.que_quan

            row, pdf_bytes, html_bytes = process_transaction_and_build(
                ho_va_ten=ho_final,
                so_cccd=so_cccd_final,
                que_quan=que_quan_final,
                so_luong_str=so_luong_input,
                don_gia_str=don_gia_input,
                don_vi_unit=don_vi_unit,
                mieu_ta=mieu_ta,
                don_vi_name=don_vi_name,
                mst=mst,
                dia_chi=dia_chi,
                dia_diem=dia_diem,
                phu_trach=phu_trach
            )
            st.success("Đã lưu giao dịch vào lịch sử.")
            if pdf_bytes:
                st.markdown("**Xem trước PDF (nếu trình duyệt hỗ trợ):**", unsafe_allow_html=True)
                try:
                    b64 = base64.b64encode(pdf_bytes).decode('utf-8')
                    embed_html = f'<embed src="data:application/pdf;base64,{b64}" width="100%" height="700px" type="application/pdf">'
                    st.components.v1.html(embed_html, height=720)
                except Exception as e:
                    logger.warning("PDF embed failed: %s", e)
                    st.info("Trình duyệt chặn nhúng PDF. Vui lòng tải PDF xuống.")
                st.download_button("📥 Tải PDF (A4)", data=pdf_bytes,
                                   file_name=f"bangke_01_TNDN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                   mime="application/pdf")
            else:
                st.warning("PDF không tạo được — hiển thị HTML (in bằng trình duyệt).")
                st.components.v1.html(html_bytes.decode('utf-8'), height=720)
                st.download_button("📥 Tải HTML", data=html_bytes,
                                   file_name=f"bangke_01_TNDN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                   mime="text/html")
            st.metric("Thành tiền (VNĐ)", fmt_money(row.get('thanh_tien', 0)))
            st.write("Số tiền bằng chữ:", to_words_vnd(row.get('thanh_tien', 0)))
        except Exception as e:
            logger.exception("Create error: %s", e)
            st.error("Lỗi khi tạo bản kê: " + str(e))
with colB:
    st.info("Ghi chú:\n- Nếu PDF hiển thị dấu lạ, hãy đặt DejaVuSans.ttf vào folder và chạy lại.\n- Nếu trình duyệt chặn iframe, tải file PDF/HTML rồi mở bằng trình xem file cục bộ.")

st.markdown("---")
st.header("4) Lịch sử giao dịch")
try:
    df_hist = read_history_df()
    if df_hist.empty:
        st.info("Chưa có giao dịch.")
    else:
        st.dataframe(df_hist.sort_values("Thời gian", ascending=False).head(500))
        st.download_button("Tải lịch sử (CSV)", data=df_hist.to_csv(index=False).encode('utf-8'),
                           file_name="lich_su_giao_dich.csv", mime="text/csv")
except Exception as e:
    logger.exception("History display error: %s", e)
    st.error("Không thể đọc lịch sử: " + str(e))

st.markdown("---")
st.caption("Nếu muốn mình thêm: logo trên PDF, nhiều dòng hàng, xuất Excel, lưu vào DB (SQLite/Postgres), hoặc UI nâng cao — mình làm tiếp.")
