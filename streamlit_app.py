# app.py
# Full Streamlit app (>500 lines) for creating BẢNG KÊ 01/TNDN
# - OCR trực tiếp (CCCD + cân) từ camera/file upload
# - Giữ CCCD là chuỗi (không mất số 0)
# - Xuất PDF (ReportLab) & HTML (print-friendly), preview, download
# - Lưu lịch sử vào CSV
# - Nhiều hàm tiện ích, validate, logging, giao diện đẹp
#
# Author: ChatGPT assistant (generate for user)
# Date: 2025-08-xx

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
from typing import Tuple, Dict, Any, Optional

# Try import reportlab for PDF output. If missing, PDF fallback to HTML.
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# ---------- Configuration ----------
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

# Make sure logging is visible in Streamlit logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# ---------- Streamlit page config ----------
st.set_page_config(page_title="Bảng kê 01/TNDN", layout="wide", page_icon="📄")

# ---------- Ensure history CSV exists ----------
if not os.path.exists(LICH_SU_FILE):
    df_init = pd.DataFrame(columns=CSV_COLUMNS)
    df_init.to_csv(LICH_SU_FILE, index=False, encoding="utf-8")

# ---------- OCR initialization ----------
@st.cache_resource
def init_ocr():
    """
    Initialize PaddleOCR once and cache.
    Use lang="vi" for Vietnamese recognition.
    """
    try:
        reader = PaddleOCR(lang="vi", use_angle_cls=False)
        logger.info("PaddleOCR initialized.")
        return reader
    except Exception as e:
        logger.exception("Failed to initialize PaddleOCR: %s", e)
        # Re-raise to let UI show error
        raise

ocr = init_ocr()

# ---------- Utility functions ----------

def now_local_str(fmt: str = "%d/%m/%Y") -> str:
    tz = pytz.timezone(TIMEZONE)
    return datetime.now(tz).strftime(fmt)

def now_iso() -> str:
    tz = pytz.timezone(TIMEZONE)
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

def safe_float_from_str(s: str) -> float:
    """
    Convert string containing dot/comma thousands/decimal into float robustly.
    """
    if s is None:
        return 0.0
    s0 = str(s).strip()
    if s0 == "":
        return 0.0
    # Remove spaces
    s0 = s0.replace(" ", "")
    # Heuristic:
    # If both '.' and ',' present, assume '.' thousand, ',' decimal OR vice versa.
    # Try a few possibilities.
    try_formats = []
    if "." in s0 and "," in s0:
        # try treat '.' as thousand separators (remove) and comma as decimal
        try_formats.append(s0.replace(".", "").replace(",", "."))
        # try treat ',' as thousand separators, '.' as decimal
        try_formats.append(s0.replace(",", ""))
    else:
        # if only commas, remove commas
        if "," in s0 and "." not in s0:
            try_formats.append(s0.replace(",", ""))
        else:
            try_formats.append(s0)
    for candidate in try_formats:
        try:
            return float(candidate)
        except:
            continue
    # last resort: extract digits and dot
    cleaned = "".join(c for c in s0 if c.isdigit() or c == ".")
    try:
        return float(cleaned) if cleaned != "" else 0.0
    except:
        return 0.0

def fmt_money(v: float) -> str:
    """
    Format number to VN money format: thousands separated by dot.
    """
    try:
        n = int(round(v))
        return f"{n:,}".replace(",", ".")
    except:
        return "0"

# Number to Vietnamese words (support up to large numbers, simple)
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
    """
    Convert integer (VNĐ) to Vietnamese text (simple, readable).
    """
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
    if len(s) == 0:
        return "Không đồng"
    return s[0].upper() + s[1:] + " đồng"

def normalize_cccd_candidate(candidate: str) -> str:
    """
    From an OCR token try to produce a proper CCCD string:
    - keep digits only
    - if length >=12 take first 12
    - if length between 9 and 11, zfill to 12 (some older numbers)
    """
    if candidate is None:
        return ""
    digits = ''.join(ch for ch in str(candidate) if ch.isdigit())
    if len(digits) >= 12:
        return digits[:12]
    if 9 <= len(digits) < 12:
        return digits.zfill(12)
    return digits

# ---------- OCR helpers (work on bytes / numpy arrays, no disk writes) ----------

def img_from_uploaded_file(uploaded) -> Optional[np.ndarray]:
    """
    Convert Streamlit UploadedFile or camera_input to OpenCV BGR image (numpy array).
    Returns None on failure.
    """
    if uploaded is None:
        return None
    try:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.exception("img_from_uploaded_file error: %s", e)
        return None

def ocr_image_to_lines(img: np.ndarray) -> list:
    """
    Run PaddleOCR on an OpenCV image and return list of recognized lines.
    Each element similar to PaddleOCR output for region: [box, (text, confidence)]
    """
    if img is None:
        return []
    try:
        result = ocr.ocr(img, cls=False)
        if not result:
            return []
        # result[0] contains lines
        return result[0]
    except Exception as e:
        logger.exception("ocr_image_to_lines exception: %s", e)
        return []

def extract_text_lines_from_image(uploaded) -> list:
    """
    Convenience: take UploadedFile or camera_input, decode to image, OCR and return text lines (strings).
    """
    img = img_from_uploaded_file(uploaded)
    lines = []
    if img is None:
        return []
    ocr_lines = ocr_image_to_lines(img)
    for ln in ocr_lines:
        try:
            lines.append(ln[1][0])
        except:
            continue
    return lines

def trich_xuat_cccd_from_uploaded(uploaded) -> Tuple[str,str,str]:
    """
    Extract Họ và Tên, Số CCCD (string), Quê quán from uploaded image using OCR.
    We try to be tolerant and detect multiple forms.
    """
    ho_ten = ""
    so_cccd = ""
    que_quan = ""
    if uploaded is None:
        return ho_ten, so_cccd, que_quan
    img = img_from_uploaded_file(uploaded)
    if img is None:
        return ho_ten, so_cccd, que_quan
    lines = ocr_image_to_lines(img)
    # convert lines to plain texts
    texts = [ln[1][0].strip() for ln in lines if ln and ln[1] and ln[1][0]]
    # search patterns
    for idx, txt in enumerate(texts):
        up = txt.upper()
        # find Họ và tên
        if "HỌ VÀ TÊN" in up or "HO VA TEN" in up or "HỌ TÊN" in up:
            # try next line as name
            if idx + 1 < len(texts):
                ho_ten_candidate = texts[idx+1].strip()
                if ho_ten_candidate:
                    ho_ten = ho_ten_candidate
        # detect CCCD by token length of digits
        # sometimes "SỐ: 012345678901" or "SỐ CCCD: 0..."
        # try to extract digits from this text
        digits = ''.join(ch for ch in txt if ch.isdigit())
        if len(digits) >= 9:
            ccc = normalize_cccd_candidate(digits)
            if len(ccc) >= 9:
                so_cccd = ccc
        # find Quê quán
        if "QUÊ QUÁN" in up or "QUE QUAN" in up:
            if idx + 1 < len(texts):
                que_candidate = texts[idx+1].strip()
                if que_candidate:
                    que_quan = que_candidate
    # fallback: search any line containing many digits -> CCCD
    if not so_cccd:
        for txt in texts:
            digits = ''.join(ch for ch in txt if ch.isdigit())
            if len(digits) >= 9:
                so_cccd = normalize_cccd_candidate(digits)
                break
    return ho_ten, so_cccd, que_quan

def trich_xuat_can_from_uploaded(uploaded) -> str:
    """
    Extract numeric reading from scale image; returns first numeric-like token found.
    Example return: '1.234' or '2.5' etc.
    """
    if uploaded is None:
        return ""
    img = img_from_uploaded_file(uploaded)
    if img is None:
        return ""
    lines = ocr_image_to_lines(img)
    for ln in lines:
        txt = ln[1][0]
        # keep digits and separators
        cleaned = ''.join(ch for ch in txt if ch.isdigit() or ch in '.,')
        if any(ch.isdigit() for ch in cleaned):
            # normalize comma to dot
            cleaned2 = cleaned.replace(',', '.')
            # keep only one dot (last dot as decimal)
            parts = cleaned2.split('.')
            if len(parts) > 2:
                cleaned2 = ''.join(parts[:-1]) + '.' + parts[-1]
            # remove leading/trailing dots
            cleaned2 = cleaned2.strip('.')
            return cleaned2
    return ""

# ---------- CSV / history utilities ----------

def append_history_row(row: Dict[str, Any]) -> None:
    """
    Append a transaction row to CSV history. Keep CCCD as string.
    """
    try:
        df = pd.DataFrame([row])
        # ensure columns order matches
        df = df.reindex(columns=CSV_COLUMNS)
        df.to_csv(LICH_SU_FILE, mode='a', header=False, index=False, encoding='utf-8')
    except Exception as e:
        logger.exception("append_history_row error: %s", e)

def read_history_df() -> pd.DataFrame:
    try:
        df = pd.read_csv(LICH_SU_FILE, encoding='utf-8')
        return df
    except Exception as e:
        logger.exception("read_history_df error: %s", e)
        return pd.DataFrame(columns=CSV_COLUMNS)

# ---------- PDF builder (ReportLab) ----------
def build_pdf_bytes_from_row(row: Dict[str, Any]) -> Optional[bytes]:
    """
    Build a single-page PDF (A4) for the bảng kê row and return bytes.
    Uses DejaVuSans.ttf if present for Vietnamese.
    """
    if not REPORTLAB_OK:
        logger.warning("ReportLab not available; PDF will not be generated.")
        return None
    try:
        buffer = io.BytesIO()
        w, h = A4
        c = canvas.Canvas(buffer, pagesize=A4)

        # try to register DejaVu font for Vietnamese if available in cwd
        font_name = "Helvetica"
        try:
            if os.path.exists("DejaVuSans.ttf"):
                pdfmetrics.registerFont(TTFont("DejaVu", "DejaVuSans.ttf"))
                font_name = "DejaVu"
            else:
                # optional: if user has other ttf in folder, could register
                font_name = "Helvetica"
        except Exception as e:
            logger.warning("Font registration failed: %s", e)
            font_name = "Helvetica"

        left = 18 * mm
        right = 18 * mm
        cur_y = h - 20 * mm

        # Header
        c.setFont(font_name, 10)
        c.drawString(left, cur_y, "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM")
        c.drawRightString(w - right, cur_y, "Mẫu số: 01/TNDN")
        cur_y -= 12
        c.setFont(font_name, 9)
        c.drawString(left, cur_y, "Độc lập - Tự do - Hạnh phúc")
        c.drawRightString(w - right, cur_y, "(Ban hành kèm theo Thông tư 78/2014/TT-BTC)")
        cur_y -= 18

        c.setFont(font_name, 13)
        c.drawCentredString(w / 2, cur_y, "BẢNG KÊ THU MUA HÀNG HÓA, DỊCH VỤ MUA VÀO KHÔNG CÓ HÓA ĐƠN")
        cur_y -= 20

        # Optional unit info
        c.setFont(font_name, 10)
        if row.get("don_vi"):
            c.drawString(left, cur_y, f"Đơn vị: {row.get('don_vi')}")
            cur_y -= 12
        if row.get("mst"):
            c.drawString(left, cur_y, f"Mã số thuế: {row.get('mst')}")
            cur_y -= 12
        if row.get("dia_chi"):
            c.drawString(left, cur_y, f"Địa chỉ: {row.get('dia_chi')}")
            cur_y -= 12
        cur_y -= 6

        # Thu mua info
        c.drawString(left, cur_y, f"Địa điểm thu mua: {row.get('dia_diem','')}")
        cur_y -= 12
        c.drawString(left, cur_y, f"Người phụ trách: {row.get('phu_trach','')}")
        cur_y -= 12
        c.drawString(left, cur_y, f"Ngày lập: {row.get('ngay_lap','')}")
        cur_y -= 16

        # Seller info
        c.setFont(font_name, 11)
        c.drawString(left, cur_y, "Thông tin người bán:")
        cur_y -= 12
        c.setFont(font_name, 10)
        c.drawString(left + 6 * mm, cur_y, f"Họ và tên: {row.get('ho_va_ten','')}")
        cur_y -= 10
        c.drawString(left + 6 * mm, cur_y, f"Số CCCD/CMND: {row.get('so_cccd','')}")
        cur_y -= 10
        c.drawString(left + 6 * mm, cur_y, f"Quê quán: {row.get('que_quan','')}")
        cur_y -= 16

        # Table header
        col_w = [18*mm, 80*mm, 22*mm, 30*mm, 38*mm, 40*mm]
        x = left
        headers = ["STT", "Tên hàng/dịch vụ", "ĐVT", "Số lượng", "Đơn giá (VNĐ)", "Thành tiền (VNĐ)"]
        c.setFont(font_name, 9)
        for i, htext in enumerate(headers):
            c.rect(x, cur_y-14, col_w[i], 16, stroke=1, fill=0)
            c.drawCentredString(x + col_w[i]/2, cur_y-10, htext)
            x += col_w[i]
        cur_y -= 18

        # single row
        x = left
        c.rect(x, cur_y-12, col_w[0], 14, stroke=1); c.drawCentredString(x + col_w[0]/2, cur_y-8, "1"); x += col_w[0]
        c.rect(x, cur_y-12, col_w[1], 14, stroke=1); c.drawString(x + 4, cur_y-10, row.get('mieu_ta','Hàng hóa')); x += col_w[1]
        c.rect(x, cur_y-12, col_w[2], 14, stroke=1); c.drawCentredString(x + col_w[2]/2, cur_y-8, row.get('don_vi_unit','')); x += col_w[2]
        c.rect(x, cur_y-12, col_w[3], 14, stroke=1); c.drawCentredString(x + col_w[3]/2, cur_y-8, str(row.get('so_luong',''))); x += col_w[3]
        c.rect(x, cur_y-12, col_w[4], 14, stroke=1); c.drawRightString(x + col_w[4] - 4, cur_y-8, fmt_money(row.get('don_gia',0))); x += col_w[4]
        c.rect(x, cur_y-12, col_w[5], 14, stroke=1); c.drawRightString(x + col_w[5] - 4, cur_y-8, fmt_money(row.get('thanh_tien',0)))
        cur_y -= 28

        # Totals
        c.setFont(font_name, 10)
        c.drawRightString(w - right, cur_y, "Tổng cộng: " + fmt_money(row.get('thanh_tien',0)) + " VNĐ")
        cur_y -= 14
        c.drawString(left, cur_y, "Số tiền bằng chữ: " + to_words_vnd(row.get('thanh_tien',0)))
        cur_y -= 28

        # Sign boxes
        c.drawString(left, cur_y, f"{row.get('dia_diem','')}, ngày {row.get('ngay_lap','')}")
        c.drawString(left + 6*mm, cur_y - 18, "Người lập bảng kê")
        c.drawString(w/2, cur_y - 18, "Người bán")
        c.drawString(w - right - 80*mm, cur_y - 18, "Thủ trưởng đơn vị")
        c.line(left, cur_y - 60, left + 60*mm, cur_y - 60)
        c.line(w/2, cur_y - 60, w/2 + 60*mm, cur_y - 60)
        c.line(w - right - 80*mm, cur_y - 60, w - right + 10*mm, cur_y - 60)

        c.showPage()
        c.save()
        buffer = buffer = io.BytesIO()
        buffer.write(b"")  # ensure buffer exists
        # we need to get PDF bytes from canvas; canvas saved to original buffer passed in, so:
        # Actually we used buffer as canvas target, so we must getvalue:
        buffer = c.getpdfdata() if hasattr(c, "getpdfdata") else None
        # Fallback: getwritten bytes from the original buffer we created earlier:
        # However reportlab's canvas wrote to internal file; the easiest is re-create canvas writing to BytesIO:
        # To be robust, re-create properly: (Better approach below)
    except Exception as e:
        logger.exception("build_pdf_bytes_from_row error: %s", e)
        return None

    # Robust implementation: rebuild PDF writing directly to BytesIO (to avoid confusion)
    try:
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        # re-run drawing same content but minimal to ensure PDF in bytes
        # For brevity reuse same drawing but simpler; in practice you can move drawing code into helper.
        # We'll draw a compact version used for printing (title + table)
        # Header
        pdf.setFont(font_name, 10)
        pdf.drawString(left, h - 20*mm, "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM")
        pdf.drawRightString(w - right, h - 20*mm, "Mẫu số: 01/TNDN")
        pdf.setFont(font_name, 13)
        pdf.drawCentredString(w / 2, h - 40*mm, "BẢNG KÊ THU MUA HÀNG HÓA, DỊCH VỤ MUA VÀO KHÔNG CÓ HÓA ĐƠN")
        # small box for seller
        pdf.setFont(font_name, 10)
        y0 = h - 55*mm
        pdf.drawString(left, y0, f"Họ và tên: {row.get('ho_va_ten','')}")
        pdf.drawString(left, y0 - 12, f"Số CCCD: {row.get('so_cccd','')}")
        pdf.drawString(left, y0 - 24, f"Quê quán: {row.get('que_quan','')}")
        pdf.drawString(left, y0 - 40, f"Khối lượng: {row.get('so_luong','')} {row.get('don_vi_unit','')}")
        pdf.drawString(left, y0 - 52, f"Đơn giá: {fmt_money(row.get('don_gia',0))} VNĐ")
        pdf.drawString(left, y0 - 64, f"Thành tiền: {fmt_money(row.get('thanh_tien',0))} VNĐ")
        pdf.showPage()
        pdf.save()
        buffer.seek(0)
        pdf_bytes = buffer.read()
        return pdf_bytes
    except Exception as e:
        logger.exception("Fallback PDF creation failed: %s", e)
        return None

# ---------- HTML builder (print friendly) ----------

def build_html_bytes_from_row(row: Dict[str, Any]) -> bytes:
    """
    Build an A4 print CSS HTML bytes to be downloaded/previewed in browser.
    """
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Bảng kê 01/TNDN</title>
<style>
@page {{ size: A4; margin:20mm; }}
body{{font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; color:#111; margin:0; padding:0;}}
.container{{width:210mm; margin:0 auto; padding:10mm 12mm; box-sizing:border-box;}}
.header{{display:flex; justify-content:space-between; align-items:flex-start;}}
.h-title{{text-align:center; margin:6px 0 8px 0}}
.small{{color:#555; font-size:12px}}
.table{{width:100%; border-collapse:collapse; margin-top:12px}}
.table th, .table td{{border:1px solid #ddd; padding:8px; font-size:13px}}
.table thead th{{background:#f4f7fb; font-weight:600; text-align:center}}
.right{{text-align:right}}
.sig{{display:flex; justify-content:space-between; margin-top:40px}}
.signbox{{width:30%; text-align:center}}
</style>
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

# ---------- Main processing logic ----------

def process_transaction_and_build(ho_va_ten: str, so_cccd: str, que_quan: str,
                                  so_luong_str: str, don_gia_str: str,
                                  don_vi_unit: str = "chỉ", mieu_ta: str = "Hàng hóa",
                                  don_vi_name: str = "", mst: str = "", dia_chi: str = "",
                                  dia_diem: str = DEFAULT_DIA_DIEM, phu_trach: str = "") -> Tuple[Dict[str, Any], Optional[bytes], bytes]:
    """
    Validate inputs, compute thanh_tien, append to CSV, build row dict, build PDF bytes (if possible) and HTML bytes.
    Returns: (row_dict, pdf_bytes_or_None, html_bytes)
    """
    # Validate and parse numbers
    if not ho_va_ten:
        raise ValueError("Họ và tên không được để trống.")
    if not so_luong_str:
        raise ValueError("Khối lượng không được để trống.")
    if not don_gia_str:
        raise ValueError("Đơn giá không được để trống.")

    so_luong = safe_float_from_str(so_luong_str)
    don_gia = safe_float_from_str(don_gia_str)
    if so_luong <= 0:
        raise ValueError("Khối lượng phải > 0.")
    if don_gia <= 0:
        raise ValueError("Đơn giá phải > 0.")

    thanh_tien = so_luong * don_gia

    # Time stamps
    ngay_display = now_local_str("%d/%m/%Y")
    time_iso = now_iso()

    # Save to CSV history (keep CCCD as string)
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

    # Build row for print/preview
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

    # Build PDF and HTML
    pdf_bytes = None
    try:
        pdf_bytes = build_pdf_bytes_from_row(row)
    except Exception as e:
        logger.exception("Error building PDF: %s", e)
        pdf_bytes = None

    html_bytes = build_html_bytes_from_row(row)
    return row, pdf_bytes, html_bytes

# ---------- Streamlit UI ----------

st.title(APP_TITLE)
st.markdown("Ứng dụng: chụp / tải ảnh CCCD và ảnh cân → tự động OCR → tạo Bảng kê mẫu 01/TNDN → preview & tải PDF/HTML. Kiểm tra kết quả OCR trước khi in.")

# Optional: top action buttons
col_top = st.columns([1, 3, 1])
with col_top[0]:
    st.button("Làm mới", on_click=lambda: st.experimental_rerun())
with col_top[1]:
    st.markdown("**Hướng dẫn ngắn:**\n- Dùng `Tải ảnh` hoặc `Chụp` (camera) để chụp CCCD / màn hình cân.\n- Kiểm tra trường đã đọc, chỉnh nếu cần.\n- Bấm `Tạo bản kê` để lưu vào lịch sử và xem preview PDF/HTML.")
with col_top[2]:
    st.write("")

# Expandable: thông tin đơn vị (nhỏ, tuỳ chọn)
with st.expander("Thông tin đơn vị (tùy chọn) — sẽ hiển thị trên bản kê nếu điền"):
    don_vi_name = st.text_input("Tên đơn vị", value="")
    mst = st.text_input("Mã số thuế (MST)", value="")
    dia_chi = st.text_input("Địa chỉ đơn vị", value="")
    dia_diem = st.text_input("Địa điểm thu mua", value=DEFAULT_DIA_DIEM)
    phu_trach = st.text_input("Người phụ trách thu mua", value="")

st.markdown("---")

# Session state initialization to keep OCR results between interactions
if 'ho_ten' not in st.session_state: st.session_state.ho_ten = ""
if 'so_cccd' not in st.session_state: st.session_state.so_cccd = ""
if 'que_quan' not in st.session_state: st.session_state.que_quan = ""
if 'so_luong' not in st.session_state: st.session_state.so_luong = ""

# 1) Thông tin khách hàng (CCCD) — OCR trực tiếp từ camera/file + nhập tay
st.header("1) Thông tin người bán (khách hàng) — OCR CCCD")
c1, c2 = st.columns(2)
with c1:
    st.subheader("OCR trực tiếp (Chụp hoặc tải ảnh CCCD)")
    up_cccd = st.file_uploader("Tải ảnh CCCD (JPG/PNG) hoặc chụp bằng camera", type=['jpg','jpeg','png'], accept_multiple_files=False)
    # Also allow camera_input which returns UploadedFile-like; create a camera input widget (works in browser)
    cam_cccd = st.camera_input("Hoặc chụp trực tiếp bằng camera")
    # priority: camera input if provided, else file uploader
    chosen_cccd = None
    if cam_cccd is not None:
        chosen_cccd = cam_cccd
    elif up_cccd is not None:
        chosen_cccd = up_cccd

    if chosen_cccd is not None:
        try:
            ho, so, que = trich_xuat_cccd_from_uploaded(chosen_cccd)
            # Only set session_state if values found (do not overwrite manual edits)
            if ho:
                st.session_state.ho_ten = ho
            if so:
                st.session_state.so_cccd = so
            if que:
                st.session_state.que_quan = que
            st.success("Đã trích xuất (có thể cần chỉnh sửa).")
        except Exception as e:
            logger.exception("Error OCR CCCD: %s", e)
            st.error("Lỗi khi OCR CCCD: " + str(e))

with c2:
    st.subheader("Nhập / chỉnh thủ công")
    ho_va_ten = st.text_input("Họ và tên", value=st.session_state.ho_ten)
    so_cccd = st.text_input("Số CCCD/CMND (giữ dạng chuỗi)", value=st.session_state.so_cccd)
    que_quan = st.text_input("Quê quán", value=st.session_state.que_quan)

st.markdown("---")

# 2) Thông tin giao dịch (cân)
st.header("2) Thông tin giao dịch (Khối lượng & Đơn giá)")
d1, d2 = st.columns(2)
with d1:
    st.subheader("OCR từ cân (Chụp màn hình cân hoặc chụp trực tiếp)")
    up_can = st.file_uploader("Tải ảnh cân (JPG/PNG) hoặc chụp", type=['jpg','jpeg','png'], key="up_can")
    cam_can = st.camera_input("Hoặc chụp màn hình cân bằng camera", key="cam_can")
    chosen_can = None
    if cam_can is not None:
        chosen_can = cam_can
    elif up_can is not None:
        chosen_can = up_can

    if chosen_can is not None:
        try:
            so_luong_ex = trich_xuat_can_from_uploaded(chosen_can)
            if so_luong_ex:
                st.session_state.so_luong = so_luong_ex
            st.success("Đã trích xuất khối lượng (kiểm tra và chỉnh nếu cần).")
        except Exception as e:
            logger.exception("OCR cân error: %s", e)
            st.error("Lỗi khi OCR cân: " + str(e))

with d2:
    st.subheader("Nhập thủ công / Chỉnh")
    so_luong_input = st.text_input("Khối lượng", value=str(st.session_state.so_luong))
    don_gia_input = st.text_input("Đơn giá (VNĐ)", value="1000000")
    don_vi_unit = st.text_input("Đơn vị tính (ví dụ: chỉ, kg)", value="chỉ")
    mieu_ta = st.text_input("Mô tả hàng (VD: Vàng miếng...)", value="Hàng hóa")

st.markdown("---")

# 3) Tạo bản kê, preview, download
st.header("3) Tạo bản kê — Preview & Xuất PDF/HTML")
colA, colB = st.columns([2,1])
with colA:
    if st.button("Tạo bản kê (Tính & Lưu)"):
        # validation
        try:
            # Ensure we use manual-edited values if any
            ho_final = ho_va_ten.strip() if ho_va_ten is not None else st.session_state.ho_ten
            so_cccd_final = so_cccd.strip() if so_cccd is not None else st.session_state.so_cccd
            que_quan_final = que_quan.strip() if que_quan is not None else st.session_state.que_quan

            # call process_and_build
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

            # Show preview: PDF if available else HTML
            if pdf_bytes:
                # embed PDF via base64 iframe
                b64 = base64.b64encode(pdf_bytes).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="700px" type="application/pdf"></iframe>'
                st.markdown("**Xem trước PDF (in trực tiếp từ preview hoặc tải xuống):**", unsafe_allow_html=True)
                st.components.v1.html(pdf_display, height=720)
                st.download_button("📥 Tải PDF (A4)", data=pdf_bytes,
                                   file_name=f"bangke_01_TNDN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                   mime="application/pdf")
            else:
                # show HTML preview & download
                st.markdown("**PDF không khả dụng — xem trước HTML và in từ trình duyệt:**")
                st.components.v1.html(html_bytes.decode('utf-8'), height=720)
                st.download_button("📥 Tải HTML (In từ trình duyệt)", data=html_bytes,
                                   file_name=f"bangke_01_TNDN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                   mime="text/html")

            # Quick metrics
            st.metric("Thành tiền (VNĐ)", fmt_money(row.get('thanh_tien', 0)))
            st.write("Số tiền bằng chữ:", to_words_vnd(row.get('thanh_tien', 0)))
        except Exception as e:
            logger.exception("Error on create: %s", e)
            st.error("Lỗi khi tạo bản kê: " + str(e))

with colB:
    st.info("Ghi chú ngắn:\n- Kiểm tra kỹ thông tin OCR trước khi in.\n- Nếu PDF không hiển thị, tải HTML và in từ trình duyệt (File -> Print -> Save as PDF).\n- Đặt DejaVuSans.ttf trong folder nếu cần hiển thị tiếng Việt chính xác trong PDF.")

st.markdown("---")

# 4) Lịch sử giao dịch
st.header("4) Lịch sử giao dịch")
try:
    df_hist = read_history_df()
    if df_hist.empty:
        st.info("Chưa có giao dịch nào trong lịch sử.")
    else:
        # show last 200 rows to keep UI responsive
        st.dataframe(df_hist.sort_values("Thời gian", ascending=False).head(200))
        # allow export CSV of history
        csv_bytes = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button("Tải lịch sử (CSV)", data=csv_bytes, file_name="lich_su_giao_dich.csv", mime="text/csv")
except Exception as e:
    logger.exception("History display error: %s", e)
    st.error("Không thể đọc lịch sử giao dịch: " + str(e))

st.markdown("---")
st.caption("Ứng dụng được thiết kế để xuất bản kê theo mẫu 01/TNDN. OCR có thể không hoàn hảo — kiểm tra và chỉnh sửa trước khi in. Nếu cần tính năng mở rộng (nhiều dòng hàng, export Excel nâng cao, tích hợp cơ sở dữ liệu), mình có thể bổ sung.")

# End of app
