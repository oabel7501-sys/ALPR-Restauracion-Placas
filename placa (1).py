import cv2
import numpy as np
import re
try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import pytesseract
except ImportError:
    pytesseract = None
import sqlite3
import itertools
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

try:
    import rawpy
except ImportError:
    rawpy = None


# ===================================================================
# CONFIGURACIÓN GENERAL
# ===================================================================
DB_NAME = "registro_placas.db"
ALLOWLIST_PLACA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
DEBUG_GUARDAR_IMAGENES = True

# Parámetros base del modelo de restauración.
# Pueden ajustarse en pruebas reales según el desenfoque observado.
PSF_TIPO = "motion"          # "motion" o "defocus"
MOTION_BLUR_LENGTH = 17      # longitud del desenfoque de movimiento, en píxeles
MOTION_BLUR_ANGLE = 0.0      # ángulo del desenfoque en grados
DEFOCUS_RADIUS = 5           # radio del disco para desenfoque óptico
WIENER_K = 0.006             # relación ruido/señal. Menor = más agresivo.
EPSILON_FFT = 1e-8

# Postprocesamiento visual posterior a la deconvolución.
POST_BILATERAL = True
POST_CLAHE = True
POST_SHARPEN = True
SAFE_RESTORE_SELECTION = True  # evita que Wiener agresivo gane si genera artefactos


# ===================================================================
# BASE DE DATOS SQLITE
# ===================================================================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS registros (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            placa TEXT NOT NULL,
            tipo_vehiculo TEXT NOT NULL,
            fecha_hora TEXT NOT NULL,
            ruta_imagen TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def guardar_registro(placa, tipo, ruta_imagen):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute(
        "INSERT INTO registros (placa, tipo_vehiculo, fecha_hora, ruta_imagen) VALUES (?, ?, ?, ?)",
        (placa, tipo, ahora, ruta_imagen),
    )
    conn.commit()
    conn.close()


# ===================================================================
# UTILIDADES DE IMAGEN
# ===================================================================
def asegurar_gris_uint8(img):
    if img is None:
        return None

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


def unsharp_mask(gray, amount=1.2, sigma=1.0):
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharp = cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def aplicar_gamma(gray, gamma=0.70):
    g = asegurar_gris_uint8(gray)
    if g is None or g.size == 0:
        return g
    gamma = max(0.20, float(gamma))
    table = ((np.arange(256) / 255.0) ** gamma * 255.0).astype(np.uint8)
    return cv2.LUT(g, table)


def mejorar_baja_luz(gray):
    g = asegurar_gris_uint8(gray)
    if g is None or g.size == 0:
        return g
    g = aplicar_gamma(g, gamma=0.62)
    clahe = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8))
    g = clahe.apply(g)
    g = unsharp_mask(g, amount=0.55, sigma=1.0)
    return g


def clahe_fuerte(gray):
    g = asegurar_gris_uint8(gray)
    if g is None or g.size == 0:
        return g
    clahe = cv2.createCLAHE(clipLimit=3.8, tileGridSize=(8, 8))
    return clahe.apply(g)


def normalizar_uint8(img):
    if img is None:
        return None
    arr = img.astype(np.float32)
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - mn) / (mx - mn)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def crear_psf_motion(size=17, angle=0.0):
    size = int(max(3, size))
    if size % 2 == 0:
        size += 1

    psf = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    cv2.line(psf, (0, center), (size - 1, center), 1.0, 1)

    if abs(float(angle)) > 1e-6:
        M = cv2.getRotationMatrix2D((center, center), float(angle), 1.0)
        psf = cv2.warpAffine(psf, M, (size, size), flags=cv2.INTER_CUBIC)

    psf_sum = float(np.sum(psf))
    if psf_sum > 0:
        psf /= psf_sum
    else:
        psf[center, center] = 1.0
    return psf


def crear_psf_defocus(radius=5):
    radius = int(max(1, radius))
    size = radius * 2 + 1
    psf = np.zeros((size, size), dtype=np.float32)
    cv2.circle(psf, (radius, radius), radius, 1.0, -1)
    psf_sum = float(np.sum(psf))
    if psf_sum > 0:
        psf /= psf_sum
    else:
        psf[radius, radius] = 1.0
    return psf


def pad_psf_to_image(psf, shape):
    h, w = shape[:2]
    padded = np.zeros((h, w), dtype=np.float32)
    kh, kw = psf.shape[:2]
    padded[:kh, :kw] = psf
    padded = np.fft.ifftshift(padded)
    return padded


def wiener_deconvolution_manual(gray, psf, k=0.006, eps=1e-8, padding=True):
    gray = asegurar_gris_uint8(gray)
    if gray is None or gray.size == 0:
        return gray

    img = gray.astype(np.float32) / 255.0

    # Padding reflectivo para reducir artefactos de borde en FFT.
    if padding:
        ph = max(8, min(40, img.shape[0] // 6))
        pw = max(8, min(40, img.shape[1] // 6))
        img_pad = cv2.copyMakeBorder(img, ph, ph, pw, pw, cv2.BORDER_REFLECT_101)
    else:
        ph = pw = 0
        img_pad = img

    psf_pad = pad_psf_to_image(psf, img_pad.shape)
    G = np.fft.fft2(img_pad)
    H = np.fft.fft2(psf_pad)
    H_conj = np.conj(H)

    F_hat = (H_conj / (np.abs(H) ** 2 + float(k) + float(eps))) * G
    restored = np.real(np.fft.ifft2(F_hat))

    if padding:
        restored = restored[ph:ph + img.shape[0], pw:pw + img.shape[1]]

    restored = np.clip(restored, 0.0, 1.0)
    return (restored * 255.0).astype(np.uint8)


def aplicar_postprocesamiento_visual(gray):
    out = asegurar_gris_uint8(gray)
    if out is None:
        return None

    if POST_BILATERAL:
        out = cv2.bilateralFilter(out, 7, 45, 45)

    if POST_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        out = clahe.apply(out)

    if POST_SHARPEN:
        out = unsharp_mask(out, amount=0.85, sigma=1.0)

    return out





def varianza_laplaciano(gray):
    g = asegurar_gris_uint8(gray)
    if g is None or g.size == 0:
        return 0.0
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def evaluar_binarizacion_inv(binary_inv):
    if binary_inv is None or binary_inv.size == 0:
        return -9999.0

    b = (binary_inv > 0).astype(np.uint8)
    h, w = b.shape[:2]
    if h < 10 or w < 20:
        return -9999.0

    fg_ratio = float(np.mean(b))
    score = 0.0

    # Una placa binarizada correctamente suele tener tinta moderada, no media imagen blanca.
    if 0.035 <= fg_ratio <= 0.34:
        score += 70.0 - abs(fg_ratio - 0.16) * 180.0
    else:
        score -= 120.0 + abs(fg_ratio - 0.16) * 120.0

    # Penalizar rayas que cruzan casi toda la banda.
    row_density = np.mean(b, axis=1)
    col_density = np.mean(b, axis=0)
    heavy_rows = float(np.mean(row_density > 0.62))
    heavy_cols = float(np.mean(col_density > 0.62))
    score -= (heavy_rows + heavy_cols) * 180.0

    # Componentes con tamaño compatible con caracteres.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    valid_cc = 0
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if area < h * w * 0.001:
            continue
        if bh < h * 0.18 or bh > h * 0.95:
            continue
        ratio = bw / float(bh) if bh else 99
        if 0.08 <= ratio <= 1.50:
            valid_cc += 1

    # Placas suelen tener alrededor de 6 caracteres; aceptamos rango amplio.
    if 3 <= valid_cc <= 10:
        score += 45.0
    else:
        score -= abs(valid_cc - 6) * 8.0

    return score


def binarizacion_robusta(gray):
    g = asegurar_gris_uint8(gray)
    if g is None or g.size == 0:
        return g, {}, {"nombre": "vacia", "score": -9999.0}

    # Ligero suavizado para estabilizar umbral, sin destruir bordes.
    suave = cv2.GaussianBlur(g, (3, 3), 0)
    variantes = {}

    _, otsu_inv = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, otsu = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adapt_inv = cv2.adaptiveThreshold(
        suave, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 9
    )
    adapt = cv2.bitwise_not(adapt_inv)

    variantes["otsu_inv"] = otsu_inv
    variantes["otsu"] = otsu
    variantes["adapt_inv"] = adapt_inv
    variantes["adapt"] = adapt

    # Normalizamos todas a texto blanco sobre fondo negro para evaluar y extraer strip.
    candidatos_inv = {
        "otsu_inv": otsu_inv,
        "otsu_to_inv": cv2.bitwise_not(otsu),
        "adapt_inv": adapt_inv,
        "adapt_to_inv": cv2.bitwise_not(adapt),
    }

    kernel = np.ones((2, 2), np.uint8)
    mejor_nombre = None
    mejor_img = None
    mejor_score = -99999.0

    for nombre, b in candidatos_inv.items():
        limpio = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel, iterations=1)
        limpio = limpiar_bordes_componentes_binary_inv(limpio)
        score = evaluar_binarizacion_inv(limpio)
        if score > mejor_score:
            mejor_score = score
            mejor_nombre = nombre
            mejor_img = limpio

    return mejor_img, variantes, {"nombre": mejor_nombre, "score": mejor_score}


def calidad_imagen_para_ocr(gray):
    g = asegurar_gris_uint8(gray)
    if g is None or g.size == 0:
        return -9999.0
    zona = recortar_zona_caracteres(g)
    b, _, info = binarizacion_robusta(zona)
    # Laplaciano ayuda, pero no debe dominar porque rayas también aumentan alta frecuencia.
    lap = varianza_laplaciano(zona)
    lap_bonus = min(30.0, np.log1p(max(lap, 0.0)) * 4.0)
    return float(info.get("score", -9999.0)) + lap_bonus


def extraer_strip_desde_banda_caracteres(binary_inv, margen=8):
    if binary_inv is None or binary_inv.size == 0:
        return None

    banda_limpia = limpiar_bordes_componentes_binary_inv(binary_inv.copy())
    ys, xs = np.where(banda_limpia > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    h, w = banda_limpia.shape[:2]
    bx1 = max(0, int(xs.min()) - margen)
    bx2 = min(w, int(xs.max()) + margen + 1)
    by1 = max(0, int(ys.min()) - margen)
    by2 = min(h, int(ys.max()) + margen + 1)
    if bx2 <= bx1 or by2 <= by1:
        return None

    strip_inv = banda_limpia[by1:by2, bx1:bx2]
    strip = cv2.bitwise_not(strip_inv)
    strip = cv2.copyMakeBorder(strip, 12, 12, 18, 18, cv2.BORDER_CONSTANT, value=255)
    return strip

def resize_por_alto(img, alto_objetivo=120):
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    if h <= 0:
        return img

    escala = alto_objetivo / float(h)
    nuevo_w = max(1, int(w * escala))
    return cv2.resize(img, (nuevo_w, alto_objetivo), interpolation=cv2.INTER_CUBIC)


def preparar_para_easyocr(img, alto_objetivo=128, fondo_blanco=True):
    gray = asegurar_gris_uint8(img)
    if gray is None or gray.size == 0:
        return None

    gray = resize_por_alto(gray, alto_objetivo)
    gray = unsharp_mask(gray, amount=0.8, sigma=1.0)

    # Si se pide fondo blanco y la imagen está invertida, corregimos polaridad.
    if fondo_blanco:
        # En placas/binarizados esperamos texto oscuro y fondo claro.
        if np.mean(gray) < 127:
            gray = cv2.bitwise_not(gray)

    gray = cv2.copyMakeBorder(gray, 20, 20, 30, 30, cv2.BORDER_CONSTANT, value=255)
    return gray


def recortar_zona_caracteres(gray):
    if gray is None or gray.size == 0:
        return gray

    h, w = gray.shape[:2]
    y1 = int(h * 0.30)
    y2 = int(h * 0.88)
    x1 = int(w * 0.03)
    x2 = int(w * 0.97)

    if y2 <= y1 or x2 <= x1:
        return gray

    return gray[y1:y2, x1:x2]


def order_points(pts):
    pts = np.array(pts, dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


# ===================================================================
# VALIDACIÓN ROBUSTA POR CANDIDATOS
# ===================================================================
FORMATOS_PLACA = [
    ("MOTO INVERTIDO", "DDDDDL", lambda s: f"{s[:4]}-{s[4:]}", 45),

    ("MOTO INVERTIDO", "DDDDLL", lambda s: f"{s[:4]}-{s[4:]}", 34),

    # Placas antiguas mixtas como P7-2478 / P6-8150: letra + dígito + 4 dígitos.
    ("MOTO ANTIGUA MIXTA", "LDDDDD", lambda s: f"{s[:2]}-{s[2:]}", 30),

    ("MOTO NORMAL", "LLDDDD", lambda s: f"{s[:2]}-{s[2:]}", 18),
    ("AUTO LIVIANO", "LLLDDD", lambda s: f"{s[:3]}-{s[3:]}", 14),
    ("AUTO NUEVO", "LDLDDD", lambda s: f"{s[:3]}-{s[3:]}", 14),
    ("AUTO ANTIGUO", "LLLDDDD", lambda s: f"{s[:3]}-{s[3:]}", 6),
    ("MOTO ANTIGUA", "LLDDDDD", lambda s: f"{s[:2]}-{s[2:]}", 8),
]

# Confusiones típicas OCR -> número.
LETRA_A_NUM = {
    "O": ["0"],
    "Q": ["0"],
    "D": ["0"],
    "C": ["0"],
    "U": ["0"],
    "I": ["1"],
    "L": ["1"],
    "Z": ["2"],
    "S": ["5", "3"],
    "B": ["8", "3"],
    "G": ["6"],
    "A": ["4"],
    "T": ["7"],
    # En placas borrosas, 9 puede verse como P/R y viceversa.
    "P": ["9"],
    "R": ["9"],
}

# Confusiones típicas OCR -> letra.
NUM_A_LETRA = {
    "0": ["O", "D", "Q", "C"],
    "1": ["I", "L"],
    "2": ["Z"],
    "3": ["B", "S"],
    "4": ["A"],
    "5": ["S"],
    "6": ["G"],
    "7": ["T"],
    "8": ["B"],
    "9": ["P", "R", "B"],  # B solo se favorecerá con reglas contextuales en validación.
}

# Letras visualmente similares.
LETRAS_SIMILARES = {
    "P": ["P", "R", "D"],
    "R": ["R", "P"],
    "D": ["D", "P", "O"],
    "O": ["O", "D", "Q", "C"],
    "Q": ["Q", "O"],
    "C": ["C", "O", "G"],
    "I": ["I", "L", "C"],
    "L": ["L", "I"],
    "E": ["E", "F"],
    "F": ["F", "E"],
    "U": ["U", "O"],
    "B": ["B", "R"],
    "S": ["S"],
    "A": ["A"],
    "G": ["G"],
    "T": ["T"],
}

DIGITOS_SIMILARES = {}

BASURA_PLACA = [
    "PERU", "PERO", "PRU", "PEPU", "REPUBLICA", "DEL", "PLACA", "RODAJE"
]


def limpiar_lectura_ocr(texto):
    texto = str(texto).upper()
    texto = re.sub(r"[^A-Z0-9]", "", texto)

    for basura in BASURA_PLACA:
        texto = texto.replace(basura, "")

    # Algunos OCR devuelven PE antes del número grande.
    if texto.startswith("PE") and len(texto) > 6:
        texto = texto[2:]

    return texto


def extraer_fragmentos(texto):
    texto = limpiar_lectura_ocr(texto)
    fragmentos = []
    for length in (6, 7):
        if len(texto) >= length:
            for i in range(len(texto) - length + 1):
                fragmentos.append(texto[i:i + length])

    if len(texto) in (6, 7):
        fragmentos.insert(0, texto)

    return list(dict.fromkeys(fragmentos))


def opciones_para_char(char_ocr, esperado):
    opciones = {}

    if esperado == "D":
        if char_ocr.isdigit():
            # Mantener el dígito leído tiene mucha prioridad.
            opciones[char_ocr] = max(opciones.get(char_ocr, 0), 18)
            # Las sustituciones entre dígitos quedan casi desactivadas por seguridad.
            for n in DIGITOS_SIMILARES.get(char_ocr, []):
                opciones[n] = max(opciones.get(n, 0), 2)

        if char_ocr in LETRA_A_NUM:
            for n in LETRA_A_NUM[char_ocr]:
                opciones[n] = max(opciones.get(n, 0), 8)

    elif esperado == "L":
        if char_ocr.isalpha():
            for letra in LETRAS_SIMILARES.get(char_ocr, [char_ocr]):
                if letra == char_ocr:
                    opciones[letra] = max(opciones.get(letra, 0), 18)
                else:
                    opciones[letra] = max(opciones.get(letra, 0), 8)

        if char_ocr in NUM_A_LETRA:
            for letra in NUM_A_LETRA[char_ocr]:
                opciones[letra] = max(opciones.get(letra, 0), 7)

    return list(opciones.items())


def generar_candidatos_para_formato(fragmento, mascara):
    if len(fragmento) != len(mascara):
        return []

    opciones_posicion = []
    for char_ocr, esperado in zip(fragmento, mascara):
        opciones = opciones_para_char(char_ocr, esperado)
        if not opciones:
            return []
        opciones_posicion.append(opciones)

    candidatos = []
    for combinacion in itertools.product(*opciones_posicion):
        texto = "".join(item[0] for item in combinacion)
        score_base = sum(item[1] for item in combinacion)
        candidatos.append((texto, score_base))

    return candidatos


def contar_mutaciones(candidato, fragmento):
    return sum(1 for a, b in zip(candidato, fragmento) if a != b)


def peso_fuente_ocr(nombre):
    nombre = str(nombre or "").lower()

    # OCR clásico sobre zona de caracteres: fuente más confiable para placas ya legibles.
    if "tess_zona_lowlight" in nombre or "tess_zona_clahe_fuerte" in nombre:
        return 82
    if "tess_zona_gray" in nombre or "tess_zona_clahe" in nombre or "tess_zona_sharp" in nombre:
        return 78
    if "tess_zona" in nombre:
        return 68

    # Zonas recortadas: más confiables que la imagen completa.
    if "zona_lowlight" in nombre or "zona_clahe_fuerte" in nombre:
        return 74
    if "zona_sharp" in nombre:
        return 70
    if "zona_gray" in nombre or "zona_clahe" in nombre:
        return 68
    if "zona_restaurada" in nombre:
        return 55
    if "zona_otsu" in nombre or "zona_adapt" in nombre:
        return 42

    # El strip/panel 5 es útil, pero no debe imponer una lectura falsa si salió contaminado.
    if "strip" in nombre:
        return 50

    # Imagen completa: baja prioridad porque PERU, tornillos y bordes contaminan.
    if "completa" in nombre:
        return 8

    return 15

def normalizar_lectura_item(item):
    if isinstance(item, dict):
        texto = limpiar_lectura_ocr(item.get("texto", ""))
        fuente = item.get("fuente", "desconocida")
        conf = float(item.get("conf", 1.0) or 0.0)
        peso = float(item.get("peso", peso_fuente_ocr(fuente)))
    else:
        texto = limpiar_lectura_ocr(item)
        fuente = "legacy"
        conf = 1.0
        peso = 15.0

    return texto, fuente, conf, peso


def evaluar_fallback_numerico_5(lecturas_info):
    numericos = []
    for lectura_limpia, info in lecturas_info.items():
        if not (len(lectura_limpia) == 5 and lectura_limpia.isdigit()):
            continue

        fuentes_join = ",".join(sorted(info["fuentes"])).lower()
        if "completa" in fuentes_join or "combinado" in fuentes_join:
            continue

        score = 70.0
        score += float(info.get("peso_max", 0.0))
        score += min(int(info.get("count", 0)), 6) * 14.0
        score += min(max(float(info.get("conf_max", 0.0)), 0.0), 1.0) * 10.0

        if info.get("tiene_strip"):
            score += 18.0
        if "zona_gray" in fuentes_join or "zona_clahe" in fuentes_join or "zona_sharp" in fuentes_join:
            score += 20.0
        if "tess" in fuentes_join:
            score += 15.0

        numericos.append({
            "score": score,
            "placa": lectura_limpia,
            "tipo": "MOTO NUMERICA",
            "candidato": lectura_limpia,
            "fragmento": lectura_limpia,
            "lectura": lectura_limpia,
            "mutaciones": 0,
            "fuentes": ",".join(sorted(info["fuentes"])),
            "count": info["count"],
            "tiene_strip": info["tiene_strip"],
        })

    numericos.sort(key=lambda x: (x["score"], x["count"], 1 if x["tiene_strip"] else 0), reverse=True)
    return numericos


def validar_pool_lecturas(lecturas):
    evaluados = []

    # Consolidar por texto, conservando cuántas veces apareció y su mejor fuente.
    lecturas_info = {}
    for item in lecturas:
        limpia, fuente, conf, peso = normalizar_lectura_item(item)
        if len(limpia) < 2:
            continue

        if limpia not in lecturas_info:
            lecturas_info[limpia] = {
                "lectura": limpia,
                "fuentes": set(),
                "count": 0,
                "peso_max": 0.0,
                "conf_max": 0.0,
                "tiene_strip": False,
            }

        info = lecturas_info[limpia]
        info["fuentes"].add(str(fuente))
        info["count"] += 1
        info["peso_max"] = max(info["peso_max"], peso)
        info["conf_max"] = max(info["conf_max"], conf)
        if "strip" in str(fuente).lower():
            info["tiene_strip"] = True

    numericos_fallback = evaluar_fallback_numerico_5(lecturas_info)

    for lectura_limpia, info in lecturas_info.items():
        if len(lectura_limpia) < 5:
            continue

        fragmentos = extraer_fragmentos(lectura_limpia)

        for frag in fragmentos:
            for tipo, mascara, formatear, bonus_formato in FORMATOS_PLACA:
                if len(frag) != len(mascara):
                    continue
                if mascara == "DDDDDL":
                    if len(frag) < 5 or frag[4] not in ["9", "P", "R"]:
                        continue

                if mascara == "LDDDDD":
                    if len(frag) != 6:
                        continue
                    if frag[0].isdigit() and frag[0] not in NUM_A_LETRA:
                        continue
                    if not frag[1:].isdigit():
                        # Permitimos solo confusiones letra->número en posiciones numéricas,
                        # no cadenas muy contaminadas.
                        parecen_num = sum(1 for c in frag[1:] if c.isdigit() or c in LETRA_A_NUM)
                        if parecen_num < 5:
                            continue

                for cand, score_base in generar_candidatos_para_formato(frag, mascara):
                    score = score_base + bonus_formato
                    mutaciones = contar_mutaciones(cand, frag)

                    # Prioridad por fuente OCR.
                    score += info["peso_max"]
                    score += min(info["count"], 5) * 7
                    score += min(max(info["conf_max"], 0.0), 1.0) * 8

                    # Premiar lecturas exactas; castigar mutaciones fuertes.
                    if cand == frag:
                        score += 25
                    else:
                        score -= mutaciones * 7

                    # Fragmento extraído de una lectura larga es menos confiable.
                    if len(lectura_limpia) > len(frag):
                        score -= 12

                    if info["tiene_strip"]:
                        score += 25

                    # Para formato 9999AA, los primeros 4 deben parecer números.
                    primeros_4_parecen_num = sum(
                        1 for c in frag[:4]
                        if c.isdigit() or c in LETRA_A_NUM
                    )
                    if mascara == "DDDDLL" and primeros_4_parecen_num >= 3:
                        score += 25

                    primeros_5_parecen_num = sum(
                        1 for c in frag[:5]
                        if c.isdigit() or c in LETRA_A_NUM
                    )
                    if mascara == "DDDDDL":
                        if primeros_5_parecen_num >= 4:
                            score += 15

                        # Solo aceptamos realmente el patrón DDDDDL
                        if len(frag) >= 5 and frag[4] == "9":
                            score += 85
                        elif len(frag) >= 5 and frag[4] in ["P", "R"]:
                            score += 35
                        else:
                            score -= 250

                    # Si el quinto carácter es claramente 9, penalizar DDDDLL porque

                    if mascara == "DDDDLL" and frag[4] == "9":
                        # Caso contextual: 3138-BP suele leerse como 3138-9P.
                        if len(frag) >= 6 and frag[5] in ["P", "R", "B"]:
                            score -= 5
                            if cand[4] == "B":
                                score += 28
                        else:
                            score -= 70
                    elif mascara == "DDDDLL" and frag[4] in ["P", "R"]:
                        score -= 70
                    elif mascara == "DDDDLL" and frag[4].isdigit():
                        score -= 10

                    # Evitar que dos dígitos finales claros se conviertan en letras.
                    if mascara == "DDDDLL" and frag[4].isdigit() and frag[5].isdigit():
                        score -= 80

                    # Para formatos que empiezan con letras, penalizar si arrancan con números claros.
                    if mascara.startswith("LL"):
                        primeros_letra = frag[:mascara.count("L")]
                        digitos_claros = sum(1 for c in primeros_letra if c.isdigit())
                        score -= digitos_claros * 15

                    #refuerzo contextual para placas antiguas mixtas tipo P7-2478.
                    if mascara == "LDDDDD":
                        if cand[0].isalpha() and cand[1:].isdigit():
                            score += 35
                            # P suele ser frecuente en estas placas antiguas/motos.
                            if cand[0] in ["P", "B", "R"]:
                                score += 12
                        if frag[0].isdigit():
                            score -= 25

                    evaluados.append({
                        "score": score,
                        "placa": formatear(cand),
                        "tipo": tipo,
                        "candidato": cand,
                        "fragmento": frag,
                        "lectura": lectura_limpia,
                        "mutaciones": mutaciones,
                        "fuentes": ",".join(sorted(info["fuentes"])),
                        "count": info["count"],
                        "tiene_strip": info["tiene_strip"],
                    })

    # Tie-break: score, si vino del strip, menos mutaciones, más repeticiones.
    evaluados.sort(
        key=lambda x: (
            x["score"],
            1 if x["tiene_strip"] else 0,
            -x["mutaciones"],
            x["count"],
        ),
        reverse=True,
    )

    mejor_alfanum = evaluados[0] if evaluados else None
    mejor_num = numericos_fallback[0] if numericos_fallback else None

    if mejor_alfanum is not None and mejor_alfanum["score"] >= 110:
        mejor = mejor_alfanum
        lista_debug = evaluados + numericos_fallback
    elif mejor_num is not None and mejor_num["score"] >= 155:
        mejor = mejor_num
        lista_debug = numericos_fallback + evaluados
    elif mejor_alfanum is not None:
        mejor = mejor_alfanum
        lista_debug = evaluados + numericos_fallback
    elif mejor_num is not None:
        mejor = mejor_num
        lista_debug = numericos_fallback
    else:
        return False, "", "No se obtuvo texto OCR útil", -1, []

    print("\n[DEBUG VALIDACION] TOP 12 CANDIDATOS")
    for e in lista_debug[:12]:
        print(
            f"{e['placa']:10s} | {e['tipo']:18s} | "
            f"score={e['score']:5.1f} | mut={e['mutaciones']} | rep={e['count']} | "
            f"strip={e['tiene_strip']} | frag={e['fragmento']} | fuentes={e['fuentes']}"
        )

    # Umbrales separados: alfanuméricos aceptan 110; numéricos requieren más evidencia.
    umbral = 155 if mejor["tipo"] == "MOTO NUMERICA" else 110
    if mejor["score"] < umbral:
        return False, mejor["placa"], f"Inválido / score bajo: {mejor['score']:.1f}", mejor["score"], lista_debug

    return True, mejor["placa"], mejor["tipo"], mejor["score"], lista_debug

def validacion_extendida(texto_crudo):
    return validar_pool_lecturas([texto_crudo])


# ===================================================================
# RESTAURACIÓN, BINARIZACIÓN Y SEGMENTACIÓN
# ===================================================================
def restaurar_imagen_borrosa(
    img_bgr,
    psf_tipo=PSF_TIPO,
    motion_length=MOTION_BLUR_LENGTH,
    motion_angle=MOTION_BLUR_ANGLE,
    defocus_radius=DEFOCUS_RADIUS,
    wiener_k=WIENER_K,
):

    gray = asegurar_gris_uint8(img_bgr)
    if gray is None or gray.size == 0:
        return gray, gray, gray, gray

    clahe_vis = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    sharp_clahe = unsharp_mask(clahe_vis, amount=0.65, sigma=1.0)

    if str(psf_tipo).lower() == "defocus":
        psf = crear_psf_defocus(defocus_radius)
    else:
        psf = crear_psf_motion(motion_length, motion_angle)

    deconv = wiener_deconvolution_manual(gray, psf, k=wiener_k, eps=EPSILON_FFT, padding=True)
    deconv_post = aplicar_postprocesamiento_visual(deconv)
    sharp_post = unsharp_mask(deconv_post, amount=0.75, sigma=1.0)

    if SAFE_RESTORE_SELECTION:
        candidatos = [
            ("gray", gray),
            ("clahe", clahe_vis),
            ("clahe_sharp", sharp_clahe),
            ("wiener", deconv),
            ("wiener_post", deconv_post),
            ("wiener_post_sharp", sharp_post),
        ]
        scores = [(nombre, calidad_imagen_para_ocr(im)) for nombre, im in candidatos]
        nombre_best, _ = max(scores, key=lambda x: x[1])
        img_restaurada = dict(candidatos)[nombre_best]
        print("[DEBUG RESTAURACION] Calidad candidatos:", [(n, round(v, 1)) for n, v in scores], "=>", nombre_best)
    else:
        img_restaurada = deconv_post

    img_sharp = unsharp_mask(img_restaurada, amount=0.80, sigma=1.0)
    return img_restaurada, clahe_vis, gray, img_sharp

def binarizaciones(gray):
    gray = asegurar_gris_uint8(gray)
    suave = cv2.GaussianBlur(gray, (3, 3), 0)

    _, otsu_inv = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, otsu = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    adapt_inv = cv2.adaptiveThreshold(
        suave,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8,
    )
    adapt = cv2.bitwise_not(adapt_inv)

    return otsu_inv, otsu, adapt_inv, adapt



def limpiar_bordes_componentes_binary_inv(binary_inv):
    if binary_inv is None or binary_inv.size == 0:
        return binary_inv

    img = binary_inv.copy()
    h, w = img.shape[:2]
    if h < 5 or w < 5:
        return img

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((img > 0).astype(np.uint8), 8)

    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]

        toca_izq = x <= 2
        toca_der = (x + bw) >= (w - 2)
        toca_arriba = y <= 2
        toca_abajo = (y + bh) >= (h - 2)

        es_linea_vertical_borde = (toca_izq or toca_der) and bh > h * 0.55 and bw < w * 0.12
        es_linea_horizontal_borde = (toca_arriba or toca_abajo) and bw > w * 0.55 and bh < h * 0.18

        # Ruido grande del marco, no caracteres.
        if es_linea_vertical_borde or es_linea_horizontal_borde:
            img[labels == label] = 0

    # Segunda pasada: eliminar columnas/filas extremas con demasiada tinta.
    # Esto ataca bordes que quedaron conectados con suciedad.
    for _ in range(2):
        h, w = img.shape[:2]
        if w <= 10 or h <= 10:
            break

        col_black_ratio = np.mean(img > 0, axis=0)
        row_black_ratio = np.mean(img > 0, axis=1)

        left = 0
        while left < w - 1 and col_black_ratio[left] > 0.55:
            left += 1

        right = w - 1
        while right > 0 and col_black_ratio[right] > 0.55:
            right -= 1

        top = 0
        while top < h - 1 and row_black_ratio[top] > 0.55:
            top += 1

        bottom = h - 1
        while bottom > 0 and row_black_ratio[bottom] > 0.55:
            bottom -= 1

        if left > 0:
            img[:, :left + 1] = 0
        if right < w - 1:
            img[:, right:] = 0
        if top > 0:
            img[:top + 1, :] = 0
        if bottom < h - 1:
            img[bottom:, :] = 0

    return img


def limpiar_strip_para_ocr(strip):
    gray = asegurar_gris_uint8(strip)
    if gray is None or gray.size == 0:
        return strip

    # Normalizar a texto negro sobre blanco.
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Foreground negro.
    fg = bw < 128
    if not np.any(fg):
        return gray

    h, w = bw.shape[:2]

    # Eliminar componentes de borde en foreground negro.
    inv = np.where(fg, 255, 0).astype(np.uint8)
    inv = limpiar_bordes_componentes_binary_inv(inv)
    fg = inv > 0

    if not np.any(fg):
        return gray

    ys, xs = np.where(fg)
    x1 = max(0, int(xs.min()) - 10)
    x2 = min(w, int(xs.max()) + 11)
    y1 = max(0, int(ys.min()) - 8)
    y2 = min(h, int(ys.max()) + 9)

    clean = bw[y1:y2, x1:x2]

    # Adelgazar texto muy grueso: en imagen binaria con texto negro/fondo blanco,
    # dilatar el blanco reduce un poco las manchas negras.
    kernel = np.ones((2, 2), np.uint8)
    clean_thin = cv2.dilate(clean, kernel, iterations=1)

    clean_thin = cv2.copyMakeBorder(clean_thin, 18, 18, 26, 26, cv2.BORDER_CONSTANT, value=255)
    return clean_thin

def extraer_strip_global_caracteres(binary_morf, margen=8):
    if binary_morf is None or binary_morf.size == 0:
        return None

    h, w = binary_morf.shape[:2]

    # Banda donde están los caracteres grandes. Evita PERU arriba y pernos abajo.
    y1 = int(h * 0.30)
    y2 = int(h * 0.88)
    x1 = int(w * 0.02)
    x2 = int(w * 0.98)

    banda = binary_morf[y1:y2, x1:x2]
    if banda.size == 0:
        return None

    # Limpiar ruido fino sin romper trazos grandes.
    kernel = np.ones((2, 2), np.uint8)
    banda_limpia = cv2.morphologyEx(banda, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Quitar líneas del marco de la placa antes de calcular el bbox.
    # Esto evita que el borde izquierdo se lea como un dígito 1.
    banda_limpia = limpiar_bordes_componentes_binary_inv(banda_limpia)

    # En binary_morf el texto queda blanco sobre fondo negro.
    ys, xs = np.where(banda_limpia > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    bx1 = max(0, int(xs.min()) - margen)
    bx2 = min(banda_limpia.shape[1], int(xs.max()) + margen)
    by1 = max(0, int(ys.min()) - margen)
    by2 = min(banda_limpia.shape[0], int(ys.max()) + margen)

    if bx2 <= bx1 or by2 <= by1:
        return None

    strip_inv = banda_limpia[by1:by2, bx1:bx2]

    # Convertimos a texto negro sobre fondo blanco para GUI y OCR.
    strip = cv2.bitwise_not(strip_inv)

    # Borde blanco para que OCR no corte las letras de los extremos.
    strip = cv2.copyMakeBorder(strip, 12, 12, 18, 18, cv2.BORDER_CONSTANT, value=255)
    return strip


def segmentar_y_leer(img_restaurada, img_clahe, reader=None):
    base = asegurar_gris_uint8(img_restaurada)
    zona_base = recortar_zona_caracteres(base)
    if zona_base is None or zona_base.size == 0:
        zona_base = base

    # Aumentar un poco la banda antes de binarizar mejora trazos para OCR.
    if zona_base.shape[0] < 90:
        escala = 90.0 / max(1.0, float(zona_base.shape[0]))
        zona_base = cv2.resize(
            zona_base,
            (max(1, int(zona_base.shape[1] * escala)), 90),
            interpolation=cv2.INTER_CUBIC,
        )

    binary_morf, bins, info_bin = binarizacion_robusta(zona_base)
    print(f"[DEBUG BIN] Mejor binarización: {info_bin.get('nombre')} | score={info_bin.get('score'):.1f}")

    contornos, _ = cv2.findContours(binary_morf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contornos = cv2.cvtColor(zona_base, cv2.COLOR_GRAY2BGR)

    altura_img, ancho_img = binary_morf.shape[:2]
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / float(h) if h > 0 else 999
        area = w * h
        if h < altura_img * 0.18:
            continue
        if area < (altura_img * ancho_img) * 0.001:
            continue
        if not (0.08 <= ratio <= 4.5):
            continue
        cv2.rectangle(img_contornos, (x, y), (x + w, y + h), (0, 255, 0), 2)

    strip_visual = extraer_strip_desde_banda_caracteres(binary_morf)
    imagenes_caracteres = []
    img_strip = None
    if strip_visual is not None:
        imagenes_caracteres = [strip_visual]
        img_strip = strip_visual.copy()

    # Asegurar claves esperadas por procesar_fragmento.
    if "otsu_inv" not in bins:
        bins["otsu_inv"] = binary_morf
    if "otsu" not in bins:
        bins["otsu"] = cv2.bitwise_not(binary_morf)
    if "adapt_inv" not in bins:
        bins["adapt_inv"] = binary_morf
    if "adapt" not in bins:
        bins["adapt"] = cv2.bitwise_not(binary_morf)
    bins["zona_base"] = zona_base

    return binary_morf, img_contornos, imagenes_caracteres, img_strip, bins


# ===================================================================
# OCR ROBUSTO
# ===================================================================
def normalizar_resultados_easyocr(resultados):
    normalizados = []
    for item in resultados:
        if isinstance(item, str):
            normalizados.append((None, item, 1.0))
        else:
            bbox = item[0]
            texto = item[1]
            conf = item[2] if len(item) > 2 else 1.0
            normalizados.append((bbox, texto, conf))
    return normalizados


def x_min_bbox(bbox):
    if bbox is None:
        return 0
    try:
        return min(p[0] for p in bbox)
    except Exception:
        return 0


def crear_item_lectura(texto, fuente, conf=1.0):
    return {
        "texto": limpiar_lectura_ocr(texto),
        "fuente": fuente,
        "conf": float(conf or 0.0),
        "peso": peso_fuente_ocr(fuente),
    }


def leer_easyocr_robusto(reader, img, nombre=""):
    lecturas = []
    if reader is None:
        return lecturas
    if img is None or img.size == 0:
        return lecturas

    # En la tira global conviene usar más altura porque allí ya está limpio el texto.
    alto = 170 if "strip" in str(nombre).lower() else 128
    img_ocr = preparar_para_easyocr(img, alto_objetivo=alto, fondo_blanco=True)
    if img_ocr is None or img_ocr.size == 0:
        return lecturas

    # Variantes internas: original normalizada, más contraste y umbralizada.
    variantes = [(nombre, img_ocr)]

    if "strip" in str(nombre).lower():
        try:
            # Variante limpia: elimina bordes que EasyOCR confunde con "1".
            strip_clean = limpiar_strip_para_ocr(img)
            if strip_clean is not None and strip_clean.size > 0:
                variantes.insert(0, (nombre + "_clean", strip_clean))

            g = asegurar_gris_uint8(img_ocr)
            g_blur = cv2.GaussianBlur(g, (3, 3), 0)
            _, th = cv2.threshold(g_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Adelgazado ligero para evitar que 5/8/L/P se vuelvan manchas.
            kernel = np.ones((2, 2), np.uint8)
            th_thin = cv2.dilate(th, kernel, iterations=1)

            variantes.append((nombre + "_otsu", th))
            variantes.append((nombre + "_thin", th_thin))
            variantes.append((nombre + "_big", resize_por_alto(g, 220)))
        except Exception:
            pass

    for nombre_var, img_var in variantes:
        try:
            resultados = reader.readtext(
                img_var,
                detail=1,
                paragraph=False,
                allowlist=ALLOWLIST_PLACA,
                decoder="beamsearch",
                mag_ratio=1.8,
                text_threshold=0.20,
                low_text=0.20,
                link_threshold=0.20,
                contrast_ths=0.05,
                adjust_contrast=0.70,
                width_ths=1.20,
                add_margin=0.08,
            )
        except TypeError:
            resultados = reader.readtext(
                img_var,
                detail=1,
                paragraph=False,
                allowlist=ALLOWLIST_PLACA,
            )
        except Exception as e:
            print(f"[WARN OCR {nombre_var}] EasyOCR falló: {e}")
            continue

        normalizados = normalizar_resultados_easyocr(resultados)

        # 1) Lecturas individuales.
        for bbox, texto, conf in normalizados:
            texto_limpio = limpiar_lectura_ocr(texto)
            if len(texto_limpio) >= 2:
                lecturas.append(crear_item_lectura(texto_limpio, nombre_var, conf))
                print(f"[OCR {nombre_var}] {texto_limpio} | conf={conf:.2f}")

        # 2) Lectura unida de izquierda a derecha.
        if len(normalizados) >= 2:
            ordenados = sorted(normalizados, key=lambda x: x_min_bbox(x[0]))
            unido = "".join(limpiar_lectura_ocr(x[1]) for x in ordenados)
            unido = limpiar_lectura_ocr(unido)
            if len(unido) >= 5:
                conf_prom = sum(float(x[2] or 0.0) for x in ordenados) / max(1, len(ordenados))
                lecturas.append(crear_item_lectura(unido, nombre_var + "_unido", conf_prom))
                print(f"[OCR {nombre_var} UNIDO] {unido}")

    return lecturas


def leer_tesseract_robusto(img, nombre="tesseract"):
    lecturas = []
    if pytesseract is None or img is None or getattr(img, "size", 0) == 0:
        return lecturas

    img_ocr = preparar_para_easyocr(img, alto_objetivo=170 if "strip" in str(nombre).lower() else 140, fondo_blanco=True)
    if img_ocr is None or img_ocr.size == 0:
        return lecturas

    variantes = [(nombre, img_ocr)]
    try:
        g = asegurar_gris_uint8(img_ocr)
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variantes.append((nombre + "_otsu", th))
    except Exception:
        pass

    config = (
        "--oem 3 --psm 7 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    for fuente, im in variantes:
        try:
            data = pytesseract.image_to_data(
                im,
                config=config,
                output_type=pytesseract.Output.DICT,
            )

            textos = []
            confs = []
            for txt, conf in zip(data.get("text", []), data.get("conf", [])):
                limpio = limpiar_lectura_ocr(txt)
                try:
                    c = float(conf)
                except Exception:
                    c = -1.0
                if limpio:
                    textos.append(limpio)
                    if c >= 0:
                        confs.append(c / 100.0)

            unido = limpiar_lectura_ocr("".join(textos))
            if len(unido) >= 2:
                conf_prom = float(np.mean(confs)) if confs else 0.50
                lecturas.append(crear_item_lectura(unido, fuente, conf_prom))
                print(f"[OCR {fuente}] {unido} | conf={conf_prom:.2f}")

        except Exception as e:
            print(f"[WARN OCR {fuente}] Tesseract falló: {e}")

    return lecturas


def agregar_variante(lista, nombre, img):
    if img is not None and getattr(img, "size", 0) > 0:
        lista.append((nombre, img))


def procesar_fragmento(img, reader):
    if img is None or img.size == 0:
        return {"valido": False, "tipo": "Imagen vacía", "placa": ""}

    # Si la ROI es extremadamente pequeña, se amplía moderadamente para que OCR tenga más píxeles.
    # La deconvolución sigue trabajando solo sobre la placa, no sobre toda la escena.
    h0, w0 = img.shape[:2]
    if h0 < 80:
        escala = 80.0 / max(1.0, float(h0))
        img = cv2.resize(img, (max(1, int(w0 * escala)), 80), interpolation=cv2.INTER_CUBIC)

    img_restaurada, img_clahe, img_gray, img_sharp = restaurar_imagen_borrosa(img)

    img_bin, img_contornos, caracteres, img_strip, bins = segmentar_y_leer(img_restaurada, img_clahe, reader)

    h_img = img_restaurada.shape[0]
    zona_gray = recortar_zona_caracteres(img_gray)
    zona_clahe = recortar_zona_caracteres(img_clahe)
    zona_rest = recortar_zona_caracteres(img_restaurada)
    zona_sharp = recortar_zona_caracteres(img_sharp)
    zona_lowlight = mejorar_baja_luz(zona_gray)
    zona_clahe_fuerte = clahe_fuerte(zona_gray)
    zona_sharp_fuerte = unsharp_mask(zona_clahe_fuerte, amount=0.95, sigma=1.0) if zona_clahe_fuerte is not None else None
    # bins ya corresponden a la zona de caracteres; no recortar otra vez.
    zona_otsu = bins["otsu"]
    zona_adapt = bins["adapt"]
    zona_otsu_inv = bins["otsu_inv"]
    zona_adapt_inv = bins["adapt_inv"]
    zona_base_segmentacion = bins.get("zona_base", zona_rest)

    variantes_ocr = []
    agregar_variante(variantes_ocr, "gray_completa", img_gray)
    agregar_variante(variantes_ocr, "clahe_completa", img_clahe)
    agregar_variante(variantes_ocr, "restaurada_completa", img_restaurada)
    agregar_variante(variantes_ocr, "sharp_completa", img_sharp)

    agregar_variante(variantes_ocr, "zona_gray", zona_gray)
    agregar_variante(variantes_ocr, "zona_clahe", zona_clahe)
    agregar_variante(variantes_ocr, "zona_restaurada", zona_rest)
    agregar_variante(variantes_ocr, "zona_sharp", zona_sharp)
    agregar_variante(variantes_ocr, "zona_lowlight", zona_lowlight)
    agregar_variante(variantes_ocr, "zona_clahe_fuerte", zona_clahe_fuerte)
    agregar_variante(variantes_ocr, "zona_sharp_fuerte", zona_sharp_fuerte)
    agregar_variante(variantes_ocr, "zona_base_segmentacion", zona_base_segmentacion)
    agregar_variante(variantes_ocr, "zona_otsu", zona_otsu)
    agregar_variante(variantes_ocr, "zona_adapt", zona_adapt)
    agregar_variante(variantes_ocr, "zona_otsu_inv", cv2.bitwise_not(zona_otsu_inv))
    agregar_variante(variantes_ocr, "zona_adapt_inv", cv2.bitwise_not(zona_adapt_inv))
    agregar_variante(variantes_ocr, "strip_caracteres", img_strip)

    if DEBUG_GUARDAR_IMAGENES:
        try:
            cv2.imwrite("debug_roi_gray.jpg", img_gray)
            cv2.imwrite("debug_zona_caracteres.jpg", zona_sharp)
            cv2.imwrite("debug_bin_otsu_panel.jpg", img_bin)
            if img_strip is not None:
                cv2.imwrite("debug_strip_caracteres.jpg", img_strip)
        except Exception as e:
            print(f"[WARN] No se pudieron guardar imágenes debug: {e}")

    textos_crudos = []
    for nombre, imagen_ocr in variantes_ocr:
        # OCR clásico principal/compatible si Tesseract está instalado.
        textos_crudos.extend(leer_tesseract_robusto(imagen_ocr, "tess_" + nombre))
        # EasyOCR queda como motor robusto adicional/comparativo.
        textos_crudos.extend(leer_easyocr_robusto(reader, imagen_ocr, nombre))

    # IMPORTANTE:
    # textos_crudos contiene dicts con texto/fuente/confianza. No hay que convertirlos
    # con str(dict), porque eso mete palabras como TEXTO/FUENTE/CONF en la validación.
    lecturas_para_validar = list(textos_crudos)

    textos_unicos_debug = []
    textos_unicos_para_combinar = []
    for item in textos_crudos:
        texto, fuente, conf, peso = normalizar_lectura_item(item)
        if texto and texto not in textos_unicos_debug:
            textos_unicos_debug.append(f"{texto} [{fuente}]")
            textos_unicos_para_combinar.append(texto)

    # No unir todo como verdad principal. Solo una lectura secundaria de baja prioridad.
    if textos_unicos_para_combinar:
        combinado_corto = "".join(textos_unicos_para_combinar[:3])
        combinado_corto = limpiar_lectura_ocr(combinado_corto)
        if len(combinado_corto) >= 6:
            lecturas_para_validar.append(crear_item_lectura(combinado_corto, "combinado_baja_prioridad", 0.30))
            textos_unicos_debug.append(f"{combinado_corto} [combinado_baja_prioridad]")

    print("\n[DEBUG OCR] Lecturas crudas normalizadas:")
    print(textos_unicos_debug)

    valido, placa, tipo, score, candidatos = validar_pool_lecturas(lecturas_para_validar)

    print("\n[DEBUG OCR] --- RESULTADO FINAL ---")
    if valido:
        print(f"GANADOR : {placa}")
        print(f"FORMATO : {tipo}")
        print(f"PUNTAJE : {score:.1f}")
    else:
        print("GANADOR : Ninguno")
        print(f"MOTIVO  : {tipo}")
        print(f"MEJOR APROX.: {placa}")
    print("-----------------------------------\n")

    return {
        "roi": img,
        "preprocesada": img_clahe,
        "binarizada": img_bin,
        "segmentacion": img_contornos,
        "caracteres": caracteres,
        "placa": placa,
        "valido": valido,
        "tipo": tipo,
        "score": score,
    }


# ===================================================================
# INTERFAZ GRÁFICA TKINTER
# ===================================================================
class AppPlacas:
    def __init__(self, root):
        self.root = root
        self.root.title("Restauración y OCR de Placas Desenfocadas")
        self.root.geometry("950x780")
        self.root.configure(bg="#2c3e50")

        init_db()

        print("Cargando modelo OCR... por favor espera.")
        self.reader = None
        if easyocr is not None:
            try:
                self.reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
                print("EasyOCR cargado correctamente.")
            except Exception as e:
                print(f"[WARN] EasyOCR no se pudo cargar: {e}")
        else:
            print("[WARN] EasyOCR no está instalado. Se usará Tesseract si está disponible.")

        if pytesseract is None and self.reader is None:
            messagebox.showwarning(
                "OCR no disponible",
                "No se encontró EasyOCR ni pytesseract. Instala al menos uno para leer placas."
            )

        lbl_titulo = tk.Label(
            self.root,
            text="Restauración y OCR de Placas Desenfocadas",
            font=("Helvetica", 20, "bold"),
            bg="#2c3e50",
            fg="#ecf0f1",
        )
        lbl_titulo.pack(pady=10)

        btn_subir = tk.Button(
            self.root,
            text="Subir foto y seleccionar placa",
            font=("Helvetica", 12),
            bg="#3498db",
            fg="white",
            cursor="hand2",
            command=self.cargar_imagen,
        )
        btn_subir.pack(pady=5)

        self.lbl_resultado = tk.Label(
            self.root,
            text="Sube una foto para analizar...",
            font=("Helvetica", 14),
            bg="#2c3e50",
            fg="#f1c40f",
        )
        self.lbl_resultado.pack(pady=5)

        self.canvas_frame = tk.Frame(self.root, bg="#34495e")
        self.canvas_frame.pack(pady=5)
        self.canvas = tk.Canvas(self.canvas_frame, bg="#34495e", cursor="crosshair", highlightthickness=0)
        self.canvas.pack()

        self.frame_etapas = tk.Frame(self.root, bg="#2c3e50")
        self.lbls_etapas = {}

        nombres = [
            ("roi", "1. ROI Aplanado"),
            ("preprocesada", "2. Preprocesada (CLAHE)"),
            ("binarizada", "3. Binarizada"),
            ("segmentacion", "4. Segmentación"),
            ("caracteres", "5. Caracteres Extraídos"),
            ("placa", "6. Placa Detectada"),
        ]

        for i, (key, title) in enumerate(nombres):
            row = i // 3
            col = i % 3
            frame_celda = tk.Frame(self.frame_etapas, bg="#34495e", bd=2, relief=tk.GROOVE)
            frame_celda.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

            lbl_tit = tk.Label(
                frame_celda,
                text=title,
                font=("Helvetica", 10, "bold"),
                bg="#34495e",
                fg="#ecf0f1",
            )
            lbl_tit.pack(side=tk.TOP, pady=2)

            lbl_img = tk.Label(frame_celda, bg="#2c3e50", width=20, height=6)
            lbl_img.pack(side=tk.BOTTOM, padx=5, pady=5, expand=True)
            self.lbls_etapas[key] = lbl_img

        self.puntos_roi = []
        self.original_img_bgr = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.ruta_actual = ""
        self.img_tk = None

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)

    def cargar_imagen(self):
        ruta_imagen = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Archivos de Imagen", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.arw *.ARW")],
        )
        if not ruta_imagen:
            return

        self.frame_etapas.pack_forget()
        self.ruta_actual = ruta_imagen
        self.lbl_resultado.configure(
            text="Haz clic en las 4 esquinas de la placa. El sistema ordenará los puntos automáticamente.",
            fg="#f39c12",
        )
        self.root.update()

        try:
            if ruta_imagen.lower().endswith(".arw"):
                if rawpy is None:
                    raise RuntimeError("rawpy no está instalado. Instala con: pip install rawpy")
                with rawpy.imread(ruta_imagen) as raw:
                    rgb = raw.postprocess()
                self.original_img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                img_pil = Image.fromarray(rgb)
            else:
                self.original_img_bgr = cv2.imread(ruta_imagen)
                if self.original_img_bgr is None:
                    raise RuntimeError("OpenCV no pudo leer la imagen. Revisa la ruta o el formato.")
                img_rgb = cv2.cvtColor(self.original_img_bgr, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)

            canvas_max_width = 850
            canvas_max_height = 360
            orig_w, orig_h = img_pil.size

            ratio = min(canvas_max_width / orig_w, canvas_max_height / orig_h)
            new_w = max(1, int(orig_w * ratio))
            new_h = max(1, int(orig_h * ratio))

            self.scale_x = orig_w / new_w
            self.scale_y = orig_h / new_h

            img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.img_tk = ImageTk.PhotoImage(img_resized)

            self.canvas.config(width=new_w, height=new_h)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
            self.puntos_roi = []

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{e}")

    def on_button_press(self, event):
        if self.original_img_bgr is None:
            return

        x_orig = int(event.x * self.scale_x)
        y_orig = int(event.y * self.scale_y)
        self.puntos_roi.append((x_orig, y_orig))

        r = 4
        self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill="red", outline="red")

        if len(self.puntos_roi) > 1:
            prev_x = int(self.puntos_roi[-2][0] / self.scale_x)
            prev_y = int(self.puntos_roi[-2][1] / self.scale_y)
            self.canvas.create_line(prev_x, prev_y, event.x, event.y, fill="red", width=2)

        if len(self.puntos_roi) == 4:
            first_x = int(self.puntos_roi[0][0] / self.scale_x)
            first_y = int(self.puntos_roi[0][1] / self.scale_y)
            self.canvas.create_line(event.x, event.y, first_x, first_y, fill="red", width=2)
            self.procesar_perspectiva()
            self.puntos_roi = []

    def procesar_perspectiva(self):
        if len(self.puntos_roi) != 4:
            return

        pts = order_points(self.puntos_roi)

        width_a = np.linalg.norm(pts[2] - pts[3])
        width_b = np.linalg.norm(pts[1] - pts[0])
        max_width = max(int(width_a), int(width_b), 80)

        height_a = np.linalg.norm(pts[1] - pts[2])
        height_b = np.linalg.norm(pts[0] - pts[3])
        max_height = max(int(height_a), int(height_b), 30)

        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(pts, dst)
        recorte_plano = cv2.warpPerspective(self.original_img_bgr, M, (max_width, max_height))

        if DEBUG_GUARDAR_IMAGENES:
            cv2.imwrite("roi_aplanado_debug.jpg", recorte_plano)

        self.lbl_resultado.configure(text="Procesando restauración + OCR robusto...", fg="#f39c12")
        self.root.update()

        etapas = procesar_fragmento(recorte_plano, self.reader)

        self.frame_etapas.pack(pady=10)
        self.actualizar_panel_etapas(etapas)

    def actualizar_panel_etapas(self, etapas):
        def preparar_img_gui(img, size=(170, 105)):
            if img is None:
                return None
            if isinstance(img, list):
                return None
            if getattr(img, "size", 0) == 0:
                return None

            if len(img.shape) == 2:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail(size)
            return ImageTk.PhotoImage(img_pil)

        for key in ["roi", "preprocesada", "binarizada", "segmentacion"]:
            if key in etapas and etapas[key] is not None:
                tk_img = preparar_img_gui(etapas[key])
                if tk_img is not None:
                    self.lbls_etapas[key].configure(image=tk_img, text="", width=0, height=0)
                    self.lbls_etapas[key].image = tk_img

        if "caracteres" in etapas and len(etapas["caracteres"]) > 0:
            try:
                chars = [
                    cv2.copyMakeBorder(c, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
                    for c in etapas["caracteres"]
                ]
                alto_max = max(c.shape[0] for c in chars)
                chars_resized = []
                for c in chars:
                    nuevo_w = max(1, int(c.shape[1] * alto_max / c.shape[0]))
                    chars_resized.append(cv2.resize(c, (nuevo_w, alto_max), interpolation=cv2.INTER_CUBIC))
                strip = np.hstack(chars_resized)
                tk_strip = preparar_img_gui(strip, size=(220, 105))
                self.lbls_etapas["caracteres"].configure(image=tk_strip, text="", width=0, height=0)
                self.lbls_etapas["caracteres"].image = tk_strip
            except Exception as e:
                self.lbls_etapas["caracteres"].configure(
                    image="",
                    text=f"Error panel chars:\n{e}",
                    width=24,
                    height=6,
                    fg="#ecf0f1",
                )
        else:
            self.lbls_etapas["caracteres"].configure(
                image="",
                text="N/A\nSin caracteres claros",
                width=20,
                height=6,
                fg="#ecf0f1",
            )

        placa = etapas.get("placa", "")
        es_valido = etapas.get("valido", False)
        tipo = etapas.get("tipo", "")
        score = etapas.get("score", 0)

        if es_valido:
            self.lbls_etapas["placa"].configure(
                text=f"{placa}\n\n({tipo})\nScore: {score:.0f}",
                font=("Helvetica", 12, "bold"),
                fg="#2ecc71",
                image="",
                width=18,
                height=5,
            )
            self.lbl_resultado.configure(text=f"PLACA RECUPERADA: {placa} | TIPO: {tipo} | SCORE: {score:.0f}", fg="#2ecc71")
            # Guardar solo si hay confianza suficiente. Puedes bajar este umbral si deseas.
            if score >= 130:
                guardar_registro(placa, tipo, self.ruta_actual)
        else:
            texto_mostrar = "No Detectada"
            if placa:
                texto_mostrar += f"\n\nMejor aprox.:\n{placa}"
            self.lbls_etapas["placa"].configure(
                text=texto_mostrar,
                font=("Helvetica", 11, "bold"),
                fg="#e74c3c",
                image="",
                width=20,
                height=6,
            )
            self.lbl_resultado.configure(text=f"Error: {tipo}", fg="#e74c3c")


if __name__ == "__main__":
    root = tk.Tk()
    app = AppPlacas(root)
    root.mainloop()
