# ALPR-Restauracion-Placas
---
## DESCRIPCIÓN
El sistema ALPR procesa matemáticamente imágenes degradadas en el dominio de la frecuencia usando el Filtro de Wiener para corregir el desenfoque de movimiento o de lente, aislando la placa. Al restaurarla, un ensamble OCR lee los caracteres y un motor heurístico basado en las normativas del MTC los valida. Al finalizar, la lectura correcta se almacena en una base de datos local.
---
## REQUISITOS DEL SISTEMA

* **Sistema Operativo:** Windows 10/11, macOS o distribuciones Linux
* **Hardware:** CPU estándar (mínimo 4 núcleos), 8GB RAM, cámara/archivos locales
* **Entorno:** Python 3.8 o superior
* **Conexión a internet:** Requerida para la primera ejecución (descarga de modelos de EasyOCR)

---
## INSTALACIÓN Y CONFIGURACIÓN
1. **Instalar dependencias**

El sistema utiliza librerías de visión por computadora y PDS. Se recomienda instalar dentro de un entorno virtual (venv).

* **pip install opencv-python numpy easyocr pytesseract pillow rawpy**
  
---
## INTERFAZ DE CONTROL

| Acción en Interfaz | Resultado del Sistema |
| :--- | :--- |
| **Clic en 4 esquinas de la imagen** | Ejecuta la corrección de perspectiva (Warp Perspective) para aplanar la placa |
| **Procesamiento Automático** | Genera y evalúa candidatas de restauración |
| **Validación Heurística** | Filtra mutaciones y descarta "alucinaciones" del OCR basado en reglas MTC |

--
## ESTADO DEL PROYECTO (AVANCE 50%)
El presente repositorio justifica el desarrollo del 50% del sistema. Hasta este punto, se ha logrado implementar la interfaz gráfica, el módulo completo de PDS (restauración de Wiener en el dominio de la frecuencia), la segmentación robusta y la conexión base con el ensamble OCR. La fase restante contempla la calibración estadística final del motor heurístico MTC y pruebas de estrés.

---
## Equipo de Desarrollo:
* •	BRUNO CORDOVA ANTONELLA STEFANY
* •	MIJAHUANGA JIMÉNEZ JHEYLER
* •	PEDEMONTE TIMANA CRISTHIAN JOSUE
* •	PINGO UMBO JERSSON YAIR
* •	SAAVEDRA CARRILLO GIANCARLO GUSTAVO
* •	VILELA OROZCO KEVIN ABEL
