"""
Sistema de Análisis Automático de Facturas en PDF
Extrae datos financieros, clasifica ingresos/gastos y genera resúmenes automáticos
Autor: Sistema de análisis de facturas
"""

import streamlit as st
import pdfplumber
import PyPDF2
import pandas as pd
import os
import json
import re
from datetime import datetime
from openai import OpenAI

# Configuración de la página
st.set_page_config(
    page_title="Analizador de Facturas PDF",
    page_icon="📊",
    layout="wide"
)

# Inicializar cliente OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def extraer_texto_pdf(archivo_pdf):
    """
    Extrae el texto de un archivo PDF usando pdfplumber (principal)
    con fallback a PyPDF2 si falla
    
    Args:
        archivo_pdf: Objeto de archivo PDF subido
        
    Returns:
        str: Texto extraído del PDF
    """
    texto_completo = ""
    
    try:
        # Método principal: pdfplumber (mejor para facturas con tablas)
        with pdfplumber.open(archivo_pdf) as pdf:
            for pagina in pdf.pages:
                texto = pagina.extract_text()
                if texto:
                    texto_completo += texto + "\n"
        
        if texto_completo.strip():
            return texto_completo
    except Exception as e:
        st.warning(f"pdfplumber falló, intentando con PyPDF2: {str(e)}")
    
    try:
        # Fallback: PyPDF2
        archivo_pdf.seek(0)  # Resetear el puntero del archivo
        lector_pdf = PyPDF2.PdfReader(archivo_pdf)
        for pagina in lector_pdf.pages:
            texto = pagina.extract_text()
            if texto:
                texto_completo += texto + "\n"
    except Exception as e:
        st.error(f"Error al extraer texto con PyPDF2: {str(e)}")
        return None
    
    return texto_completo if texto_completo.strip() else None


def analizar_factura_con_openai(texto_pdf, nombre_archivo):
    """
    Utiliza OpenAI GPT-5 para extraer datos estructurados de la factura
    
    Args:
        texto_pdf: Texto extraído del PDF
        nombre_archivo: Nombre del archivo para contexto
        
    Returns:
        dict: Datos estructurados de la factura
    """
    prompt = f"""
Analiza el siguiente texto extraído de una factura PDF y extrae TODOS los datos disponibles.
Devuelve un JSON con la siguiente estructura exacta:

{{
    "numero_factura": "número de factura si existe, o null",
    "fecha": "fecha en formato YYYY-MM-DD si existe, o null",
    "proveedor_cliente": "nombre del proveedor o cliente",
    "tipo": "ingreso o gasto (determina según el contexto: si dice 'factura emitida' o tiene datos del receptor, es ingreso; si dice 'factura recibida' o muestra datos del emisor como proveedor, es gasto)",
    "base_imponible": número decimal o null,
    "iva": número decimal o null,
    "porcentaje_iva": número decimal o null,
    "irpf": número decimal o null,
    "porcentaje_irpf": número decimal o null,
    "total": número decimal o null,
    "conceptos": "breve descripción de productos/servicios",
    "observaciones": "cualquier dato adicional relevante"
}}

IMPORTANTE:
- Extrae SOLO los datos que realmente existan en el texto
- Si un campo no existe, usa null
- Los números deben ser decimales sin símbolos de moneda
- La fecha debe estar en formato YYYY-MM-DD
- Para determinar tipo (ingreso/gasto): analiza si la factura es emitida por nosotros (ingreso) o recibida de un proveedor (gasto)

Texto de la factura:
{texto_pdf}

Responde ÚNICAMENTE con el JSON, sin texto adicional.
"""
    
    try:
        # Usar GPT-5 (el modelo más reciente de OpenAI lanzado el 7 de agosto de 2025)
        # No cambiar a modelos antiguos a menos que el usuario lo solicite explícitamente
        response = openai_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un experto contador que extrae datos de facturas con precisión. Respondes únicamente con JSON válido."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=2048
        )
        
        # Parsear respuesta JSON
        datos_factura = json.loads(response.choices[0].message.content)
        datos_factura["nombre_archivo"] = nombre_archivo
        
        return datos_factura
        
    except Exception as e:
        st.error(f"Error al analizar con OpenAI: {str(e)}")
        # Retornar estructura básica en caso de error
        return {
            "nombre_archivo": nombre_archivo,
            "numero_factura": None,
            "fecha": None,
            "proveedor_cliente": "Error al procesar",
            "tipo": "gasto",
            "base_imponible": 0.0,
            "iva": 0.0,
            "porcentaje_iva": 0.0,
            "irpf": 0.0,
            "porcentaje_irpf": 0.0,
            "total": 0.0,
            "conceptos": "",
            "observaciones": f"Error: {str(e)}"
        }


def procesar_factura(archivo_pdf):
    """
    Procesa un archivo PDF completo: extracción + análisis
    
    Args:
        archivo_pdf: Objeto de archivo PDF subido
        
    Returns:
        dict: Datos estructurados de la factura o None si falla
    """
    nombre_archivo = archivo_pdf.name
    
    with st.spinner(f"📄 Extrayendo texto de {nombre_archivo}..."):
        texto = extraer_texto_pdf(archivo_pdf)
    
    if not texto:
        st.error(f"❌ No se pudo extraer texto de {nombre_archivo}")
        return None
    
    with st.spinner(f"🤖 Analizando factura {nombre_archivo} con IA..."):
        datos = analizar_factura_con_openai(texto, nombre_archivo)
    
    return datos


def calcular_resumen_financiero(facturas_df):
    """
    Calcula totales financieros y genera el balance general
    
    Args:
        facturas_df: DataFrame con todas las facturas procesadas
        
    Returns:
        dict: Resumen financiero completo
    """
    # Separar ingresos y gastos
    ingresos = facturas_df[facturas_df['tipo'] == 'ingreso']
    gastos = facturas_df[facturas_df['tipo'] == 'gasto']
    
    # Calcular totales
    total_ingresos = ingresos['total'].sum()
    total_gastos = gastos['total'].sum()
    beneficio = total_ingresos - total_gastos
    
    # IVA: en ingresos es repercutido (cobrado), en gastos es soportado (pagado)
    iva_repercutido = ingresos['iva'].sum()
    iva_soportado = gastos['iva'].sum()
    balance_iva = iva_repercutido - iva_soportado
    
    # IRPF: en ingresos es retenido (restado), en gastos es aplicado
    irpf_total_ingresos = ingresos['irpf'].sum()
    irpf_total_gastos = gastos['irpf'].sum()
    
    return {
        'total_ingresos': total_ingresos,
        'total_gastos': total_gastos,
        'beneficio': beneficio,
        'iva_soportado': iva_soportado,
        'iva_repercutido': iva_repercutido,
        'balance_iva': balance_iva,
        'irpf_ingresos': irpf_total_ingresos,
        'irpf_gastos': irpf_total_gastos,
        'num_ingresos': len(ingresos),
        'num_gastos': len(gastos)
    }


def mostrar_resumen_financiero(resumen):
    """
    Muestra el dashboard financiero con métricas visuales
    
    Args:
        resumen: Diccionario con el resumen financiero
    """
    st.header("📊 Resumen Financiero General")
    
    # Fila 1: Métricas principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="💰 Total Ingresos",
            value=f"{resumen['total_ingresos']:.2f} €",
            delta=f"{resumen['num_ingresos']} facturas"
        )
    
    with col2:
        st.metric(
            label="💸 Total Gastos",
            value=f"{resumen['total_gastos']:.2f} €",
            delta=f"{resumen['num_gastos']} facturas"
        )
    
    with col3:
        beneficio = resumen['beneficio']
        st.metric(
            label="📈 Beneficio/Pérdida",
            value=f"{beneficio:.2f} €",
            delta="Positivo" if beneficio >= 0 else "Negativo",
            delta_color="normal" if beneficio >= 0 else "inverse"
        )
    
    st.divider()
    
    # Fila 2: IVA e IRPF
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📤 IVA Repercutido (cobrado)",
            value=f"{resumen['iva_repercutido']:.2f} €"
        )
    
    with col2:
        st.metric(
            label="📥 IVA Soportado (pagado)",
            value=f"{resumen['iva_soportado']:.2f} €"
        )
    
    with col3:
        balance_iva = resumen['balance_iva']
        st.metric(
            label="⚖️ Balance IVA",
            value=f"{balance_iva:.2f} €",
            delta="A ingresar" if balance_iva > 0 else "A compensar" if balance_iva < 0 else "Neutro",
            delta_color="normal" if balance_iva >= 0 else "inverse"
        )
    
    # Información adicional IRPF
    if resumen['irpf_ingresos'] > 0 or resumen['irpf_gastos'] > 0:
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="🏦 IRPF Retenido (ingresos)",
                value=f"{resumen['irpf_ingresos']:.2f} €"
            )
        
        with col2:
            st.metric(
                label="🏦 IRPF en Gastos",
                value=f"{resumen['irpf_gastos']:.2f} €"
            )


def main():
    """
    Función principal de la aplicación Streamlit
    """
    # Título y descripción
    st.title("📊 Analizador Automático de Facturas PDF")
    st.markdown("""
    ### Sistema completo de análisis financiero
    
    **Características:**
    - 📤 Sube uno o varios PDFs de facturas
    - 🤖 Extracción automática de datos con IA
    - 📊 Clasificación automática (ingresos/gastos)
    - 💡 Cálculos financieros completos
    - 📈 Dashboard de resumen visual
    
    ---
    """)
    
    # Inicializar estado de sesión
    if 'facturas_procesadas' not in st.session_state:
        st.session_state.facturas_procesadas = []
    
    # Sección de carga de archivos
    st.subheader("📁 Subir Facturas PDF")
    
    archivos_pdf = st.file_uploader(
        "Arrastra y suelta tus facturas PDF aquí, o haz clic para seleccionar",
        type=['pdf'],
        accept_multiple_files=True,
        help="Puedes subir múltiples archivos PDF a la vez"
    )
    
    # Botones de acción
    col1, col2 = st.columns([1, 3])
    
    with col1:
        procesar_btn = st.button("🚀 Procesar Facturas", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Limpiar Todo", use_container_width=True):
            st.session_state.facturas_procesadas = []
            st.rerun()
    
    # Procesar archivos cuando se presiona el botón
    if procesar_btn and archivos_pdf:
        st.session_state.facturas_procesadas = []  # Limpiar datos previos
        
        progress_bar = st.progress(0)
        total_archivos = len(archivos_pdf)
        
        for idx, archivo in enumerate(archivos_pdf):
            # Actualizar barra de progreso
            progress_bar.progress((idx + 1) / total_archivos)
            
            # Procesar factura
            datos_factura = procesar_factura(archivo)
            
            if datos_factura:
                st.session_state.facturas_procesadas.append(datos_factura)
                st.success(f"✅ {archivo.name} procesado correctamente")
            else:
                st.error(f"❌ Error al procesar {archivo.name}")
        
        progress_bar.empty()
        st.success(f"🎉 ¡Proceso completado! {len(st.session_state.facturas_procesadas)} facturas procesadas.")
    
    # Mostrar resultados si hay facturas procesadas
    if st.session_state.facturas_procesadas:
        st.divider()
        
        # Convertir a DataFrame para mejor visualización
        df_facturas = pd.DataFrame(st.session_state.facturas_procesadas)
        
        # Asegurar que los campos numéricos sean float
        campos_numericos = ['base_imponible', 'iva', 'porcentaje_iva', 'irpf', 'porcentaje_irpf', 'total']
        for campo in campos_numericos:
            df_facturas[campo] = pd.to_numeric(df_facturas[campo], errors='coerce').fillna(0.0)
        
        # Mostrar resumen financiero
        resumen = calcular_resumen_financiero(df_facturas)
        mostrar_resumen_financiero(resumen)
        
        st.divider()
        
        # Tabla detallada de facturas
        st.header("📋 Detalle de Facturas Procesadas")
        
        # Preparar DataFrame para visualización
        df_display = df_facturas[[
            'nombre_archivo', 'numero_factura', 'fecha', 'proveedor_cliente', 
            'tipo', 'base_imponible', 'iva', 'irpf', 'total', 'conceptos'
        ]].copy()
        
        # Renombrar columnas para mejor lectura
        df_display.columns = [
            'Archivo', 'Nº Factura', 'Fecha', 'Proveedor/Cliente', 
            'Tipo', 'Base Imponible (€)', 'IVA (€)', 'IRPF (€)', 'Total (€)', 'Conceptos'
        ]
        
        # Formatear números
        columnas_euro = ['Base Imponible (€)', 'IVA (€)', 'IRPF (€)', 'Total (€)']
        for col in columnas_euro:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}")
        
        # Mostrar tabla con estilo
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True
        )
        
        # Opción de descarga
        st.divider()
        st.subheader("💾 Exportar Datos")
        
        # Convertir a CSV
        csv = df_display.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Descargar Resumen en CSV",
            data=csv,
            file_name=f"resumen_facturas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    elif not archivos_pdf:
        # Instrucciones iniciales
        st.info("""
        👆 **Comienza subiendo tus facturas PDF**
        
        El sistema extraerá automáticamente:
        - Base imponible, IVA, IRPF
        - Fecha y número de factura
        - Proveedor o cliente
        - Clasificación automática (ingreso/gasto)
        - Y generará un resumen financiero completo
        """)
    
    # Footer
    st.divider()
    st.caption("🔒 Tus datos se procesan de forma segura. Los PDFs no se almacenan permanentemente.")


if __name__ == "__main__":
    main()
