import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

path_evaluar = None
img_evaluar = None

# Función para verificar si hay imágenes en ambos canvas
def verificar_imagenes():
    if hasattr(canvas_evaluar, 'image'):
        boton_evaluar.grid(row=0, column=0, padx=10)  # Mostrar botón Evaluar
        boton_borrar.grid(row=1, column=0, padx=10)   # Mostrar botón Borrar
    else:
        boton_evaluar.grid_remove()  # Ocultar botón Evaluar
        boton_borrar.grid_remove()   # Ocultar botón Borrar

# Modificar las funciones existentes para incluir la validación
def seleccionar_imagen(canvas_widget, tipo_examen):
    global path_referencia, path_evaluar, img1, img_evaluar

    archivo = filedialog.askopenfilename(
        title="Seleccionar " + tipo_examen,
        filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp")],
        initialdir="Examenes"
    )

    if archivo:
        try:
            # Leer imagen con OpenCV para poder ajustarla
            img_evaluar = cv2.imread(archivo)
            _, resultados = ajustar_perspectiva(img_evaluar)
            imagen_ajustada = resultados["perspectiva"]

            # Convertir de BGR a RGB para tkinter
            imagen_rgb = cv2.cvtColor(imagen_ajustada, cv2.COLOR_BGR2RGB)
            imagen_pil = Image.fromarray(imagen_rgb)
            imagen_pil = imagen_pil.resize((300, 450))  # Redimensionar para canvas
            img_tk = ImageTk.PhotoImage(imagen_pil)

            # Mostrar en el canvas
            canvas_widget.delete("all")
            canvas_widget.image = img_tk
            canvas_widget.create_image(0, 0, anchor=tk.NW, image=img_tk)

            # Guardar la imagen corregida
            if canvas_widget == canvas_evaluar:
                archivofoto = os.path.splitext(os.path.basename(archivo))[0]
                label_evaluar.config(text=archivofoto)

            verificar_imagenes()

        except Exception as e:
            print(f"Error al cargar la imagen: {e}")

# Función para borrar las imágenes de ambos canvas y reiniciar valores
def borrar_imagenes():
    # Eliminar imágenes del canvas de evaluación
    canvas_evaluar.delete("all")
    if hasattr(canvas_evaluar, 'image'):
        del canvas_evaluar.image  # Eliminar referencia a la imagen

    # Reiniciar las etiquetas de aciertos y calificación
    label_aciertos.config(text="Aciertos: 0")
    label_calificacion.config(text="Calificación: 0")

    label_evaluar.config(text="Examen a Evaluar")
    label_referencia.config(text="Examen de Referencia")

    # Reiniciar los labels de respuestas identificadas
    for j in range(3):  # 3 columnas
        for i in range(20):  # 20 preguntas
            labels_identificados[j][i].config(text=f"{i+1}-", fg="black")

    verificar_imagenes()

def ajustar_perspectiva(imagen):
    def ordenar_puntos(puntos):
        puntos = puntos.reshape(4, 2)
        suma = puntos.sum(axis=1)
        resta = np.diff(puntos, axis=1)

        ordenados = np.zeros((4, 2), dtype="float32")
        ordenados[0] = puntos[np.argmin(suma)]     # arriba-izquierda
        ordenados[2] = puntos[np.argmax(suma)]     # abajo-derecha
        ordenados[1] = puntos[np.argmin(resta)]    # arriba-derecha
        ordenados[3] = puntos[np.argmax(resta)]    # abajo-izquierda
        return ordenados
    
    # Preprocesamiento más robusto
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    masksize = int(imagen.shape[1] / 150) | 1
    gris = cv2.GaussianBlur(gris,(masksize, masksize), 0) # Suavizado
    gridsize = int(imagen.shape[1]/30)

    blur_resized = cv2.resize(gris, (600, 800))
    #cv2.imshow('Img BLUR', blur_resized)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))
    gris = clahe.apply(gris)

    gris_resized = cv2.resize(gris, (600, 800))
    #cv2.imshow('Img CLAHE', gris_resized)

    # Umbral adaptativo (mejor para iluminación irregular)
    binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, masksize, 12)

    binaria_resized = cv2.resize(binaria, (600, 800))
    #cv2.imshow('Img BIN', binaria_resized)

    canny = cv2.Canny(gris, 50, 100)
    canny_resized = cv2.resize(canny, (600, 800))
    #cv2.imshow('Img CANNY', canny_resized)

    bincanny = cv2.bitwise_or(binaria, canny)
    bincanny_resized = cv2.resize(bincanny, (600, 800))
    #cv2.imshow('Img BINCANNY', bincanny_resized)

    # Detección de contornos
    contornos, _ = cv2.findContours(bincanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

    encontrado = False
    for c in contornos:
        area = cv2.contourArea(c)
        if area < int(imagen.shape[0]*imagen.shape[1] / 30):  # No tomar contornos pequeños
            continue

        perimetro = cv2.arcLength(c, True)
        aproximado = cv2.approxPolyDP(c, 0.02 * perimetro, True)

        if len(aproximado) == 4:
            puntos = ordenar_puntos(aproximado)
            encontrado = True
            break
    
    if encontrado:
        ancho, alto = 500, 700
        destino = np.float32([[0, 0], [ancho, 0], [ancho, alto], [0, alto]])
        M = cv2.getPerspectiveTransform(puntos, destino)
        dst = cv2.warpPerspective(imagen, M, (ancho,alto))
        imagen_resized = cv2.resize(imagen, (ancho, alto))

        dst_resized = cv2.resize(dst, (600, 800))
        #cv2.imshow('Img PERSPECTIVA', dst_resized)

        return encontrado, {
            "original": imagen_resized,
            "perspectiva": dst,
        }
    else:
        print(f"No se encontró una forma con 4 lados.")

    return False, None

def procesar_bolitas(dst):
    bol_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    bol_blur = cv2.GaussianBlur(bol_gray,(7, 7), 0) # Suavizado
    #cv2.imshow("bol_blur", bol_blur)
       
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(40, 100))
    bol_equ = clahe.apply(bol_blur)
    #cv2.imshow("bol_equ", bol_equ)

    '''
    bol_bin = cv2.threshold(bol_equ, (np.max(bol_blur)*(1/2)), 255, cv2.THRESH_BINARY_INV)[1]
    #bol_bin = cv2.adaptiveThreshold(bol_equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 12)
    cv2.imshow("bol_bin", bol_bin)

    # Remarcar las bolitas y eliminar lo demás
    kernel_clausura = np.ones((3, 3), np.float32)
    kernel_apertura = np.ones((9, 9), np.float32)
    kernel_erosion = np.ones((7, 7), np.float32)
    kernel_dilatacion = np.ones((11,11), np.float32) # (21,21) (11,11) 

    bol_limpia = cv2.morphologyEx(bol_bin, cv2.MORPH_CLOSE, kernel_clausura)
    bol_limpia = cv2.morphologyEx(bol_limpia, cv2.MORPH_OPEN, kernel_apertura)
    bol_limpia = cv2.erode(bol_limpia, kernel_erosion)
    bol_limpia = cv2.dilate(bol_limpia, kernel_dilatacion)
    cv2.imshow("bol_limpia", bol_limpia)
    '''

    crop = (500, 700) # (16, 20) (100, 140) (500, 700)
    bolitas = bol_equ[40:, 35:] # Imagen de resultado con títulos recortados
    bolitas = cv2.resize(bolitas, crop) # Reescalado

    return {
        "bol gris": bol_gray,
        #"bol ecualizada": bol_equ,
        "bol suavizada": bol_blur,
        #"bol binaria": bol_bin,
        #"bol limpia": bol_limpia,
        "bolitas": bolitas
    }

def procesar_imagen(imagen):
    resultados = {}
    encontrado, resultados = ajustar_perspectiva(imagen)

    if encontrado and resultados["perspectiva"] is not None:
        resultados.update(procesar_bolitas(resultados["perspectiva"]))
    
    return resultados

def identificar_respuestas(resultado):
    alto, ancho = resultado.shape
    alto_parte = alto // 20
    
    config_secciones = [
        {"num_columnas": 4, "recortar_izq": 15, "recortar_der": 20},  # sección 1
        {"num_columnas": 4, "recortar_izq": 25, "recortar_der": 15},  # sección 2
        {"num_columnas": 4, "recortar_izq": 45, "recortar_der": 10}   # sección 3
    ]

    respuestas_seccion1 = []
    respuestas_seccion2 = []
    respuestas_seccion3 = []

    conteo_bolitas = np.zeros((20, 3, 4), dtype=np.int32)

    for i in range(20):  # preguntas
        for j, cfg in enumerate(config_secciones):  # por sección
            num_columnas = cfg["num_columnas"]
            recortar_izq = cfg["recortar_izq"]
            recortar_der = cfg["recortar_der"]

            ancho_seccion = ancho // len(config_secciones)
            y1 = i * alto_parte
            y2 = (i + 1) * alto_parte
            x1 = j * ancho_seccion
            x2 = (j + 1) * ancho_seccion

            parte = resultado[y1:y2, x1:x2]
            parte = parte[:, recortar_izq:parte.shape[1] - recortar_der]

            ancho_parte = parte.shape[1] // num_columnas

            for k in range(num_columnas):
                x_ini = k * ancho_parte
                x_fin = (k + 1) * ancho_parte if k < num_columnas - 1 else parte.shape[1]
                region = parte[:, x_ini:x_fin]

                bol_bin = cv2.threshold(region, (np.max(region)*(1/2)), 255, cv2.THRESH_BINARY_INV)[1]
                #cv2.imshow("bol_bin", bol_bin)

                # Remarcar las bolitas y eliminar lo demás
                kernel_clausura = np.ones((3, 3), np.float32)
                kernel_apertura = np.ones((9, 9), np.float32)
                #kernel_erosion = np.ones((7, 7), np.float32)
                #kernel_dilatacion = np.ones((11,11), np.float32) # (21,21) (11,11) 

                bol_limpia = cv2.morphologyEx(bol_bin, cv2.MORPH_OPEN, kernel_apertura)
                bol_limpia = cv2.morphologyEx(bol_limpia, cv2.MORPH_CLOSE, kernel_clausura)
                
                #bol_limpia = cv2.erode(bol_limpia, kernel_erosion)
                # = cv2.dilate(bol_limpia, kernel_dilatacion)
                #cv2.imshow("bol_limpia", bol_limpia)

                blancos = cv2.countNonZero(bol_limpia)
                conteo_bolitas[i, j, k] = blancos

    valor_maximo_total = np.max(conteo_bolitas)
    umbral_minimo = (1/2) * valor_maximo_total

    for i in range(20):  # preguntas
        for j, cfg in enumerate(config_secciones):  # por sección
            num_columnas = cfg["num_columnas"]

            conteos = conteo_bolitas[i, j, :num_columnas]  # Extraer los valores válidos
            marcas_validas = [idx for idx, val in enumerate(conteos) if val >= umbral_minimo]

            opciones = ['A', 'B', 'C', 'D']
            respuesta = '[]'  
            # Clasificación
            if len(marcas_validas) == 0:
                respuesta = '[]'
            elif len(marcas_validas) == 1:
                marcas_validas = [idx for idx, val in enumerate(conteos) if val >= umbral_minimo or val >= max(conteos)/4]
                if len(marcas_validas) == 1:
                    respuesta = opciones[marcas_validas[0]]
                else:
                    respuesta = 'Mult.'  
            else:
                respuesta = 'Mult.'

            # Guardar en la lista correspondiente
            tupla = (i + 1, respuesta)
            if j == 0:
                respuestas_seccion1.append(tupla)
            elif j == 1:
                respuestas_seccion2.append(tupla)
            elif j == 2:
                respuestas_seccion3.append(tupla)

    # Convertir listas a arrays
    respuestas_seccion1 = np.array(respuestas_seccion1, dtype=object)
    respuestas_seccion2 = np.array(respuestas_seccion2, dtype=object)
    respuestas_seccion3 = np.array(respuestas_seccion3, dtype=object)

    return respuestas_seccion1, respuestas_seccion2, respuestas_seccion3

def evaluar():
    bol_eva = procesar_imagen(img_evaluar)["bolitas"]

    columnas = []  # Lista para guardar los 3 arreglos

    for textbox in textboxes:
        texto = textbox.get("1.0", "end-1c").replace('\n', '')
        caracteres = [c for c in texto if c in ('A', 'B', 'C', 'D')]

        while len(caracteres) < 20:
            caracteres.append('') 
        columnas.append(caracteres[:20]) 

    # Ahora tienes:
    ref_s1 = columnas[0]  # Primer textbox
    ref_s2 = columnas[1]  # Segundo textbox
    ref_s3 = columnas[2]  # Tercer textbox
    
    eva_s1, eva_s2, eva_s3 = identificar_respuestas(bol_eva)

    # Mostrar y colorear respuestas evaluadas por sección
    secciones_ref = [ref_s1, ref_s2, ref_s3]
    secciones_eva = [eva_s1, eva_s2, eva_s3]

    for j in range(3):  # para cada sección
        for i in range(20):  # para cada pregunta
            respuesta_ref = secciones_ref[j][i]
            respuesta_eva = secciones_eva[j][i][1]

            texto = f"{i+1}- {respuesta_eva}"

            if respuesta_eva not in {'A', 'B', 'C', 'D'}:
                color = "#5a5a5a"  # No marcada o inválida
            elif respuesta_eva == respuesta_ref:
                color = "#00bb0c"  # Correcta
            else:
                color = "#8e0000"  # Incorrecta

            labels_identificados[j][i].config(text=texto, fg=color)


    def comp_resp(ref_s, eva_s):
        aciertos = 0
        for i in range(20):
            resp_ref = ref_s[i]
            resp_eva = eva_s[i][1]
            if resp_ref in {'A', 'B', 'C', 'D'} and resp_eva in {'A', 'B', 'C', 'D'} and resp_ref == resp_eva:
                aciertos += 1

        return aciertos

    aciertos_s1 = comp_resp(ref_s1, eva_s1)
    aciertos_s2 = comp_resp(ref_s2, eva_s2)
    aciertos_s3 = comp_resp(ref_s3, eva_s3)

    aciertos = aciertos_s1 + aciertos_s2 + aciertos_s3
    calificacion = aciertos / 60 * 100

    aciertos_text = "Aciertos: " + str(aciertos) + "/60"
    calificacion_text = "Calificación: " + str(round(calificacion,1)) + "%"

    label_aciertos.config(text=aciertos_text)
    label_calificacion.config(text=calificacion_text)

    return aciertos, ref_s1, ref_s2, ref_s3, eva_s1, eva_s2, eva_s3

def validar_input(event, textbox):
    char = event.char.upper()
    keysym = event.keysym
    
    # Permitir teclas especiales
    if keysym in ("BackSpace", "Delete", "Left", "Right", "Up", "Down", "Tab", "Return"):
        return
    
    # Verificar que no haya más de 20 líneas
    contenido = textbox.get("1.0", "end-1c")
    if len(contenido) >= 20:
        return "break"

    # Detectar combinación Ctrl (event.state & 0x4)
    ctrl = (event.state & 0x4) != 0
    keysym = event.keysym.lower()

    if ctrl and keysym in ("c", "x", "v"):  # Ctrl+C, Ctrl+X, Ctrl+V
        return  # Permitir copiar, cortar, pegar

    char = event.char.upper()

    # Solo permitir letras A, B, C, D
    if char not in ("A", "B", "C", "D"):
        return "break"
    
    # Insertar con color según la letra
    colores = {
        "A": "azul",
        "B": "naranja",
        "C": "purpura",
        "D": "rosa"
    }

    tag = colores.get(char, "")
    textbox.insert("insert", char, tag)

    return "break"

# Crear la ventana principal
root = tk.Tk()
root.title("Sistema Automático de Evaluación de Exámenes de Hoja de Respuestas")
root.geometry("1100x800")

# Título en la parte superior
titulo = tk.Label(root, text="Sistema Automático de Evaluación de\nExámenes de Hoja de Respuestas", font=("Arial", 18, "bold", "italic"), fg="#0f3257")
titulo.pack(pady=20)

# Crear un frame para centrar para las imágenes o inputs
frame_central = tk.Frame(root)
frame_central.pack(pady=5, anchor="center")

# Agregar etiquetas debajo de los recuadros
label_referencia = Label(frame_central, text="Introducir respuestas", font=("Arial", 14, "italic"))
label_referencia.grid(row=0, column=0)

label_evaluar = Label(frame_central, text="Examen a evaluar", font=("Arial", 14, "italic"))
label_evaluar.grid(row=0, column=1)

# Crear un sub-frame para colocar los 3 textboxes
frame_textboxes = tk.Frame(frame_central)
frame_textboxes.grid(row=1, column=0, pady=20)

textboxes = []
num_filas = 20

for i in range(3):
    # Frame contenedor de label y textbox
    frame_col = tk.Frame(frame_textboxes)
    frame_col.grid(row=0, column=i, padx=(0 if i == 0 else 30, 0))  # padding solo a la izquierda de columnas 1 y 2

    # Frame para labels numerados, en columna 0
    frame_labels = tk.Frame(frame_col)
    frame_labels.grid(row=0, column=0, pady=5, sticky="ns")

    for n in range(1, num_filas + 1):
        label = tk.Label(frame_labels, text=f"{n}- ", font=("Arial", 8, "bold"))
        label.pack(anchor="e", pady=0)  # Sin espacio vertical extra

    # Textbox en columna 1, sin padx para que esté pegado a los labels
    textbox = tk.Text(frame_col, width=1, height=num_filas, font=("Courier", 13, "bold"), bg="#dedede")
    textbox.grid(row=0, column=1)

    # Configurar colores por letra (tags)
    textbox.tag_configure("azul", foreground="#0084FF")
    textbox.tag_configure("naranja", foreground="#FF9900")
    textbox.tag_configure("purpura", foreground="#7E00B9")
    textbox.tag_configure("rosa", foreground="#FF0062")

    textbox.bind("<Key>", lambda event, t=textbox: validar_input(event, t))
    textboxes.append(textbox)

frame_examen = tk.Frame(frame_central)
frame_examen.grid(row=1, column=1, pady=20, padx=50)

# Canvas para el examen por evaluar
canvas_evaluar = tk.Canvas(frame_examen, width=300, height=450, bg="gray")
canvas_evaluar.grid(row=0)

# Botón para seleccionar examen
boton_exameneva = tk.Button(frame_examen, text="Seleccionar examen", bg="#ffb368", command=lambda: seleccionar_imagen(canvas_evaluar, "Examen a Evaluar"), font=("Arial", 13, "bold"), width=20)
boton_exameneva.grid(row=1, pady=20)

# Frame para mostrar respuestas identificadas
frame_resultados = tk.Frame(frame_central)
frame_resultados.grid(row=1, column=2)

label_resultados = tk.Label(frame_central, text="Respuestas identificadas", font=("Arial", 14, "italic"))
label_resultados.grid(row=0, column=2)

# Crear subframes por sección
labels_identificados = [[], [], []]  # Tres listas para guardar los labels

for j in range(3):
    subframe = tk.Frame(frame_resultados)
    if i != 1: subframe.pack(side="left", padx=10)
    
    for i in range(1, 21):
        lbl = tk.Label(subframe, text=f"{i}- ", font=("Arial", 10, "bold"))
        lbl.pack(anchor="w")
        labels_identificados[j].append(lbl)


# Frame para aciertos, calificación y botones en la misma fila
frame_abajo = tk.Frame(frame_central)
frame_abajo.grid(row=2, column=1)

# Botón Evaluar
boton_evaluar = tk.Button(frame_abajo, text="Evaluar", command=evaluar, bg="#0cc945", fg="white", width=10, height=1, font=("Arial", 13, "bold"))
boton_evaluar.grid(row=0, column=0, padx=10)
boton_evaluar.grid_remove()  # Ocultar inicialmente

# Botón Borrar
boton_borrar = tk.Button(frame_abajo, text="Borrar", command=borrar_imagenes, bg="#c90c0c", fg="white", width=10, height=1, font=("Arial", 13, "bold"))
boton_borrar.grid(row=1, column=0, padx=10)
boton_borrar.grid_remove()  # Ocultar inicialmente

# Etiqueta Aciertos
label_aciertos = tk.Label(frame_abajo, text="Aciertos: -/60", font=("Arial", 14))
label_aciertos.grid(row=0, column=1, padx=10)

# Etiqueta Calificación
label_calificacion = tk.Label(frame_abajo, text="Calificación: -%", font=("Arial", 14))
label_calificacion.grid(row=1, column=1, padx=10)

# Mostrar la interfaz
root.mainloop()

