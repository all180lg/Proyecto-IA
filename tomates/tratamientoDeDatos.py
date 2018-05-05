from __future__ import division
import os
import cv2
import numpy as np
from PIL import Image
from os import listdir

def datosDeEntrenamiento(carpetaEntrada, carpetaSalida, archivos):
    for nombre_imagen in archivos:
        print nombre_imagen
        imagen = cv2.imread(carpetaEntrada + "/" +nombre_imagen)
        encontrar = ecnontrar_tomate(imagen)
        cv2.imwrite(carpetaSalida + "/" + nombre_imagen, encontrar)

def datosDeEntrenamiento2(carpetaEntrada, archivos, salida):
    for nombre_imagen in archivos:
        print nombre_imagen
        sacar_pixels(carpetaEntrada + "/" +nombre_imagen, salida)

def encontar_contorno(imagen):
    imagen = imagen.copy()
    img, contornos, jerarquia =\
        cv2.findContours(imagen, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = \
        [(cv2.contourArea(contorno), contorno) for contorno in contornos]
    mayor_contorno = max(contour_sizes, key=lambda x: x[0])[1]

    mascara = np.zeros(imagen.shape, np.uint8)
    cv2.drawContours(mascara, [mayor_contorno], -1, 255, -1)
    return mayor_contorno, mascara

"""
    con los datos encontrados de la imagen en su controno
    se cacula las dimensiones del cuadrdo y su hubicacion
"""

def contorno_rectangulo(imagen, contorno):
    imagenConElipse = imagen.copy()
    elipse = cv2.fitEllipse(contorno)
    factor_redn = 0.5
    sx = int((elipse[1][0]*factor_redn)/2)
    sy = int((elipse[1][1]*factor_redn)/2)
    x = int(elipse[0][0]) - sy
    y = int(elipse[0][1]) - sx
    imagenConElipse = imagenConElipse[y:(y + sx*2), x:(x + sy*2)]
    return imagenConElipse

"""
    Trata la imagen para poder encontrar el cuadrado
"""
def ecnontrar_tomate(imagen):
    imagen2 = imagen.copy()
    imagen3 = imagen.copy()
    imagen2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2HSV)
    max_dimension = max(imagen2.shape)
    scale = 700/max_dimension
    imagen2 = cv2.resize(imagen2, None, fx=scale, fy=scale)
    imagen3 = cv2.resize(imagen3, None, fx=scale, fy=scale)
    imagen_azul = cv2.GaussianBlur(imagen2, (7, 7), 0)
    min_rojo = np.array([0, 100, 80])
    max_rojo = np.array([10, 256, 256])

    mascara1 = cv2.inRange(imagen_azul, min_rojo, max_rojo)
    min_rojo2 = np.array([170, 100, 80])
    max_rojo2 = np.array([180, 256, 256])

    mascara2 = cv2.inRange(imagen_azul, min_rojo2, max_rojo2)
    mascara = mascara1 + mascara2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mascara_cerrada = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara_limpia = cv2.morphologyEx(mascara_cerrada, cv2.MORPH_OPEN, kernel)

    contorno_tomate_gramde, mascara_tomate = encontar_contorno(mascara_limpia)

    rectangulo_tomate = contorno_rectangulo(imagen3, contorno_tomate_gramde)
    #rectangulo_tomate = cv2.resize(rectangulo_tomate, (100, 50))
    # recortar(rectangulo_tomate)
    return rectangulo_tomate

def sacar_pixels(direccion, entrada):
    #se abre la imagen
    im = Image.open(direccion)
    #redimensiona la imagen con ANTIALIS algoritmo con menos perdida
    im = im.resize((40, 10), Image.ANTIALIAS)
    #im = im.resize((100, 50), Image.ANTIALIAS)
    #im.save("hola.jpg")
    #lectura de pixeles
    pixeles = im.load()
    #se abre el archivo para lectura escritura
    datosDeEntrada = open("datos.csv", "a")
    x, y = im.size
    for columna in range (y):
        for fila in range(x):
            #se separan los valores RGB y se escriben en el archivo
            r = str((pixeles[fila,columna][0])/255.)
            g = str((pixeles[fila,columna][1])/255.)
            b = str((pixeles[fila,columna][2])/255.)
            cadena = r[:r.find(".")+4] + " " + g[:g.find(".")+4] + " " + b[:b.find(".")+4] + ""
            datosDeEntrada.write(cadena)

    #pix[x,y] = value # Set the RGBA Value of the image (tuple) 
    datosDeEntrada.write(entrada)
    datosDeEntrada.write("\n")
    datosDeEntrada.close()

datosDeEntrenamiento("imagenesMaduras", "recortesMaduras", listdir("./imagenesMaduras"))
datosDeEntrenamiento("imagenesPodridas", "recortesPodridas", listdir("./imagenesPodridas"))
datosDeEntrenamiento("imagenesVerdes", "recortesVerdes", listdir("./imagenesVerdes"))

if(os.path.exists("datos.csv")== True):
    os.remove("datos.csv")

datosDeEntrenamiento2("recortesMaduras", listdir("./recortesMaduras"),"1")
datosDeEntrenamiento2("recortesPodridas", listdir("./recortesPodridas"),"1")
datosDeEntrenamiento2("recortesVerdes", listdir("./recortesVerdes"),"1")
