from __future__ import division
import os
import sys
import cv2
import scipy as sp
import numpy as np
from PIL import Image
import neurolab as nl

def main(direccion):
    archivo = "imgTemporal.jpg"
    datosDeEntrada = "datosTemporales.csv"
    red1 = "maduros.rna"
    red2 = "podridos.rna"
    red3 = "verdes.rna"
    imagen = cv2.imread(direccion)
    cv2.imshow("Imagen a examinar, presione una tecla para continuar.",imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    imagen = filtrar(imagen)
    cv2.imwrite(archivo,imagen)
    datos = obtenerPixeles(archivo)
    archivoDeEntrada = open(datosDeEntrada, "a")
    archivoDeEntrada.write(datos)
    archivoDeEntrada.close()
    datos = np.matrix(sp.genfromtxt(datosDeEntrada, delimiter=" "))
    #Borras los archivos temporales, necesarios para 
    os.remove(archivo)
    os.remove(datosDeEntrada)
    #Carga los archivos de las redes neuronales
    r1 = nl.load(red1)
    r2 = nl.load(red2)
    r3 = nl.load(red3)
    #Resultados de las redes neuronales
    s1 = r1.sim(datos)
    s2 = r2.sim(datos)
    s3 = r3.sim(datos)

    comparacion(s1,s2,s3)

def filtrar(imagen):
    copia1 = imagen.copy()
    copia2 = imagen.copy()
    copia1 = cv2.cvtColor(copia1, cv2.COLOR_BGR2HSV)
    tamxy = max(copia1.shape)
    scale = 700/tamxy
    copia1 = cv2.resize(copia1, None, fx=scale, fy=scale)
    copia2 = cv2.resize(copia2, None, fx=scale, fy=scale)
    fAzul = cv2.GaussianBlur(copia1,(7, 7),0)
    f1 = cv2.inRange(fAzul, np.array([0, 100, 80]), np.array([10, 256, 256]))
    f2 = cv2.inRange(fAzul, np.array([170, 100, 80]), np.array([180, 256, 256]))
    f = f1 + f2
    datosDeElipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fC = cv2.morphologyEx(f, cv2.MORPH_CLOSE, datosDeElipse)
    fO = cv2.morphologyEx(fC, cv2.MORPH_OPEN, datosDeElipse)
    aRecortar, fT = hallarBorde(fO)
    recorte = ponerRectangulo(copia2, aRecortar)
    return recorte

def hallarBorde(imagen):
    imagen = imagen.copy()
    img, borde, escalas =\
        cv2.findContours(imagen, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = \
        [(cv2.contourArea(contorno), contorno) for contorno in borde]
    wrapper = max(contour_sizes, key=lambda x: x[0])[1]

    f = np.zeros(imagen.shape, np.uint8)
    cv2.drawContours(f, [wrapper], -1, 255, -1)
    return wrapper, f

def ponerRectangulo(imagen, contorno):
    imgElipse = imagen.copy()
    elipse = cv2.fitEllipse(contorno)
    sx = int((elipse[1][0]*0.5)/2)
    sy = int((elipse[1][1]*0.5)/2)
    x = int(elipse[0][0]) - sy
    y = int(elipse[0][1]) - sx
    imgElipse = imgElipse[y:(y + sx*2), x:(x + sy*2)]
    return imgElipse

def obtenerPixeles(imagen):
    im = Image.open(imagen)
    im = im.resize((40, 10), Image.ANTIALIAS)
    pixels = im.load()
    fil, col = im.size
    decimales = 4
    datos = ""
    for y in range (col):
        for x in range(fil):
            #se separan los valores RGB y se escriben en el archivo
            r = str((pixels[x,y][0])/255.)
            g = str((pixels[x,y][1])/255.)
            b = str((pixels[x,y][2])/255.)
            datos = datos + r[:r.find(".")+decimales] + " " + g[:g.find(".")+decimales] + " " + b[:b.find(".")+decimales] + " "
    return datos

def comparacion(maduro,podrido,g):
    m = maduro[0][1]**2
    p = podrido[0][0]**2
    v = g[0][2]**2
    if(m >= p) and (m >= v):
        print("\nSalida de la red: {}\n\nEl tomate esta maduro\n").format(m)
    elif ((p >= m) and (p >= v)):
        print("\nSalida de la red: {}\n\nEl tomate esta podrido\n").format(p)
    else:
        print("\nSalida de la red: {}\n\nEl tomate esta verde\n").format(v)

if __name__ == '__main__':
    main(sys.argv[1])