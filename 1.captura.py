# pip install opencv-contrib-python

import cv2
import numpy as np

classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
classificadorOlho = cv2.CascadeClassifier("haarcascade_eye.xml")
camera = cv2.VideoCapture(0)

# inicializar variáveis
amostra = 1
numeroAmostras = 25
largura, altura = 220, 220

id = input('Digite o seu identificador: ')
print('Capturando as faces...')

while (True):
    conectado, imagem = camera.read()
    imagemcinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesdetectadas = classificador.detectMultiScale(imagemcinza, scaleFactor = 1.5, minSize = (100,100))

    for (x,y,l,a) in facesdetectadas:
        cv2.rectangle(imagem, (x,y), (x+a, y+l), (0,0,255), 3)

        regiao = imagem[y:y+a, x:x+l]
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho)

        for (ox, oy, ol, oa)in olhosDetectados:
            cv2.rectangle(regiao, (ox,oy), (ox+ol, oy+oa), (0,255,0), 3)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(np.average(imagemcinza))
                if np.average(imagemcinza) >70:
                    imagemFace = cv2.resize(imagemcinza[y:y+a, x:x+l], (largura, altura))
                    cv2.imwrite('fotos/pessoa.'+ str(id) + '.' + str(amostra) + '.jpg', imagemFace )
                    print('[foto ' + str(amostra) + 'capturada com sucesso')
                    amostra +=1


    cv2.imshow("Face", imagem)
    cv2.waitKey(1)
    if (amostra >= numeroAmostras +1):
        break

print('Faces Capturadas com sucesso')
camera.release()
cv2.destroyAllWindows()
