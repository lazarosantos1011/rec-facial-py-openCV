import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImagemComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    #print(caminhos)
    faces = [] #guardando faces.
    ids = [] #guardando ids.
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY) #Lendo cada uma das imagems #Convertendo para escala de cinza, evita erro no treinamento do classificador.
        id = int(os.path.split(caminhoImagem)[-1].split(".")[1]) #Verificar e quebrar string quando tem um ponto, pegando somente posição e retorna somente os ids.
        #print(id) #mostra somente o id das fotos das pessoas na pasta.
        ids.append(id) #append adiciona elemento na lista.
        faces.append(imagemFace)
        #cv2.imshow("Face", imagemFace) #Titulo da janela que abre no processo.
        #cv2.waitKey(10) #10ms pra processamento de cada imagem.
    return np.array(ids), faces #converter lista de ids no tipo np.array para realizar o treinamento.

ids, faces = getImagemComId()
#print(faces) #retorna todos os ids ou imagens que estão na pasta fotos para treinamento.

print("Treinando...")
eigenface.train(faces, ids) #apredinzagem supervisionada, ID 1,1,1; foto 1,2,3 #Processo de treinamento: Leitura e aprendizado das imagens.
eigenface.write('classificadorEigen.yml') #Gravação deste arquivo responsável pela classificação dos registros, ex: id 1/2

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print("Treinamento realizado!")