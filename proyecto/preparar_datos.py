import nltk
import re
import os
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
nltk.download("punkt")
# Ruta del archivo
FILE_PATH = "DonQuijote.txt"

# Leer el texto
with open(FILE_PATH, "r", encoding="utf-8") as f:
    texto = f.read()

texto = re.sub(r"\s+", " ", texto) 
texto = texto.replace("“", "\"").replace("”", "\"")


letras = list(texto)
print(f"Letras: {letras[:20]}")

# Tokenización por palabra
palabras = word_tokenize(texto)
print(f"Palabras: {palabras[:20]}")
# Tokenización por oración
oraciones = sent_tokenize(texto)
print(f"Oraciones: {oraciones[:2]}")
# Secuencias supervisadas (ejemplo por palabra)
def generar_secuencias(tokens, ventana=5):
    secuencias = []
    for i in range(ventana, len(tokens)):
        secuencia = tokens[i - ventana:i]
        objetivo = tokens[i]
        secuencias.append((" ".join(secuencia), objetivo))
    return secuencias

secuencias_palabras = generar_secuencias(palabras, ventana=5)
print("Ejemplo secuencia por palabra:")
for entrada, salida in secuencias_palabras[:3]:
    print(f"Input: {entrada} -> Output: {salida}")
