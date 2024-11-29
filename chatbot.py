import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download("punkt")

# Datos de entrenamiento: síntomas y enfermedades asociadas
sintomas = [
    "Me siento triste y sin ganas de hacer nada",  # Depresión
    "Tengo ansiedad constante y no puedo dormir bien",  # Ansiedad
    "Escucho voces y siento que alguien me persigue",  # Esquizofrenia
    "Estoy muy cansado y no tengo energía durante el día",  # Fatiga crónica
    "No puedo concentrarme y me distraigo con facilmente",  # TDAH
    "Episodios alternados de mania euforia extrema y depresion, cambios bruscos de humor",  # Trastorno bipolar
]

enfermedades = [
    "Depresión",
    "Ansiedad",
    "Esquizofrenia",
    "Fatiga crónica",
    "TDAH",
    "Trastorno bipolar",
]

# Recomendaciones para cada enfermedad
recomendaciones = {
    "Depresión": "Busca apoyo profesional de de inmediato y programa una rutina diaria.",
    "Ansiedad": "Asegurate de dormir bien, seguir rutinas y ejercicios para la meditacion.",
    "Esquizofrenia": "Acude con un psiquiatra para que recete algun tratamiento.",
    "Fatiga crónica": "Duerme lo suficiente, manten una buena dieta, prueba una rutina de ejercicio.",
    "TDAH": "Organiza tus tareas, de preferencia en una lista, intenta trabajar en un lugar libre de distracciones.",
    "Trastorno bipolar": "Manten la calma ante situaciones espontaneas, acude con un profesional y sigue un tratamiento."
}

# Aqui es la vectorizacion básicamente la conversión a números
vectorizador = CountVectorizer()
X = vectorizador.fit_transform(sintomas)

# Aquí se presenta la clasificación de los datos
modelo = MultinomialNB()
modelo.fit(X, enfermedades)

# Aqui se Función del chatbot
def respuesta_chatbot(entrada_usuario):
    # Preprocesar la entrada del usuario y predice
    entrada_usuario_vectorizada = vectorizador.transform([entrada_usuario])
    prediccion = modelo.predict(entrada_usuario_vectorizada)
    enfermedad = prediccion[0]
    recomendacion = recomendaciones[enfermedad]
    return enfermedad, recomendacion

print("Chatbot de salud mental (Escribe 'salir' para terminar)")

while True:
    mensaje_usuario = input("Ernesto: ")
    if mensaje_usuario.lower() == "salir":
        print("Chatbot: Espero haberte ayudado, cuida de tu salud mental.")
        break
    
    enfermedad, recomendacion = respuesta_chatbot(mensaje_usuario)
    print(f"Chatbot: Según los síntomas descritos, puedes estar experimentando: {enfermedad}. {recomendacion}")