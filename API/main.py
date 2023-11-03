import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

#Carregar modelo trainado
model = joblib.load('modelo_treinado.pkl')

#Criar instancia da fastapi
app = FastAPI()

#Definir o modelo paydentic para estrutura de entrada
class InputData(BaseModel):
    input: list[float]


#Criar fastapi com modelo
@app.post('/predict')
async def predict(input_data: InputData):
    data = np.array(input_data.input)
    prediction = model.predict(data.reshape(1,-1))
    return {'prediction': prediction.tolist()}