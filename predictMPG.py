#Predict MPG function
def predictMPG(model, scaler, input_data):
    input_data_scaled = scaler.transform([input_data])
    predicted_mpg = model.predict(input_data_scaled)
    return predicted_mpg[0]