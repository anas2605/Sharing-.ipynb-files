import math
# 4. Std. deviation function 
def standardDeviation(data):
    mean = sum(data) / len(data)    
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    # Return the standard deviation (square root of variance)
    return math.sqrt(variance)