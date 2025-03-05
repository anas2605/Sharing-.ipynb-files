import math
# 4. Std. deviation function 
def standardDeviation(data):
    mean = sum(data) / len(data)    
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    # Return the standard deviation (square root of variance)
    return math.sqrt(variance)


def correlationCoefficient(x, y):
    if len(x) != len(y):
        print("The two input vectors must have the same length.")
    # Calculate means of x and y
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    # Calculate the covariance numerator
    covariance = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    # Return the correlation coefficient
    return covariance / (len(x) * standardDeviation(x) * standardDeviation(y))