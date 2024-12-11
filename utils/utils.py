import numpy
from sklearn.utils import class_weight

# Deve ser utilizada com labels de tipo int
def calculate_weight_dict(y):
    weights = class_weight.compute_class_weight(
        'balanced',
        classes=numpy.unique(y),
        y=y
    )
    weight_dict = dict(enumerate(weights))
    return weight_dict

'''
    Deve ser utilizada com labels NORMALIZADAS de tipo int,
    ou seja, label <= num_classes - 1, como as labels produzidas
    por sklearn.preprocessing.LabelEnconder
'''
class BiasInitializer:
    def __init__(self, y):
        self.y = y
    
    def calculate_biases(self):
        unique_values = numpy.unique(self.y)
        total_count = self.y.size
        temp = numpy.zeros(shape=len(unique_values))
        
        for value in unique_values:
            count = numpy.count_nonzero(self.y == value)
            class_prop = count / total_count
            temp[value] = class_prop / (1 - class_prop)
        
        return numpy.log(numpy.array(temp))