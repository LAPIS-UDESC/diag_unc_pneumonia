from keras import Sequential, Input
from keras import layers
from keras import optimizers
from keras import metrics
import keras
import keras_tuner as kt
from keras import backend
import numpy

'''
    Funções construtoras de modelos para
    tarefas de busca de hiperparametros
'''

def build_hp_mlp(hp, input_size, out_size):
    model = Sequential()
    model.add(Input(shape=(input_size,)))
    hidden_num = hp.Int('hidden_num', min_value=2, max_value=6, step=1)
    
    for i in range(hidden_num):
        model.add(
            layers.Dense(
                units=hp.Int(f'units{i}', min_value=16, max_value=2048, step=8),
                activation='relu',
                kernel_regularizer=keras.regularizers.L1L2(l1=0.01, l2=0.01)
            )
        )
        if hp.Boolean(f'dropout{i}'):
            model.add(layers.Dropout(rate=0.25))

    model.add(layers.Dense(out_size, activation='softmax'))
    
    lr = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling='log')
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        weighted_metrics=[metrics.SparseCategoricalAccuracy(name='accuracy')]
    )
    return model

class MLPHyperModel(kt.HyperModel):
    def __init__(self, inputs_num, outputs_num, bias_initializer):
        super().__init__()
        self.bias_initializer = bias_initializer
        self.inputs_num = inputs_num
        self.outputs_num = outputs_num
        
    def build(self, hp):
        backend.clear_session(free_memory=True)
        num_hidden_layers = hp.Int('num_hidden_layers', min_value=1, max_value=6, step=1)
        model = Sequential()
        model.add(Input(shape=(self.inputs_num,)))
        
        '''
            Valores muito altos de l1 e/ou l2 nos regularizadores de
            kernel podem massacrar os pesos da rede, especialmente se
            a range das features for pequena. Essas constantes precisam
            ser escolhidas com cautela.
        '''
        for i in range(num_hidden_layers):
            model.add(
                layers.Dense(
                    units=hp.Int(f'hidden_units_{i}', min_value=32, max_value=512, step=32),
                    activation='relu',
                    kernel_regularizer=keras.regularizers.L2(l2=0.01)
                )
            )
            model.add(layers.Dropout(rate=0.5))

        # Adicionando a ultima camada
        model.add(
            layers.Dense(
                name='output_layer',
                units=self.outputs_num,
                activation='softmax',
            )
        )
        
        # Configurando os viezes iniciais
        weights, _ = model.get_layer('output_layer').get_weights()
        new_biases = self.bias_initializer.calculate_biases()
        model.get_layer('output_layer').set_weights([weights, new_biases])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=[
                keras.metrics.CategoricalAccuracy(name='accuracy'),
                keras.metrics.F1Score(average='weighted'),
                keras.metrics.AUC(name='roc_auc', curve='ROC'),
                keras.metrics.AUC(name='pr_auc', curve='PR')
            ]
        )
        
        return model