from typing import Tuple, List, Any, Optional, cast
from enum import Enum
import math
import tensorflow as tf
import numpy as np
import teddy as td
import nnOrsi3 as nn
import time
import pandas as pd


# This is just a regular 2-layer neural network
class ModelClassic:
    def __init__(self, in_size:int, out_size:int) -> None:
        self.h0 = nn.LayerLinear(in_size, 30)
        self.h2 = nn.LayerLinear(30, out_size)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)
        self.params = self.h0.params + self.h2.params

    # x is a batch of input values of shape (b, f), where b is the batch size and f is the number of input (feature) values.
    # Returns predicted labels in a tensor of shape (b, l), where l is the number of output (label) values.
    def activate(self, x: tf.Tensor) -> tf.Tensor:
        y = self.h0.act(x)
        y = tf.nn.tanh(y)
        y = self.h2.act(y)
        y = tf.nn.tanh(y)
        return y

    # x is a batch of input values of shape (b, f), where b is the batch size and f is the number of input (feature) values.
    # y is the corresponding batch of labels of shape (b, l).
    # Activates the network with a mini-batch and computes the output cost.
    def cost(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.reduce_sum(tf.square(y - self.activate(x)), axis = 1), axis = 0)

    # Presents one mini-batch for gradient descent training.
    def refine(self, x: tf.Tensor, y: tf.Tensor) -> None:
        self.optimizer.minimize(lambda: self.cost(x, y), self.params)

    # Activates the network with a mini-batch and computes the total number of misclassified patterns.
    def misclassifications(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        preds = self.activate(x)
        if (y.shape[1] < 2): # binary classification
            err = tf.not_equal(tf.minimum(0, tf.sign(y - 0.5)), tf.minimum(0, tf.sign(preds - 0.5)))
        else: # multi-class
            err = tf.not_equal(tf.argmax(y, axis=1), tf.argmax(preds, axis=1))
        return tf.reduce_sum(tf.cast(err, tf.int32))



class NodeType(Enum):
    INPUT = 1
    ADD = 2
    NOT = 3
    LOGIC = 4
    CONSTANT = 5

def is_same(a: List[Any], b: List[Any]) -> bool:
    if a[0] != b[0]:
        return False
    if a[0] == NodeType.INPUT:
        return True if a[1] == b[1] else False
    elif a[0] == NodeType.ADD:
        if is_same(a[1], b[1]) and is_same(a[2], b[2]):
            return True
        if is_same(a[1], b[2]) and is_same(a[2], b[1]):
            return True
        return False
    elif a[0] == NodeType.NOT:
        return is_same(a[1], b[1])
    elif a[0] == NodeType.LOGIC:
        if a[2] != b[2]:
            return False
        if is_same(a[1], b[1]) and is_same(a[3], b[3]):
            return True
        if is_same(a[1], b[3]) and is_same(a[3], b[1]):
            return True
        return False
    elif a[0] == NodeType.CONSTANT:
        return True if a[1] == b[1] else False
    else:
        raise ValueError(f'Unrecognized node type: {a}')

def simplify(desc: List[Any]) -> Optional[List[Any]]:
    if desc[0] == NodeType.INPUT:
        return None
    elif desc[0] == NodeType.ADD:
        x = simplify(desc[1])
        if x is not None:
            desc[1] = x
            return desc
        x = simplify(desc[2])
        if x is not None:
            desc[2] = x
            return desc
        return None
    elif desc[0] == NodeType.NOT:
        x = simplify(desc[1])
        if x is not None:
            desc[1] = x
            return desc
        if desc[1][0] == NodeType.NOT: # remove double-not
            return cast(List[Any], desc[1][1])
        if desc[1][0] == NodeType.LOGIC: # combine 'not' with 'logical operator'
            if desc[1][2] == 'and':
                return [NodeType.LOGIC, desc[1][1], 'nand', desc[1][3]]
            elif desc[1][2] == 'or':
                return [NodeType.LOGIC, desc[1][1], 'nor', desc[1][3]]
            elif desc[1][2] == 'xor':
                return [NodeType.LOGIC, desc[1][1], 'nxor', desc[1][3]]
            elif desc[1][2] == 'uni':
                return [NodeType.LOGIC, desc[1][1], 'notuni', desc[1][3]]
            elif desc[1][2] == 'notuni':
                return [NodeType.LOGIC, desc[1][1], 'uni', desc[1][3]]
            elif desc[1][2] == 'nand':
                return [NodeType.LOGIC, desc[1][1], 'and', desc[1][3]]
            elif desc[1][2] == 'nor':
                return [NodeType.LOGIC, desc[1][1], 'or', desc[1][3]]
            elif desc[1][2] == 'nxor':
                return [NodeType.LOGIC, desc[1][1], 'xor', desc[1][3]]
        return None
    elif desc[0] == NodeType.LOGIC:
        x = simplify(desc[1])
        if x is not None:
            desc[1] = x
            return desc
        x = simplify(desc[3])
        if x is not None:
            desc[3] = x
            return desc
        if desc[2] == 'and':
            if desc[1][0] == NodeType.CONSTANT:
                if desc[1][1]:
                    return cast(List[Any], desc[3]) # (true and x) -> (x)
                else:
                    return [NodeType.CONSTANT, False] # (false and x) -> (false)
            if desc[3][0] == NodeType.CONSTANT:
                if desc[3][1]:
                    return cast(List[Any], desc[1]) # (x and true) -> (x)
                else:
                    return [NodeType.CONSTANT, False] # (x and false) -> (false)
            if is_same(desc[1], desc[3]):
                return cast(List[Any], desc[1]) # (x and x) -> (x)
        elif desc[2] == 'or':
            if desc[1][0] == NodeType.CONSTANT:
                if desc[1][1]:
                    return [NodeType.CONSTANT, True] # (true or x) -> (true)
                else:
                    return cast(List[Any], desc[3]) # (false or x) -> (x)
            if desc[3][0] == NodeType.CONSTANT:
                if desc[3][1]:
                    return [NodeType.CONSTANT, True] # (x or true) -> (true)
                else:
                    return cast(List[Any], desc[1]) # (x or false) -> (x)
            if is_same(desc[1], desc[3]):
                return cast(List[Any], desc[1]) # (x or x) -> (x)
        elif desc[2] == 'xor':
            if desc[1][0] == NodeType.CONSTANT:
                if desc[1][1]:
                    return [NodeType.NOT, desc[3]] # (true xor x) -> (not x)
                else:
                    return cast(List[Any], desc[3]) # (false xor x) -> (x)
            if desc[3][0] == NodeType.CONSTANT:
                if desc[3][1]:
                    return [NodeType.NOT, desc[1]] # (x xor true) -> (not x)
                else:
                    return cast(List[Any], desc[1]) # (x xor false) -> (x)
            if is_same(desc[1], desc[3]):
                return [NodeType.CONSTANT, False] # (x xor x) -> (false)
        elif desc[2] == 'nand':
            if desc[1][0] == NodeType.CONSTANT:
                if desc[1][1]:
                    return [NodeType.NOT, desc[3]] # (true nand x) -> (not x)
                else:
                    return [NodeType.CONSTANT, True] # (false nand x) -> (true)
            if desc[3][0] == NodeType.CONSTANT:
                if desc[3][1]:
                    return [NodeType.NOT, desc[1]] # (x nand true) -> (not x)
                else:
                    return [NodeType.CONSTANT, True] # (x nand false) -> (true)
            if is_same(desc[1], desc[3]):
                return [NodeType.NOT, desc[1]] # (x nand x) -> (not x)
        elif desc[2] == 'nor':
            if desc[1][0] == NodeType.CONSTANT:
                if desc[1][1]:
                    return [NodeType.CONSTANT, False] # (true nor x) -> (false)
                else:
                    return [NodeType.NOT, desc[3]] # (false nor x) -> (not x)
            if desc[3][0] == NodeType.CONSTANT:
                if desc[3][1]:
                    return [NodeType.CONSTANT, False] # (x nor true) -> (false)
                else:
                    return [NodeType.NOT, desc[1]] # (x nor false) -> (not x)
            if is_same(desc[1], desc[3]):
                return [NodeType.NOT, desc[1]] # (x nor x) -> (not x)
        elif desc[2] == 'nxor':
            if desc[1][0] == NodeType.CONSTANT:
                if desc[1][1]:
                    return cast(List[Any], desc[3]) # (true nxor x) -> (x)
                else:
                    return [NodeType.NOT, desc[3]] # (false nxor x) -> (not x)
            if desc[3][0] == NodeType.CONSTANT:
                if desc[3][1]:
                    return cast(List[Any], desc[1]) # (x nxor true) -> (x)
                else:
                    return [NodeType.NOT, desc[1]] # (x nxor false) -> (not x)
            if is_same(desc[1], desc[3]):
                return [NodeType.CONSTANT, True] # (x nxor x) -> (true)
        return None
    elif desc[0] == NodeType.CONSTANT:
        return None
    else:
        raise ValueError(f'Unrecognized node type: {desc[0]}')

def simplify_logic(desc: List[Any]) -> List[Any]:
    while True:
        simpler = simplify(desc)
        if simpler is None:
            return desc
        else:
            desc = simpler

def print_description(desc: List[Any]) -> None:
    desc = simplify_logic(desc)
    if desc[0] == NodeType.INPUT:
        print(f'{desc[1]}', end='')
    elif desc[0] == NodeType.ADD:
        print_description(desc[1])
        print(' + ', end='')
        print_description(desc[2])
    elif desc[0] == NodeType.NOT:
        print('not (', end='')
        print_description(desc[1])
        print(')', end='')
    elif desc[0] == NodeType.LOGIC:
        print('(', end='')
        print_description(desc[1])
        print(f') {desc[2]} (', end='')
        print_description(desc[3])
        print(')', end='')
    elif desc[0] == NodeType.CONSTANT:
        print('true' if desc[1] else 'false', end='')
    else:
        raise ValueError(f'Unrecognized node type: {desc[0]}')

# This is an experimental model that uses a fuzzy logic layer.
class ModelFuzzyLogic:
    def __init__(self, in_size:int, out_size:int) -> None:
        self.h3 = nn.LayerAllPairings(in_size)
        h3_outsize = in_size * (in_size - 1 + 4)
        h4_outsize = h3_outsize // 2
        self.h4 = nn.LayerFuzzyLogic(h4_outsize)
        self.h5 = nn.LayerFeatureSelector(h4_outsize, 16)
        mid_size = 6
        self.h7 = nn.LayerAllPairings(mid_size)
        h7_outsize = mid_size * (mid_size - 1 + 4)
        h8_outsize = h7_outsize // 2
        self.h8 = nn.LayerFuzzyLogic(h8_outsize)
        self.h9 = nn.LayerFeatureSelector(h8_outsize, out_size)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)
        self.params = self.h4.params + self.h5.params + self.h8.params + self.h9.params

    # x is a batch of input values of shape (b, f), where b is the batch size and f is the number of input (feature) values.
    # Returns predicted labels in a tensor of shape (b, l), where l is the number of output (label) values.
    def activate(self, x: tf.Tensor) -> tf.Tensor:
        y = self.h3.act(x)
        y = self.h4.act(y)
        y = self.h5.act(y)
        y = tf.nn.tanh(y)
        y = self.h7.act(y)
        y = self.h8.act(y)
        y = self.h9.act(y)
        return y

    # x is a batch of input values of shape (b, f), where b is the batch size and f is the number of input (feature) values.
    # y is the corresponding batch of labels of shape (b, l).
    # Activates the network with a mini-batch and computes the output cost.
    def cost(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.reduce_sum(tf.square(y - self.activate(x)), axis = 1), axis = 0)

    # Presents one mini-batch for gradient descent training.
    def refine(self, x: tf.Tensor, y: tf.Tensor) -> None:
        self.optimizer.minimize(lambda: self.cost(x, y), self.params)
        self.h4.clip_weights()
        self.h5.clip_weights()
        self.h5.clip_weights()
        self.h9.clip_weights()

    # Activates the network with a mini-batch and computes the total number of misclassified patterns.
    def misclassifications(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        preds = self.activate(x)
        if (y.shape[1] < 2): # binary classification
            err = tf.not_equal(tf.minimum(0, tf.sign(y - 0.5)), tf.minimum(0, tf.sign(preds - 0.5)))
        else: # multi-class
            err = tf.not_equal(tf.argmax(y, axis=1), tf.argmax(preds, axis=1))
        return tf.reduce_sum(tf.cast(err, tf.int32))

    # Describes the specified unit in the last layer of the provided list
    def describe_unit(self, layers:List[nn.Layer], unit:int) -> List[Any]:
        if len(layers) == 0:
            return [NodeType.INPUT, unit]
        ll = layers[-1]
        if isinstance(ll, nn.LayerFeatureSelector):
            w = ll.weights.numpy()
            assert len(w.shape) == 2
            assert unit < w.shape[1], f'unit {unit} out of range {w.shape[1]}'
            result: Optional[List[Any]] = None
            maxmag = 0.
            for i in range(w.shape[0]):
                maxmag = max(maxmag, abs(w[i,unit]))
            maxmag = min(0.5, maxmag)
            for i in range(w.shape[0]):
                if abs(w[i,unit]) >= maxmag:
                    node = self.describe_unit(layers[:-1], i)
                    if abs(w[i,unit]) > 1.01:
                        raise ValueError('weight out of range: {w[i,unit]}')
                    if w[i,unit] < 0:
                        node = [NodeType.NOT, node]
                    if result:
                        result = [NodeType.ADD, result, node]
                    else:
                        result = node
            if not result:
                raise ValueError('Failed to find a result')
            return result
        elif isinstance(ll, nn.LayerFuzzyLogic):
            a = ll.alpha.numpy()
            assert len(a.shape) == 1
            assert unit < a.shape[0], f'unit {unit} out of range {a.shape[0]}'
            bef = self.describe_unit(layers[:-1], unit)
            aft = self.describe_unit(layers[:-1], a.shape[0] + unit)
            if a[unit] < -0.333:
                return [NodeType.LOGIC, bef, 'and', aft]
            elif a[unit] < 0.333:
                return [NodeType.LOGIC, bef, 'uni', aft]
            else:
                return [NodeType.LOGIC, bef, 'or', aft]
        elif isinstance(ll, nn.LayerAllPairings):
            p = ll.pairings
            assert unit < len(p), f'unit {unit} out of range {len(p)}'
            in_size = (math.floor(math.sqrt(4 * len(p) + 9)) - 3) // 2
            i = p[unit]
            assert i < in_size + 2
            if i < in_size:
                return self.describe_unit(layers[:-1], i)
            elif i == in_size:
                return [NodeType.CONSTANT, False]
            elif i == in_size + 1:
                return [NodeType.CONSTANT, True]
            else:
                raise ValueError(f'out of range in value: {i}')
        else:
            raise ValueError(f'unsupported layer type: {type(ll)}')

    # Prints a logic expression to represent this network
    def describe(self) -> None:
        print('\nNearest approximating logic: (Integers are a zero-indexed input attribute)')
        layers = [self.h3, self.h4, self.h5, self.h7, self.h8, self.h9]
        w = self.h9.weights.numpy()
        first = True
        for unit in range(w.shape[1]):
            if first:
                first = False
            else:
                print(', ', end='')
            description: List[Any] = self.describe_unit(layers, unit)
            print_description(description)
        print('\n\n')

# A list of available UCI datasets. Uncomment the ones you want to test
datasets = [
    'abalone',
    'arrhythmia',
    'audiology',
    'autos',
    'badges2',
    'balance-scale',
    'balloons',
    'breast-cancer',
    'breast-w',
    'bupa',
    'chess',
    'chess-KingRookVKingPawn',
    'colic',
    'credit-a',
    'credit-g',
    'dermatology',
    'diabetes',
    'ecoli',
    'glass',
    'heart-c',
    'heart-h',
    'heart-statlog',
    'hepatitis',
    'ionosphere',
    'iris',
    'kr-vs-kp',
    'labor',
    'lenses',
    'letter',
    'lungCancer',
    'lymph',
    'mushroom',
    'nursery',
    'ozone',
    'primary-tumor',
    'segment',
    'sonar',
    'spambase',
    'spectrometer',
    'splice',
    'teachingAssistant',
    'titanic',
    'vehicle',
    'vote',
    'vowel',
    'waveform-5000',
    'wine',
    'yeast',
]
#datasets = ["be"]
datasets = ["breast-cancer","diabetes","vehicle"]
#datasets = ['kimentes']
lista = []
# Run the test
for dataset in datasets:
    # Training variables
    try:
        repetitions = 1 # increase this value before publishing
        train_portion = 0.8 # 80/20 split
        training_epochs = 500 # it would be better to dynamically tune this
        batch_size = 32 # small batch because UCI datasets are pretty small data
        print("loading")
    # Load the data
        raw:td.Tensor = td.load_arff(f'uci_data/{dataset}.arff')
       # raw: td.Tensor = td.load_csv(f'uci_data/{dataset}.csv')
       # print(raw)
        print("tensor")
        features = raw[:,:-1].normalize().one_hot().data # inputs. All continuous values in the range [0,1]
        print("feature")
        labels = raw[:,-1:].normalize().one_hot().data # outputs. All continuous values in the range [0,1]
        print("index")
        indexes = np.arange(raw.data.shape[0]) # 0,1,2,3,4,...,n-1
        print("loaded")
    # Do some repetitions
        start_time = time.time()
        total_mis: List[float] = []
        total_tests = 0
        for i in range(repetitions):

        # Divide the raw data into the parts we will need
            np.random.shuffle(indexes)
            train_rows = max(1, math.floor(raw.data.shape[0] * train_portion))
            train_features = features[indexes[:train_rows]]
            train_labels = labels[indexes[:train_rows]]
            test_features = features[indexes[train_rows:]]
            test_labels = labels[indexes[train_rows:]]
            print("model")
        # Instantiate the models
            models = [
                ModelClassic(train_features.shape[1], train_labels.shape[1]),
                ModelFuzzyLogic(train_features.shape[1], train_labels.shape[1]),
            ]

        # Train the models
            train_indexes = np.arange(train_features.shape[0]) # 0,1,2,...
            print("train")
            for epoch in range(training_epochs):
                np.random.shuffle(train_indexes)
                for i in range(train_indexes.shape[0] // batch_size):
                    batch_features = train_features[train_indexes[i*batch_size:(i+1)*batch_size]]
                    batch_labels = train_labels[train_indexes[i*batch_size:(i+1)*batch_size]]
                    for model in models:
                        model.refine(batch_features, batch_labels) # type: ignore

        # Test the models
            while len(total_mis) < len(models):
                total_mis.append(0.)
            for j, model in enumerate(models):
                total_mis[j] += model.misclassifications(test_features, test_labels).numpy() # type: ignore
            total_tests += test_labels.shape[0]

        # Display the fuzzy logic model
            models[1].describe() # type: ignore

    # Report results on this dataset
        elapsed_time = time.time() - start_time
        print(f'{dataset}, {elapsed_time}', end='')
        adatok = [dataset, elapsed_time]
        for j, m in enumerate(total_mis):
            print(f', {total_mis[j] / total_tests}', end='')
            adatok.append(total_mis[j] / total_tests)
        print('')
        lista.append(adatok)
    except:
        pass
#dataframe = pd.DataFrame(data=np.array(lista).reshape(len(lista),4),columns=["adat","elteltido","test mis normal miss","test fuzzy miss"])
#dataframe.to_csv("simma",index=False)

