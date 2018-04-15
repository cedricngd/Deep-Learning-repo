from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import pandas as pd
import tflearn
from tflearn.data_utils import to_categorical

#Importation des donnees a partir des fichiers csv

#Donnees d'entrainement
trainx =pd.read_csv('./input/csvTrainImages 13440x1024.csv',header=None)
trainy =pd.read_csv('./input/csvTrainLabel 13440x1.csv',header=None)

#Donnees de test
testx =pd.read_csv('./input/csvTestImages 3360x1024.csv',header=None)
testy =pd.read_csv('./input/csvTestLabel 3360x1.csv',header=None)

#Mise a l'echelle des donnees
trainx = trainx.values.astype('float32')/255
trainy = trainy.values.astype('int32')-1


testx = testx.values.astype('float32')/255
testy = testy.values.astype('int32')-1

"""
# Nous allons separer l'ensemble de test en 5 parties pour effectuer de la crossvalidation

# Premiere separation
valx_1 = trainx[0:2688]
valy_1 = trainy[0:2688]
trainx_1 = trainx[2688:13440]
trainy_1 = trainy[2688:13440]


#Deuxieme separation
valx_2 = trainx[2688:5376]
valy_2 = trainy[2688:5376]
trainx_2_a = trainx[0:2688]
trainx_2_b = trainx[5376:13440]
trainx_2 = np.concatenate((trainx_2_a,trainx_2_b),axis=0)
trainy_2_a = trainy[0:2688]
trainy_2_b = trainy[5376:13440]
trainy_2 = np.concatenate((trainy_2_a,trainy_2_b),axis=0)


#Troisieme separation
valx_3 = trainx[5376:8064]
valy_3 = trainy[5376:8064]
trainx_3_a = trainx[0:5376]
trainx_3_b = trainx[8064:13440]
trainx_3 = np.concatenate((trainx_3_a,trainx_3_b),axis=0)
trainy_3_a = trainy[0:5376]
trainy_3_b = trainy[8064:13440]
trainy_3 = np.concatenate((trainy_3_a,trainy_3_b),axis=0)


#Quatrieme separation
valx_4 = trainx[8064:10752]
valy_4 = trainy[8064:10752]
trainx_4_a = trainx[0:8064]
trainx_4_b = trainx[10752:13440]
trainx_4 = np.concatenate((trainx_4_a,trainx_4_b),axis=0)
trainy_4_a = trainy[0:8064]
trainy_4_b = trainy[10752:13440]
trainy_4 = np.concatenate((trainy_4_a,trainy_4_b),axis=0)


#Cinquieme separation
valx_5 = trainx[10752:13440]
valy_5 = trainy[10752:13440]
trainx_5 = trainx[0:10752]
trainy_5 = trainy[0:10752]

"""
#Fonction qui permet d'effectuer des batch d'une taille donnee en argument
def next_batch(num,data,labels):
	idx = np.arange(0, len(data))
	np.random.shuffle(idx)
	idx = idx[:num]
	data_shuffle = [data[ i] for i in idx]
	labels_shuffle = [labels[ i] for i in idx]
	return np.asarray(data_shuffle), np.asarray(labels_shuffle)

#Fonction qui convertit un np-array en one-hot-vector
def convert_to_one_hot(my_np_array):
	values = array(my_np_array)
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(values)
	my_np_array = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
	my_np_array = my_np_array.fit_transform(integer_encoded)
	return my_np_array
"""
#Conversions des labels de validation en one hot vector
valy_1=convert_to_one_hot(valy_1)
valy_2=convert_to_one_hot(valy_2)
valy_3=convert_to_one_hot(valy_3)
valy_4=convert_to_one_hot(valy_4)
valy_5=convert_to_one_hot(valy_5)

#Conversion des labels d'entrainement en one hot vector
trainy_1 = convert_to_one_hot(trainy_1)
trainy_2 = convert_to_one_hot(trainy_2)
trainy_3 = convert_to_one_hot(trainy_3)
trainy_4 = convert_to_one_hot(trainy_4)
trainy_5 = convert_to_one_hot(trainy_5)
"""

#Conversion en one hot vector
trainy = convert_to_one_hot(trainy)
testy = convert_to_one_hot(testy)


NUM_DIGITS = 1024 # 32x32 pixels
NUM_HIDDEN = 1024 # Nombre de neurones par couche ( hyperparametre )
NUM_CLASSES = 28   # 28 lettres dans l'alphabet arabe

# PLaceholders des donnees d'entree et de verite terrain
X = tf.placeholder(tf.float32, [None, NUM_DIGITS]) # donnees entree
Y = tf.placeholder(tf.float32, [None, NUM_CLASSES]) # verite terrain

# initialisation aleatoire des poids
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
w_h1 = init_weights([NUM_DIGITS, NUM_HIDDEN])
w_h2 = init_weights([NUM_HIDDEN, NUM_HIDDEN])
#w_h3 = init_weights([NUM_HIDDEN, NUM_HIDDEN])
#w_h4 = init_weights([NUM_HIDDEN, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, NUM_CLASSES])

# Definition du reseau
def model(X, w_h1, w_h2, w_o):
    h1 = tf.nn.relu(tf.matmul(X, w_h1))
    h2 = tf.nn.relu(tf.matmul(h1, w_h2))
    #h3 = tf.nn.relu(tf.matmul(h2, w_h3))
    #h4 = tf.nn.relu(tf.matmul(h3, w_h4))
    return tf.matmul(h2, w_o)

# Calcul de la sortie Y_p pour une entree X
Y_p = model(X, w_h1, w_h2, w_o)

# Fonction de cout et optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_p,labels= Y))
optimization_algorithm = tf.train.GradientDescentOptimizer(0.5).minimize(cost_function)

# Lancer une session interactive tensorflow
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Precision
correct_prediction = tf.equal(tf.argmax(Y_p,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Entrainement du reseau puis test. A effectuer a la fin apres avoir tune les hyperparametres

for iteration in range(20000):
   A,B = next_batch(50,trainx,trainy) # every batch of 50 images
   if iteration%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={X:A, Y: B})
      print("batch: %d, training accuracy: %g"%(iteration, train_accuracy))
   optimization_algorithm.run(feed_dict={X: A, Y: B})

test_result = "\n\n Test accuracy: %g"%accuracy.eval(feed_dict={X: testx, Y: testy})
print(test_result)

"""
#Decommentez ces 9 lignes pour la premiere etape de la cross validation
for iteration in range(20000):
   A,B = next_batch(50,trainx_1,trainy_1) # every batch of 50 images
   if iteration%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={X:A, Y: B})
      print("batch: %d, training accuracy: %g"%(iteration, train_accuracy))
   optimization_algorithm.run(feed_dict={X: A, Y: B})

validation_1 = "\n\n Validation accuracy for the first validation set : %g"%accuracy.eval(feed_dict={X: valx_1, Y: valy_1})
print(validation_1)

#Decommentez ces 9 lignes pour la deuxieme etape de la cross validation
for iteration in range(20000):
   A,B = next_batch(50,trainx_2,trainy_2) # every batch of 50 images
   if iteration%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={X:A, Y: B})
      print("batch: %d, training accuracy: %g"%(iteration, train_accuracy))
   optimization_algorithm.run(feed_dict={X: A, Y: B})

validation_2 = "\n\n Validation accuracy for the second validation set : %g"%accuracy.eval(feed_dict={X: valx_2, Y: valy_2})
print(validation_2)


#Decommentez ces 9 lignes pour la troisieme etape de la cross validation
for iteration in range(20000):
   A,B = next_batch(50,trainx_3,trainy_3) # every batch of 50 images
   if iteration%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={X:A, Y: B})
      print("batch: %d, training accuracy: %g"%(iteration, train_accuracy))
   optimization_algorithm.run(feed_dict={X: A, Y: B})

validation_3 = "\n\n Validation accuracy for the third validation set : %g"%accuracy.eval(feed_dict={X: valx_3, Y: valy_3})
print(validation_3)


#Decommentez ces 9 lignes pour la quatrieme etape de la cross validation
for iteration in range(20000):
   A,B = next_batch(50,trainx_4,trainy_4) # every batch of 50 images
   if iteration%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={X:A, Y: B})
      print("batch: %d, training accuracy: %g"%(iteration, train_accuracy))
   optimization_algorithm.run(feed_dict={X: A, Y: B})

validation_4 = "\n\n Validation accuracy for the fourth validation set : %g"%accuracy.eval(feed_dict={X: valx_4, Y: valy_4})

print(validation_4)


#Decommentez ces 9 lignes pour la cinquieme etape de la cross validation
for iteration in range(20000):
   A,B = next_batch(50,trainx_5,trainy_5) # every batch of 50 images
   if iteration%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={X:A, Y: B})
      print("batch: %d, training accuracy: %g"%(iteration, train_accuracy))
   optimization_algorithm.run(feed_dict={X: A, Y: B})

validation_5 = "\n\n Validation accuracy for the fifth validation set : %g"%accuracy.eval(feed_dict={X: valx_5, Y: valy_5})

print(validation_5)

"""

# Enregistrement du modele
saver = tf.train.Saver()
# > Variables to save 
tf.add_to_collection('vars', w_h1) 
tf.add_to_collection('vars', w_h2)
tf.add_to_collection('vars', w_o)
# > Save the variables to disk 
#save_path = saver.save(sess, "./tensorflow_model.ckpt") 
save_path = saver.save(sess, "./Model_MLP_arabic.ckpt") 
print("Model saved in file: %s" % save_path)

"""
# Restituion des variables enregistrees dans le modele
new_saver = tf.train.import_meta_graph('./tensorflow_model.ckpt.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
all_vars = tf.get_collection('vars')
i = 0


for v in all_vars:
    v_ = sess.run(v)
    if i == 0:
       w_h1 = v_ # restore w_h1 
    if i == 1:
       w_h2 = v_ # restore w_h2
    if i == 2:
       w_o = v_ # restore w_o 
    i = i + 1
"""
#print("Model restored correctly!")
