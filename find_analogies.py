import json

print("Importing tensorflow...")
import tensorflow as tf
import numpy as np
print("Importing data utilities...")
from utils import *
from data_utils import create_vocab_tables
from functools import partial

print("Loading in unigrams...")
with open('embeddings/unigrams.json' ,'r') as file:
    unigrams = json.loads(file.read())

print("Recreating vocab tables...")
unigrams = list(unigrams.keys())
vocab_table, inv_vocab_table = create_vocab_tables(unigrams)

print("Loading word embeddings...")
model = tf.keras.models.load_model('embeddings/w2v_model')

U = model.layers[0].get_weights()[0]
V = model.layers[1].get_weights()[0]
W = 0.5*(U+V)

print_analg = partial(print_analogies, U=W, vocab_table=vocab_table, inv_vocab_table=inv_vocab_table)

print_clos = partial(print_closest, U=W, vocab_table=vocab_table, inv_vocab_table=inv_vocab_table, K=6)


def get_analogy_inputs():
    a = input("A: ")
    b = input("B: ")
    c = input("C: ")
    return a, b, c

def get_closest_input():
    a = input("Root word: ")
    return a

def exit_script():
    print("\nThanks for using the analogy generator!")

if __name__=="__main__":
    print('\n\n'+'*'*10+' Analogy and Neighbor embedding locator ' +'*'*10+'\n\n')
    while True:
        choice = input("\nEnter [a] for analogy and [c] to find a words nearest neighbors. Enter [q] to quit\n")
        if choice=='q':
            exit_script()
            break
        elif choice=='a':
            a,b,c = get_analogy_inputs()
            try:
                print_analg(a,b,c)
            except:
                x=input("\nThere was an error in your inputs. Try again? [y/n]")
                if x!='y':
                    exit_script()
                    break
        elif choice=='c':
            a = get_closest_input()
            try:
                print_clos(a)
            except:
                x=input("\nThere was an error in your inputs. Try again? [y/n]")
                if x!='y':
                    exit_script()
                    break
