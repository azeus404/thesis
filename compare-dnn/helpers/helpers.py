
#!/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
import itertools
"""
        Updated 22-09-2020
"""
def summarize_diagnostics(filename,history):
    """
    updated
    """
    plt.title('Training and validation  Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(np.arange(len(history.history['loss'])), history.history['loss'], color='blue', label='train')
    plt.plot(np.arange(len(history.history['val_loss'])),history.history['val_loss'], color='red', label='validation')
    plt.legend(['train','validation'],loc=0)
    plt.tight_layout()
    plt.savefig(filename + '_learn_plot_loss.png')
    plt.close()

def plot_history(filename,history):
	plt.plot(np.arange(len(history.history['accuracy'])),history.history['accuracy'], color='blue')
	plt.plot(np.arange(len(history.history['val_accuracy'])),history.history['val_accuracy'], color='red')
	plt.title('Training and validation accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['train','validation'],loc=0)
	plt.tight_layout()
	plt.savefig(filename + '_learn_plot_accuracy.png')
	plt.close()


def plot_confusion_matrix(filename,cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    """
    plt.figure(figsize=(40,40))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    accuracy = np.trace(cm)/float(np.sum(cm))
    misclass = 1 - accuracy

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        print('[*] Normalized confusion matrix')
    else:
        print('[*] Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}\n missclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(filename + '_cm.png')
    plt.close()
