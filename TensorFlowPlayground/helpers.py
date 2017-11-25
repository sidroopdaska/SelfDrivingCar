from collections import defaultdict
import matplotlib.pyplot as plt
import time


class ModelTrainingLogger(object):
    def __init__(self, n_epochs, use_learning_curves=True):
        if use_learning_curves:
            self.use_learning_curves = use_learning_curves
            self.fig, self.axes = plt.subplots(2, sharex=True)
            self.lines = {}
            self.data = defaultdict(list)

            # Setup the loss function vs epochs plot
            self.axes[0].set_xlim(0, n_epochs)
            self.axes[0].set_ylim(0, 2.0)
            self.lines['loss'] = self.axes[0].plot([], [], 'r-')[0]

            # Setup the training/validation accuracy vs epochs plot
            self.axes[1].set_ylim(0, 100)
            self.lines['train_acc'] = self.axes[1].plot([], [], 'r-', label='Train Accuracy')[0]
            self.lines['val_acc'] = self.axes[1].plot([], [], 'b-', label='Validation Accuracy')[0]
            self.axes[1].legend(loc='lower right')

            plt.show(block=False)

    def update(self, epoch_i, loss, tr_acc, val_acc):
        print("Epoch: {}  Loss: {:.4f} Tr Accuracy: {:.4f}  Val Accuracy: {:.4f}"
              .format(epoch_i, loss, tr_acc, val_acc))

        if not self.use_learning_curves:
            return

        self.data['loss'].append(loss)
        self.data['epochs'].append(epoch_i)
        self.data['train_acc'].append(tr_acc)
        self.data['val_acc'].append(val_acc)

        self.lines['loss'].set_data(self.data['epochs'], self.data['loss'])
        self.lines['train_acc'].set_data(self.data['epochs'], self.data['train_acc'])
        self.lines['val_acc'].set_data(self.data['epochs'], self.data['val_acc'])

        plt.draw()
        plt.pause(1e-17)
        time.sleep(0.1)

