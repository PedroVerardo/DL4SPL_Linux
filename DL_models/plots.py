import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_results(epoch, train_loss, validation_loss):
    plt.figure(figsize=(20,6))
    plt.subplot(1,3,1)
    ax = sns.lineplot(y=np.array(train_loss), x=np.array(epoch), label="loss", palette="binary")
    ax.set(xlabel= 'epochs', ylabel= 'train_loss')
    plt.title("Train Loss")
    #----------------------------------------------------------------
    plt.subplot(1,3,2)
    ax = sns.lineplot(y=np.array(train_loss), x=np.array(epoch), label="test loss", palette="flare")
    ax.set(xlabel= 'epochs', ylabel= 'train_loss')
    ax = sns.lineplot(x=np.array(epoch), y=np.array(validation_loss), label="Loss Convergence", color="red")
    ax.set(xlabel= 'epochs', ylabel= 'validation_loss')
    plt.title("Train Vs Validation")
    #----------------------------------------------------------------
    plt.subplot(1,3,3)
    absolut_loss = np.subtract(validation_loss, train_loss)
    ax = sns.lineplot(x=np.array(epoch), y=absolut_loss, label="Loss Convergence", color="red")
    ax.set(xlabel= 'epochs', ylabel= 'Vloss - Tloss')
    plt.title("Loss Convergence")

def plot_original_vs_predicted(ytest, pred):
    plt.plot(ytest.cpu().detach().numpy()[:100], color = 'b')
    plt.plot(pred[:100], color = 'r', linestyle = 'dashed')
    plt.show()