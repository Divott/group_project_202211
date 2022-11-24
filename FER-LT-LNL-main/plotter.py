import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


# At each epoch you need to mkdir str(epoch) in image/


# from torch.tensor to numpy

# np_arr = torch_tensor.detach().cpu().numpy()


# test code

# x = np.array([1,  2,  3,  4,  5,  6,  7, 8])
# y = np.array([20, 30, 5, 12, 39, 48, 50, 3])

# plt.plot(x, y)
# plt.title("Curve plotted using the given points")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.savefig('test_plotter.png')
# plt.close()

# data = {'C': 20, 'C++': 15, 'Java': 30,
#         'Python': 35}
# courses = list(data.keys())
# values = list(data.values())

# plt.bar(courses, values, color='maroon',
#         width=0.4)
# plt.xlabel("Courses offered")
# plt.ylabel("No. of students enrolled")
# plt.title("Students enrolled in different courses")
# plt.show()


def plot_train_loss(train_loss: np.array, epoch: int, model_code: int):
    num_steps = np.arange(len(train_loss))
    plt.plot(num_steps, train_loss)
    plt.title("Train loss in epoch %s(net %s)" % (str(epoch), str(model_code)))
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.savefig("images/%s/Train loss in epoch %s(net %s).png" %
                (str(epoch), str(epoch), str(model_code)))
    # plt.savefig('test_plotter.png')
    plt.close()


# y = np.zeros(10)
# for i in range(10):
#     y[i] = i+1
# plot_train_loss(y, 0)


def plot_test_loss(test_loss: np.array):
    num_steps = np.arange(len(test_loss))
    plt.plot(num_steps, test_loss)
    plt.title("Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(
        "images/Test loss.png")
    # plt.savefig('test_plotter.png')
    plt.close()


# y = np.zeros(10)
# for i in range(10):
#     y[i] = i+1
# plot_test_loss(y, 0)


def plot_confusion_matrix(confusion_matrix: np.array, epoch: int):
    # exp_classes = ['Neutral', 'Happy', 'Sad', 'Surprise',
    #                'Fear', 'Disgust', 'Anger', 'Contempt']
    exp_classes = ['Neutral', 'Happy', 'Sad', 'Surprise',
                   'Fear', 'Disgust', 'Anger']
    df_cm = pd.DataFrame(confusion_matrix, index=exp_classes,
                         columns=exp_classes)
    plt.figure()
    plt.title("Confusion matrix in epoch %s" % str(epoch))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.savefig("images/%s/Confusion matrix in epoch %s.png" %
                (str(epoch), str(epoch)))
    # plt.savefig('test_plotter.png')
    plt.close()


# y = np.zeros((8, 8))
# for i in range(8):
#     for j in range(8):
#         y[i, j] = i
# plot_confusion_matrix(y, 0)


def plot_loss_num(loss_num: np.array, epoch: int, cls: int, model_code: int):
    # loss_num = (loss - loss_min) / (loss_max - loss_min), the distribution is divided by 1%
    # exp_classes = ['Neutral', 'Happy', 'Sad', 'Surprise',
    #                'Fear', 'Disgust', 'Anger', 'Contempt', 'All']
    exp_classes = ['Neutral', 'Happy', 'Sad', 'Surprise',
                   'Fear', 'Disgust', 'Anger', 'All']
    indices = np.arange(101)
    plt.bar(indices, loss_num, color='maroon',
            width=0.8)
    plt.xlabel("Loss distribution(%)")
    plt.ylabel("Total number")
    plt.title("Num-loss(%s) in epoch %s(net %s)" %
              (exp_classes[cls], str(epoch), str(model_code)))
    plt.savefig("images/%s/Num-loss(%s) in epoch %s(net %s).png" %
                (str(epoch), exp_classes[cls], str(epoch), str(model_code)))
    # plt.savefig('test_plotter.png')
    plt.close()


# y = np.zeros(101)
# for i in range(101):
#     y[i] = i+1
# plot_loss_num(y, 0, 1)


def plot_prob_num(prob_num: np.array, epoch: int, cls: int, model_code: int):
    # exp_classes = ['Neutral', 'Happy', 'Sad', 'Surprise',
    #                'Fear', 'Disgust', 'Anger', 'Contempt', 'All']
    exp_classes = ['Neutral', 'Happy', 'Sad', 'Surprise',
                   'Fear', 'Disgust', 'Anger',  'All']
    indices = np.arange(101)
    plt.bar(indices, prob_num, color='maroon',
            width=0.4)
    plt.xlabel("Prob distribution(%)")
    plt.ylabel("Total number")
    plt.title("Num-prob(%s) in epoch %s(net %s)" %
              (exp_classes[cls], str(epoch), str(model_code)))
    plt.savefig("images/%s/Num-prob(%s) in epoch %s(net %s).png" %
                (str(epoch), exp_classes[cls], str(epoch), str(model_code)))
    # plt.savefig('test_plotter.png')
    plt.close()


# y = np.zeros(101)
# for i in range(101):
#     y[i] = 100*i
# plot_loss_num(y, 0, 2)
