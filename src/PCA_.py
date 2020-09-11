import numpy as np
import scipy
import matplotlib.pyplot as plt 
import seaborn as sns
import random


def ComputeCovariance(X):
    mu = np.mean(X, axis = 0)
    return mu, (X-mu).T.dot(X-mu)/len(X)

if __name__ == "__main__":
    #Part a
    # X_train, y_train, X_test, y_test = load_dataset()
    # mu, Sigma = ComputeCovariance(X_train)
    # l, V = np.linalg.eig(Sigma)
    # indices = [0,1,9,29,49]
    # print('The requested eigvals:', np.sort(l)[::-1][indices])
    # print('The sum of eigvals:', np.sum(l))

    # #Sort the eigvals and eigvecs:
    # sorted_indices = np.argsort(l)[::-1]
    # l = l[sorted_indices].astype('float')
    # V = V[:,sorted_indices].astype('float')

    # #Part c
    # train_err = []
    # test_err = []
    # k_range = np.arange(1,101)
    # for k in k_range:
    #     print('Current k:', k)
    #     train_prediction = (X_train - mu).dot(V[:,:k]).dot(V[:,:k].T) + mu
    #     train_err.append(np.mean(np.linalg.norm(X_train - train_prediction, axis = 1)**2))
    #     test_prediction = (X_test - mu).dot(V[:,:k]).dot(V[:,:k].T) + mu
    #     test_err.append(np.mean(np.linalg.norm(X_test - test_prediction, axis = 1)**2))

    # plt.figure(figsize = (15,10))
    # plt.plot(k_range, train_err, '-o', label = 'train_error')
    # plt.plot(k_range, test_err, '-o', label = 'test_error')
    # plt.title('A6c: PCA Errors')
    # plt.legend()
    # plt.xlabel('k')
    # plt.ylabel('error')
    # plt.savefig('figures/A6c_errors.pdf')
    # plt.show()

    # unexplained_variance = []
    # for k in k_range:
    #     unexplained_variance.append(1 - np.sum(l[:k])/np.sum(l))

    # plt.figure(figsize = (15,10))
    # plt.plot(k_range, unexplained_variance, '-o', label = 'unexplained_variance')
    # plt.title('A6c: PCA Fraction of Unexplained Variance')
    # plt.legend()
    # plt.xlabel('k')
    # plt.ylabel('unexplained_variance')
    # plt.savefig('figures/A6c_variance.pdf')
    # plt.show()


    # #Part d

    # num_row = 2
    # num_col = 5
    # fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    # for i, ax in enumerate(axes.flatten()):
    #     ax.imshow(V[:,i].reshape(28,28), cmap='gray')
    #     ax.set_title('PCA Mode {}'.format(i))
    # plt.tight_layout()
    # plt.savefig('figures/A6d_modes.pdf')
    # plt.show()

    # #Part e

    # #2
    # def display_digit_reconstruction(digit_num):
    #     digit = X_train[y_train == digit_num][0]
    #     num_row = 1
    #     num_col = 5
    #     k_range = [5, 15, 40, 100]
    #     names = ['original'] + ['k='+str(k) for k in k_range]
    #     reconstructions = [digit] + [(digit - mu).dot(V[:,:k]).dot(V[:,:k].T) + mu for k in k_range]
    #     fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    #     for i, ax in enumerate(axes.flatten()):
    #         ax.imshow(reconstructions[i].reshape(28,28), cmap='gray')
    #         ax.set_title(names[i])
    #     plt.tight_layout()
    #     plt.savefig('figures/A6e_{}.pdf'.format(digit_num))
    #     plt.show()
    # display_digit_reconstruction(2)
    # display_digit_reconstruction(6)
    # display_digit_reconstruction(7)