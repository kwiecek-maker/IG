from main.preprocessUnit import PreprocessUnit
import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np


# Compare the plots of original data and normalize data
def test_normalization(inputData, normalizeData):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))

    axs[0].plot(inputData, 'tab:blue')
    axs[0].set_title('Original input data')

    axs[1].plot(normalizeData, 'tab:green')
    axs[1].set_title('Normalize input data')

    plt.tight_layout()
    plt.show()


# Compare the plots of original data and data after downsampling process
def test_downsampling(inputData, downsampleData):
    downsampleDataWithZeros = np.zeros(len(inputData))

    idx = 0
    for i, value in enumerate(downsampleData):
        if idx < len(inputData):
            downsampleDataWithZeros[idx] = value
            idx += 6

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    axs[0].plot(inputData, 'tab:blue')
    axs[0].set_title('Original input data')

    axs[1].plot(inputData)
    axs[1].plot(downsampleDataWithZeros)
    axs[1].set_title('Original and downsample input data')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    path = ''
    fs, data = wavfile.read(path)

    p = PreprocessUnit()

    normalizeData = p.normalize(data)
    test_normalization(data, normalizeData)

    downsampleData = p.downsample(data)
    test_downsampling(data, downsampleData)
