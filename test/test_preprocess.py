import unittest
import soundfile
import numpy as np
import matplotlib.pyplot as plt
from main.preprocessUnit import PreprocessUnit

path = 'test/wylacz.wav'
data, fs = soundfile.read(path)
preprocess = PreprocessUnit()


def plotNormalization(inputData, normalizeData):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))

    axs[0].plot(inputData, 'tab:blue')
    axs[0].set_title('Original input data')

    axs[1].plot(normalizeData, 'tab:green')
    axs[1].set_title('Normalize input data')

    plt.tight_layout()
    plt.savefig('test/acquiredData/normalization.png', format='png')
    plt.close()


def plotDownsampling(inputData, downsampleData):
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
    plt.savefig('test/acquiredData/downsampling.png', format='png')
    plt.close()


def test_normalization():
    normalizeData = preprocess.normalize(data)
    plotNormalization(data, normalizeData)


def test_downsampling():
    downsampledData = preprocess.downsample(data)
    plotDownsampling(data, downsampledData)
