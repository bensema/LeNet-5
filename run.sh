#!/usr/bin/env bash


go run main.go -train_images=./data/train-images-idx3-ubyte -train_labels=./data/train-labels-idx1-ubyte  -test_images=./data/t10k-images-idx3-ubyte  -test_labels=./data/t10k-labels-idx1-ubyte -model=train