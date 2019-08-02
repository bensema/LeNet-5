package main

import (
	"LeNet-5/lenet5"
	"LeNet-5/mnist"
	"flag"
	"fmt"
	"os"
	"time"
)

func main() {
	fmt.Println("train start:", time.Now())
	sourceLabelFile := flag.String("train_labels", "", "source label file")
	sourceImageFile := flag.String("train_images", "", "source image file")
	testLabelFile := flag.String("test_labels", "", "test label file")
	testImageFile := flag.String("test_images", "", "test image file")
	model := flag.String("model", "train", "model")

	flag.Parse()

	if *sourceLabelFile == "" || *sourceImageFile == "" {
		flag.Usage()
		os.Exit(-2)
	}
	fmt.Println("Loading training data...")
	labelData := mnist.ReadMNISTLabels(mnist.OpenFile(*sourceLabelFile))
	imageData, width, height := mnist.ReadMNISTImages(mnist.OpenFile(*sourceImageFile))

	fmt.Println(len(imageData), len(imageData[0]), width, height)
	fmt.Println(len(labelData), labelData[0:10])

	inputs := mnist.PrepareImages(imageData)
	targets := mnist.PrepareLabels(labelData)

	testLabelData := mnist.ReadMNISTLabels(mnist.OpenFile(*testLabelFile))
	testImageData, width, height := mnist.ReadMNISTImages(mnist.OpenFile(*testImageFile))
	testInputs := mnist.PrepareImages(testImageData)
	testTargets := mnist.PrepareLabels(testLabelData)

	lenet5.Run(inputs, targets, testInputs, testTargets, *model)

}
