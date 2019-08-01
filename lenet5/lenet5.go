package lenet5

import (
	"fmt"
	"math"
	"time"
)

const (
	Scale float64 = 6.0

	BatchSize = 10 // 批量训练数

	TrainCount = 60000
	TestCount  = 10000
)

var (
	O               = true
	X               = false
	connectionTable = []bool{
		O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
		O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
		O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
		X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
		X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
		X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O,
	}
	InputLayer  *Layer
	C1Layer     *Layer
	S2Layer     *Layer
	C3Layer     *Layer
	S4Layer     *Layer
	C5Layer     *Layer
	F6Layer     *Layer
	OutputLayer *Layer
)

type Kernel struct {
	Weight  []float64
	DWeight []float64
}

type FeatureMap struct {
	Data []float64 // 数据
	Err  []float64

	Bias  float64
	DBias float64
}

type Layer struct {
	MapCount    int
	MapW        int
	MapH        int
	FeatureMaps []FeatureMap

	KernelCount int
	KernelW     int
	KernelH     int
	Kernels     []Kernel

	Name   string
	TmpMap []float64
}

func (layer *Layer) Init(PrevLayerMapCount int, mapCount int, kernelW int, kernelH int, mapW int, mapH int, isPooling bool) {
	var (
		fanIn, fanOut int
		denominator   int
		weightBase    float64
		kSize         int
		mSize         int
	)
	if isPooling == true {
		fanIn, fanOut = 4, 1
	} else {
		fanIn, fanOut = PrevLayerMapCount*kernelW*kernelH, mapCount*kernelW*kernelH
	}
	denominator = fanIn + fanOut
	if denominator != 0 {
		weightBase = math.Sqrt(Scale / float64(denominator))
	} else {
		weightBase = 0.5
	}
	weightBase *= 1

	layer.KernelCount = PrevLayerMapCount * mapCount
	layer.KernelW = kernelW
	layer.KernelH = kernelH
	layer.Kernels = make([]Kernel, layer.KernelCount)

	kSize = layer.KernelW * layer.KernelH
	for i := 0; i < PrevLayerMapCount; i++ {
		for j := 0; j < mapCount; j++ {
			layer.Kernels[i*mapCount+j].Weight = make([]float64, kSize)
			initKernel(layer.Kernels[i*mapCount+j].Weight, kSize, weightBase)
			layer.Kernels[i*mapCount+j].DWeight = make([]float64, kSize)
		}
	}

	layer.MapCount = mapCount
	layer.MapW = mapW
	layer.MapH = mapH
	mSize = layer.MapW * layer.MapH
	layer.FeatureMaps = make([]FeatureMap, layer.MapCount)
	for i := 0; i < layer.MapCount; i++ {
		layer.FeatureMaps[i].Data = make([]float64, mSize)
		layer.FeatureMaps[i].Err = make([]float64, mSize)
	}

	layer.TmpMap = make([]float64, mSize)
}

// 重置参数
func (layer *Layer) ResetParams() {
	kSize := layer.KernelW * layer.KernelH
	for i := 0; i < layer.KernelCount; i++ {
		for j := 0; j < kSize; j++ {
			layer.Kernels[i].DWeight[j] = 0.0
		}
	}

	for i := 0; i < layer.MapCount; i++ {
		layer.FeatureMaps[i].DBias = 0.0
	}
}

func (layer *Layer) UpdateParams(learningRate float64) {
	lambda := 0.0
	kSize := layer.KernelW * layer.KernelH
	// w
	for i := 0; i < layer.KernelCount; i++ {
		for k := 0; k < kSize; k++ {
			layer.Kernels[i].Weight[k] = GradientDescent(layer.Kernels[i].Weight[k], layer.Kernels[i].DWeight[k]/BatchSize, learningRate, lambda)
		}
	}

	//b
	for i := 0; i < layer.MapCount; i++ {
		layer.FeatureMaps[i].Bias = GradientDescent(layer.FeatureMaps[i].Bias, layer.FeatureMaps[i].DBias/BatchSize, learningRate, lambda)
	}

}

func (layer *Layer) ResetTmpMap() {
	mSize := layer.MapH * layer.MapW
	for i := 0; i < mSize; i++ {
		layer.TmpMap[i] = 0.0
	}
}

func initKernel(weight []float64, size int, weightBase float64) {
	for i := 0; i < size; i++ {
		weight[i] = (Rand0To1() - 0.5) * 2 * weightBase
	}
}

// 卷积
func ConvolutionValid(inData []float64, inW int, inH int, kernel []float64, kW int, kH int, outData []float64, outW int, outH int) {
	sum := 0.0
	for i := 0; i < outH; i++ {
		for j := 0; j < outW; j++ {
			sum = 0.0
			for n := 0; n < kH; n++ {
				for m := 0; m < kW; m++ {
					sum += inData[(i+n)*inW+j+m] * kernel[n*kW+m]
				}
			}
			outData[i*outW+j] += sum
		}
	}
	return
}

func FullyConnectedPropagation(prev_layer *Layer, layer *Layer) {

	for i := 0; i < layer.MapCount; i++ {
		sum := 0.0
		for j := 0; j < prev_layer.MapCount; j++ {
			sum += prev_layer.FeatureMaps[j].Data[0] * layer.Kernels[j*layer.MapCount+i].Weight[0]
		}
		sum += layer.FeatureMaps[i].Bias
		layer.FeatureMaps[i].Data[0] = Tanh(sum)
	}
}

// 向后
func FullyConnectedBackPropagation(layer *Layer, prev_layer *Layer) {
	// delta
	for i := 0; i < prev_layer.MapCount; i++ {
		prev_layer.FeatureMaps[i].Err[0] = 0.0
		for j := 0; j < layer.MapCount; j++ {
			prev_layer.FeatureMaps[i].Err[0] += layer.FeatureMaps[j].Err[0] * layer.Kernels[i*layer.MapCount+j].Weight[0]
		}
		prev_layer.FeatureMaps[i].Err[0] *= DTanh(prev_layer.FeatureMaps[i].Data[0])
	}
	// dW
	for i := 0; i < prev_layer.MapCount; i++ {
		for j := 0; j < layer.MapCount; j++ {
			layer.Kernels[i*layer.MapCount+j].DWeight[0] += layer.FeatureMaps[j].Err[0] * prev_layer.FeatureMaps[i].Data[0]
		}
	}

	// db
	for i := 0; i < layer.MapCount; i++ {
		layer.FeatureMaps[i].DBias += layer.FeatureMaps[i].Err[0]
	}

}

// 向前
func MaxPoolingForwardPropagation(prev_layer *Layer, layer *Layer) {

	mapW := layer.MapW
	mapH := layer.MapH
	upMapW := prev_layer.MapW

	for k := 0; k < layer.MapCount; k++ {
		for i := 0; i < mapH; i++ {
			for j := 0; j < mapW; j++ {
				maxVal := prev_layer.FeatureMaps[k].Data[2*i*upMapW+2*j]
				for n := 2 * i; n < 2*(i+1); n++ {
					for m := 2 * j; m < 2*(j+1); m++ {
						if prev_layer.FeatureMaps[k].Data[n*upMapW+m] > maxVal {
							maxVal = prev_layer.FeatureMaps[k].Data[n*upMapW+m]
						}
					}
				}
				layer.FeatureMaps[k].Data[i*mapW+j] = Tanh(maxVal)
			}
		}
	}
}

// 向后
func MaxPoolingBackPropagation(layer *Layer, prev_layer *Layer) {

	mapW := layer.MapW
	mapH := layer.MapH
	upMapW := prev_layer.MapW

	for k := 0; k < layer.MapCount; k++ {
		// delta
		for i := 0; i < mapH; i++ {
			for j := 0; j < mapW; j++ {
				row, col := 2*i, 2*j
				maxVal := prev_layer.FeatureMaps[k].Data[row*upMapW+col]

				for n := 2 * i; n < 2*(i+1); n++ {
					for m := 2 * j; m < 2*(j+1); m++ {
						if prev_layer.FeatureMaps[k].Data[n*upMapW+m] > maxVal {
							row = n
							col = m
							maxVal = prev_layer.FeatureMaps[k].Data[n*upMapW+m]
						} else {
							prev_layer.FeatureMaps[k].Err[n*upMapW+m] = 0.0
						}
					}
				}

				prev_layer.FeatureMaps[k].Err[row*upMapW+col] = layer.FeatureMaps[k].Err[i*mapW+j] * DTanh(maxVal)
			}
		}

	}
}

// 向前
func ConvolutionForward(prev_layer *Layer, layer *Layer, f bool, pconnection []bool) {

	index := 0
	mSize := layer.MapW * layer.MapH
	for i := 0; i < layer.MapCount; i++ {
		layer.ResetTmpMap()
		for j := 0; j < prev_layer.MapCount; j++ {
			index = j*layer.MapCount + i
			if f == true {
				if pconnection[index] == false {
					continue
				}
			}

			ConvolutionValid(
				prev_layer.FeatureMaps[j].Data, prev_layer.MapW, prev_layer.MapH,
				layer.Kernels[index].Weight, layer.KernelW, layer.KernelH,
				layer.TmpMap, layer.MapW, layer.MapH)
		}
		for k := 0; k < mSize; k++ {
			layer.FeatureMaps[i].Data[k] = Tanh(layer.TmpMap[k] + layer.FeatureMaps[i].Bias)
		}
	}

}

// 向后传播
func ConvolutionBackPropagation(layer *Layer, prev_layer *Layer, f bool, pconnection []bool) {

	index := 0
	mSize := prev_layer.MapW * prev_layer.MapH
	// delta
	for i := 0; i < prev_layer.MapCount; i++ {
		prev_layer.ResetTmpMap()
		for j := 0; j < layer.MapCount; j++ {
			index = i*layer.MapCount + j
			if f == true {
				if pconnection[index] == false {
					continue
				}
			}

			for n := 0; n < layer.MapH; n++ {
				for m := 0; m < layer.MapW; m++ {
					er := layer.FeatureMaps[j].Err[n*layer.MapW+m]
					for ky := 0; ky < layer.KernelH; ky++ {
						for kx := 0; kx < layer.KernelW; kx++ {
							prev_layer.TmpMap[(n+ky)*prev_layer.MapW+m+kx] += er * layer.Kernels[index].Weight[ky*layer.KernelW+kx]
						}
					}
				}
			}
		}
		for k := 0; k < mSize; k++ {
			prev_layer.FeatureMaps[i].Err[k] = prev_layer.TmpMap[k] * DTanh(prev_layer.FeatureMaps[i].Data[k])
		}
	}

	// dW
	for i := 0; i < prev_layer.MapCount; i++ {
		for j := 0; j < layer.MapCount; j++ {
			index = i*layer.MapCount + j
			if f == true {
				if pconnection[index] == false {
					continue
				}
			}

			ConvolutionValid(
				prev_layer.FeatureMaps[i].Data, prev_layer.MapW, prev_layer.MapH,
				layer.FeatureMaps[j].Err, layer.MapW, layer.MapH,
				layer.Kernels[index].DWeight, layer.KernelW, layer.KernelH)
		}
	}

	// db
	mSize = layer.MapW * layer.MapH
	for i := 0; i < layer.MapCount; i++ {
		sum := 0.0
		for k := 0; k < mSize; k++ {
			sum += layer.FeatureMaps[i].Err[k]
		}
		layer.FeatureMaps[i].DBias += sum
	}

}

func FindIndex(layer *Layer) (index int) {
	index = 0
	maxVal := layer.FeatureMaps[0].Data[0]
	for i := 0; i < layer.MapCount; i++ {
		if layer.FeatureMaps[i].Data[0] > maxVal {
			maxVal = layer.FeatureMaps[i].Data[0]
			index = i
		}
	}

	return
}

func FindIndexLabel(label []float64) (index int) {
	index = 0
	maxVal := label[0]
	for i := 0; i < 10; i++ {
		if label[i] > maxVal {
			maxVal = label[i]
			index = i
		}
	}
	return
}

func forwardPropagation(data []float64) {

	InputLayer.FeatureMaps[0].Data = data

	//t1 := time.Now()
	// Input -> C1
	ConvolutionForward(InputLayer, C1Layer, false, []bool{})
	// C1 -> S2
	MaxPoolingForwardPropagation(C1Layer, S2Layer)
	// S2 -> C3
	ConvolutionForward(S2Layer, C3Layer, true, connectionTable)
	// C3 -> S4
	MaxPoolingForwardPropagation(C3Layer, S4Layer)
	// S4 -> C5
	ConvolutionForward(S4Layer, C5Layer, false, []bool{})
	// C5 -> Output
	FullyConnectedPropagation(C5Layer, OutputLayer)
	//fmt.Println("forward elapsed:", time.Since(t1))

}

func backwardPropagation(label []float64) {
	for i := 0; i < OutputLayer.MapCount; i++ {
		OutputLayer.FeatureMaps[i].Err[0] = DMse(OutputLayer.FeatureMaps[i].Data[0], label[i]) * DTanh(OutputLayer.FeatureMaps[i].Data[0])
	}
	// Out -> C5
	FullyConnectedBackPropagation(OutputLayer, C5Layer)
	// C5 -> S4
	ConvolutionBackPropagation(C5Layer, S4Layer, false, []bool{})
	// S4 -> C3
	MaxPoolingBackPropagation(S4Layer, C3Layer)
	// C3 -> S2
	ConvolutionBackPropagation(C3Layer, S2Layer, true, connectionTable)
	// S2 -> C1
	MaxPoolingBackPropagation(S2Layer, C1Layer)
	// C1 -> Input
	ConvolutionBackPropagation(C1Layer, InputLayer, false, []bool{})
}

func resetWeights() {
	C1Layer.ResetParams()
	S2Layer.ResetParams()
	C3Layer.ResetParams()
	S4Layer.ResetParams()
	C5Layer.ResetParams()
	OutputLayer.ResetParams()
}

func updateWeights(learningRate float64) {
	C1Layer.UpdateParams(learningRate)
	S2Layer.UpdateParams(learningRate)
	C3Layer.UpdateParams(learningRate)
	S4Layer.UpdateParams(learningRate)
	C5Layer.UpdateParams(learningRate)
	OutputLayer.UpdateParams(learningRate)
}

func train(inputs [][]float64, labels [][]float64, learningRate float64) {
	var (
		randPerm   []int
		batchCount int
	)
	randPerm = []int{}
	// 随机打乱样本index
	for i := 0; i < TrainCount; i++ {
		randPerm = append(randPerm, i)
	}

	for i := 0; i < TrainCount; i++ {
		j := int(RandInt64(TrainCount*3))%(TrainCount-i) + i
		t := randPerm[j]
		randPerm[j] = randPerm[i]
		randPerm[i] = t
	}

	// 迭代训练
	batchCount = TrainCount / BatchSize

	for i := 0; i < batchCount; i++ {

		resetWeights()
		for j := 0; j < BatchSize; j++ {
			index := i*BatchSize + j
			forwardPropagation(inputs[randPerm[index]])
			backwardPropagation(labels[randPerm[index]])
		}

		// 更新权值
		updateWeights(learningRate)

		if i%1000 == 0 {
			fmt.Println(fmt.Sprintf("progress...%d/%d \n", i, batchCount))
		}
	}
}

func predict(inputs [][]float64, labels [][]float64) {
	numSuccess := 0
	p := 0
	actual := 0
	confusionMatrix := [100]int{}
	for i := 0; i < TestCount; i++ {
		forwardPropagation(inputs[i])

		p = FindIndex(OutputLayer)
		actual = FindIndexLabel(labels[i])
		if p == actual {
			numSuccess++
		}

		confusionMatrix[p*10+actual]++
	}

	fmt.Println(fmt.Sprintf("accuracy: %d/%d", numSuccess, TestCount))
	fmt.Print("\n   *  ")
	for i := 0; i < 10; i++ {
		fmt.Print(fmt.Sprintf("%4d  ", i))
	}
	fmt.Println()
	for i := 0; i < 10; i++ {
		fmt.Print(fmt.Sprintf("%4d  ", i))
		for j := 0; j < 10; j++ {
			fmt.Print(fmt.Sprintf("%4d  ", confusionMatrix[i*10+j]))
		}
		fmt.Println()
	}
	fmt.Println()

}

func Run(inputs [][]float64, labels [][]float64, testInputs [][]float64, testLabels [][]float64) {
	var (
		learningRate     float64
		trainCount       int
		kernelW, kernelH int
	)

	InputLayer = &Layer{}
	C1Layer = &Layer{}
	S2Layer = &Layer{}
	C3Layer = &Layer{}
	S4Layer = &Layer{}
	C5Layer = &Layer{}
	OutputLayer = &Layer{}

	kernelW = 0
	kernelH = 0
	InputLayer.Init(0, 1, kernelW, kernelH, 32, 32, false)
	InputLayer.Name = "InputLayer"

	kernelW = 5
	kernelH = 5
	C1Layer.Init(1, 6, kernelW, kernelH, InputLayer.MapW-kernelW+1, InputLayer.MapH-kernelH+1, false)
	C1Layer.Name = "C1Layer"

	kernelW = 1
	kernelH = 1
	S2Layer.Init(1, 6, kernelW, kernelH, C1Layer.MapW/2, C1Layer.MapH/2, true)
	S2Layer.Name = "S2Layer"

	kernelW = 5
	kernelH = 5
	C3Layer.Init(6, 16, kernelW, kernelH, S2Layer.MapW-kernelW+1, S2Layer.MapH-kernelH+1, false)
	C3Layer.Name = "C3Layer"

	kernelW = 1
	kernelH = 1
	S4Layer.Init(1, 16, kernelW, kernelH, C3Layer.MapW/2, C3Layer.MapH/2, true)
	S4Layer.Name = "S4Layer"

	kernelW = 5
	kernelH = 5
	C5Layer.Init(16, 120, kernelW, kernelH, S4Layer.MapW-kernelW+1, S4Layer.MapH-kernelH+1, false)
	C5Layer.Name = "C5Layer"

	kernelW = 1
	kernelH = 1
	OutputLayer.Init(120, 10, kernelW, kernelH, 1, 1, false)
	OutputLayer.Name = "OutputLayer"

	trainCount = 3
	learningRate = 0.01 * math.Sqrt(float64(BatchSize))
	for i := 0; i < trainCount; i++ {
		fmt.Println("train epoch is: ", i)
		startTime := time.Now()
		train(inputs, labels, learningRate)
		fmt.Println("train time:", time.Since(startTime))

		startTime = time.Now()
		predict(testInputs, testLabels)
		fmt.Println("predict time:", time.Since(startTime))

		learningRate *= 0.85
	}
}
