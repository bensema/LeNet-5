package lenet5

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
)

const (
	LeNetFile = "lenet5.json"
)

type FLayer struct {
	Kernels [][]float64 `json:"kernels"`
	Bias    []float64   `json:"bias"`
}

type FLeNet struct {
	C1  *FLayer `json:"c_1"`
	S2  *FLayer `json:"s_2"`
	C3  *FLayer `json:"c_3"`
	S4  *FLayer `json:"s_4"`
	C5  *FLayer `json:"c_5"`
	Out *FLayer `json:"out"`
}

func SaveLeNet5() {
	fp, err := os.Create(LeNetFile)
	if err != nil {
		fmt.Println(err)
	}
	defer fp.Close()

	fl := &FLeNet{}
	fl.C1 = &FLayer{}
	fl.C1.Kernels = make([][]float64, C1Layer.KernelCount)
	fl.C1.Bias = make([]float64, C1Layer.MapCount)
	for i := 0; i < C1Layer.KernelCount; i++ {
		fl.C1.Kernels[i] = make([]float64, C1Layer.KernelW*C1Layer.KernelH)
		fl.C1.Kernels[i] = C1Layer.Kernels[i].Weight
	}
	for i := 0; i < C1Layer.MapCount; i++ {
		fl.C1.Bias[i] = C1Layer.FeatureMaps[i].Bias
	}

	fl.S2 = &FLayer{}
	fl.S2.Kernels = make([][]float64, S2Layer.KernelCount)
	fl.S2.Bias = make([]float64, S2Layer.MapCount)
	for i := 0; i < S2Layer.KernelCount; i++ {
		fl.S2.Kernels[i] = make([]float64, S2Layer.KernelW*S2Layer.KernelH)
		fl.S2.Kernels[i] = S2Layer.Kernels[i].Weight
	}
	for i := 0; i < S2Layer.MapCount; i++ {
		fl.S2.Bias[i] = S2Layer.FeatureMaps[i].Bias
	}

	fl.C3 = &FLayer{}
	fl.C3.Kernels = make([][]float64, C3Layer.KernelCount)
	fl.C3.Bias = make([]float64, C3Layer.MapCount)
	for i := 0; i < C3Layer.KernelCount; i++ {
		fl.C3.Kernels[i] = make([]float64, C3Layer.KernelW*C3Layer.KernelH)
		fl.C3.Kernels[i] = C3Layer.Kernels[i].Weight
	}
	for i := 0; i < C3Layer.MapCount; i++ {
		fl.C3.Bias[i] = C3Layer.FeatureMaps[i].Bias
	}

	fl.S4 = &FLayer{}
	fl.S4.Kernels = make([][]float64, S4Layer.KernelCount)
	fl.S4.Bias = make([]float64, S4Layer.MapCount)
	for i := 0; i < S4Layer.KernelCount; i++ {
		fl.S4.Kernels[i] = make([]float64, S4Layer.KernelW*S4Layer.KernelH)
		fl.S4.Kernels[i] = S4Layer.Kernels[i].Weight
	}
	for i := 0; i < S4Layer.MapCount; i++ {
		fl.S4.Bias[i] = S4Layer.FeatureMaps[i].Bias
	}

	fl.C5 = &FLayer{}
	fl.C5.Kernels = make([][]float64, C5Layer.KernelCount)
	fl.C5.Bias = make([]float64, C5Layer.MapCount)
	for i := 0; i < C5Layer.KernelCount; i++ {
		fl.C5.Kernels[i] = make([]float64, C5Layer.KernelW*C5Layer.KernelH)
		fl.C5.Kernels[i] = C5Layer.Kernels[i].Weight
	}
	for i := 0; i < C5Layer.MapCount; i++ {
		fl.C5.Bias[i] = C5Layer.FeatureMaps[i].Bias
	}

	fl.Out = &FLayer{}
	fl.Out.Kernels = make([][]float64, OutputLayer.KernelCount)
	fl.Out.Bias = make([]float64, OutputLayer.MapCount)
	for i := 0; i < OutputLayer.KernelCount; i++ {
		fl.Out.Kernels[i] = make([]float64, OutputLayer.KernelW*OutputLayer.KernelH)
		fl.Out.Kernels[i] = OutputLayer.Kernels[i].Weight
	}
	for i := 0; i < OutputLayer.MapCount; i++ {
		fl.Out.Bias[i] = OutputLayer.FeatureMaps[i].Bias
	}

	data, err := json.Marshal(fl)
	if err != nil {
		fmt.Println(err)
	}
	_, err = fp.Write(data)
	if err != nil {
		fmt.Println(err)
	}
	fp.Sync()

}

func LoadLeNet() {
	data, err := ioutil.ReadFile(LeNetFile)
	if err != nil {
		fmt.Print(err)
		fmt.Println("暂未找到文件")
		return
	}
	fl := &FLeNet{}
	err = json.Unmarshal(data, fl)
	if err != nil {
		fmt.Println(err)
		return
	}

	for i := 0; i < C1Layer.KernelCount; i++ {
		C1Layer.Kernels[i].Weight = fl.C1.Kernels[i]
	}
	for i := 0; i < C1Layer.MapCount; i++ {
		C1Layer.FeatureMaps[i].Bias = fl.C1.Bias[i]
	}

	for i := 0; i < S2Layer.KernelCount; i++ {
		S2Layer.Kernels[i].Weight = fl.S2.Kernels[i]
	}
	for i := 0; i < S2Layer.MapCount; i++ {
		S2Layer.FeatureMaps[i].Bias = fl.S2.Bias[i]
	}

	for i := 0; i < C3Layer.KernelCount; i++ {
		C3Layer.Kernels[i].Weight = fl.C3.Kernels[i]
	}
	for i := 0; i < C3Layer.MapCount; i++ {
		C3Layer.FeatureMaps[i].Bias = fl.C3.Bias[i]
	}

	for i := 0; i < S4Layer.KernelCount; i++ {
		S4Layer.Kernels[i].Weight = fl.S4.Kernels[i]
	}
	for i := 0; i < S4Layer.MapCount; i++ {
		S4Layer.FeatureMaps[i].Bias = fl.S4.Bias[i]
	}

	for i := 0; i < C5Layer.KernelCount; i++ {
		C5Layer.Kernels[i].Weight = fl.C5.Kernels[i]
	}
	for i := 0; i < C5Layer.MapCount; i++ {
		C5Layer.FeatureMaps[i].Bias = fl.C5.Bias[i]
	}

	for i := 0; i < OutputLayer.KernelCount; i++ {
		OutputLayer.Kernels[i].Weight = fl.Out.Kernels[i]
	}
	for i := 0; i < OutputLayer.MapCount; i++ {
		OutputLayer.FeatureMaps[i].Bias = fl.Out.Bias[i]
	}

}
