package mnist

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

const (
	Width  = 32 // 输入层图片宽
	Height = 32 // 输入层图片高

	ScaleMax float64 = 1.0  // 像素转化最大
	ScaleMin float64 = -1.0 // 像素转化最小

	LabelYT  float64 = 0.8  // 标记选中时
	LabelYNT float64 = -0.8 // 标记未选中时

	Padding = 2 // 填充格数
)

func ReadMNISTLabels(r io.Reader) (labels []byte) {
	header := [2]int32{}
	binary.Read(r, binary.BigEndian, &header)
	labels = make([]byte, header[1])
	r.Read(labels)
	return
}

func ReadMNISTImages(r io.Reader) (images [][]byte, width, height int) {
	header := [4]int32{}
	binary.Read(r, binary.BigEndian, &header)
	images = make([][]byte, header[1])
	width, height = int(header[2]), int(header[3])
	for i := 0; i < len(images); i++ {
		images[i] = make([]byte, width*height)
		r.Read(images[i])
	}
	return
}

func OpenFile(path string) *os.File {
	file, err := os.Open(path)
	if err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}
	return file
}

func PrepareImages(images [][]byte) [][]float64 {
	imagesCount := len(images)
	imageSize := Width * Height
	newImages := make([][]float64, imagesCount)

	for i := 0; i < imagesCount; i++ {
		img := make([]float64, imageSize)
		for j := 0; j < imageSize; j++ {
			img[j] = ScaleMin
		}

		for x := 0; x < 28; x++ {
			for y := 0; y < 28; y++ {
				img[(x+Padding)*Width+y+Padding] = (float64(images[i][x*28+y])/255.0)*(ScaleMax-ScaleMin) + ScaleMin
			}
		}

		newImages[i] = img
	}
	return newImages
}

func PrepareLabels(labels []byte) [][]float64 {

	newLabels := make([][]float64, len(labels))
	for i := 0; i < len(newLabels); i++ {
		tmp := make([]float64, 10)
		for j := 0; j < 10; j++ {
			tmp[j] = LabelYNT
		}
		tmp[labels[i]] = LabelYT
		newLabels[i] = tmp
	}

	return newLabels
}
