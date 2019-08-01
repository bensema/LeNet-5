package lenet5

import (
	"crypto/rand"
	"math"
	"math/big"
)

func Rand0To1() float64 {
	result, _ := rand.Int(rand.Reader, big.NewInt(4294967295))
	return float64(result.Int64()) / 4294967295.0
}

func RandInt64(n int64) int64 {
	result, _ := rand.Int(rand.Reader, big.NewInt(n))
	return result.Int64()
}

func Tanh(val float64) float64 {
	ep := math.Exp(val)
	em := math.Exp(-val)

	return (ep - em) / (ep + em)
}

func DTanh(val float64) float64 {
	return 1.0 - val*val
}

//y为计算出的值，t为已知值
func Mse(y float64, t float64) float64 {
	return (y - t) * (y - t) / 2
}

//y为计算出的值，t为已知值
func DMse(y float64, t float64) float64 {
	return y - t
}

// 梯度
func GradientDescent(W float64, dW float64, alpha float64, lambda float64) float64 {
	return W - alpha*(dW+lambda*W)
}
