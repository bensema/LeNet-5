package main

import (
	"LeNet-5/lenet5"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io/ioutil"
	"log"
	"net/http"
	"strconv"
	"strings"
)

func init() {
	lenet5.InitLayer()
	lenet5.LoadLeNet()
}

func main() {

	indexHandler := http.FileServer(http.Dir("./web/public"))
	http.ListenAndServe(":9900", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/":
			indexHandler.ServeHTTP(w, r)
		case "/classifier":
			if body, err := ioutil.ReadAll(r.Body); err != nil {
				log.Fatal(err)
			} else {
				d := strings.Split(string(body), ",")
				var inputs []float64
				for _, s := range d {
					f, _ := strconv.ParseFloat(s, 32)
					inputs = append(inputs, f)
				}

				s := predict(inputs)
				w.Write(s)
				r.Body.Close()
			}
		}
	}))
}

type Out struct {
	Num int
	C1  []string
	S2  []string
	C3  []string
	S4  []string
	C5  []string
	Out []string
}

func predict(inputs []float64) []byte {

	out := Out{}
	lenet5.ForwardPropagation(inputs)
	out.Num = lenet5.FindIndex(lenet5.OutputLayer)

	out.C1 = make([]string, lenet5.C1Layer.MapCount)
	out.S2 = make([]string, lenet5.S2Layer.MapCount)
	out.C3 = make([]string, lenet5.C3Layer.MapCount)
	out.S4 = make([]string, lenet5.S4Layer.MapCount)
	out.C5 = make([]string, lenet5.C5Layer.MapCount)
	out.Out = make([]string, lenet5.OutputLayer.MapCount)

	for i := 0; i < len(out.C1); i++ {
		out.C1[i] = imageStr(28, 28, lenet5.C1Layer.FeatureMaps[i].Data)
	}
	for i := 0; i < len(out.S2); i++ {
		out.S2[i] = imageStr(14, 14, lenet5.S2Layer.FeatureMaps[i].Data)
	}
	for i := 0; i < len(out.C3); i++ {
		out.C3[i] = imageStr(10, 10, lenet5.C3Layer.FeatureMaps[i].Data)
	}
	for i := 0; i < len(out.S4); i++ {
		out.S4[i] = imageStr(5, 5, lenet5.S4Layer.FeatureMaps[i].Data)
	}
	for i := 0; i < len(out.C5); i++ {
		out.C5[i] = imageStr(1, 1, lenet5.C5Layer.FeatureMaps[i].Data)
	}
	for i := 0; i < len(out.Out); i++ {
		out.Out[i] = imageStr(1, 1, lenet5.OutputLayer.FeatureMaps[i].Data)
	}

	jsons, _ := json.Marshal(out) //转换成JSON返回的是byte[]

	return jsons
}

func imageStr(dx int, dy int, data []float64) (out string) {
	var b bytes.Buffer

	rect := image.Rect(0, 0, dx, dy)
	gray := image.NewGray(rect)
	for x := 0; x < dx; x++ {
		for y := 0; y < dy; y++ {

			d := (data[x*dy+y] + 1) / 2 * 255
			gray.Set(y, x, color.Gray{uint8(d)})
		}
	}
	png.Encode(&b, gray)
	out = fmt.Sprintf("data:image/jpeg;base64,%s", base64.StdEncoding.EncodeToString(b.Bytes()))
	return
}
