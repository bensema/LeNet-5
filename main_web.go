package main

import (
	"LeNet-5/lenet5"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"strconv"
	"strings"
)

func main() {
	fmt.Println("ee")
	lenet5.InitLayer()
	lenet5.LoadLeNet()
	fmt.Println("ee3")

	indexHandler := http.FileServer(http.Dir("./web/public"))
	http.ListenAndServe(":12304", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
				lenet5.ForwardPropagation(inputs)
				p := lenet5.FindIndex(lenet5.OutputLayer)
				rs := strconv.Itoa(p)
				w.Write([]byte(rs))
				r.Body.Close()
			}
		}
	}))
}
