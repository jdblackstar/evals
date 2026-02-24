package main

import "fmt"

func main() {
	fmt.Println("worker starting")
	h := NewHandler()
	h.Run()
}
