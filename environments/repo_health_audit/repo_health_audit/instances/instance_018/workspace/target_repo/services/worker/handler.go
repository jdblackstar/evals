package main

import "fmt"

type Handler struct{}

func NewHandler() *Handler {
	return &Handler{}
}

func (h *Handler) Run() {
	fmt.Println("processing tasks")
}
