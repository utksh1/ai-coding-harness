package main

import "testing"

func TestAPIURLForStripsOnlyThePath(t *testing.T) {
	cases := map[string]string{
		"ws://localhost:8080/ws":         "http://localhost:8080",
		"wss://gw.example.com/ws":        "https://gw.example.com",
		"http://localhost:8080":          "http://localhost:8080",
		"https://gw.example.com":         "https://gw.example.com",
		"localhost:8080":                 "http://localhost:8080",
		"ws://host:8080/path/with/slash": "http://host:8080",
	}
	for in, want := range cases {
		if got := apiURLFor(in); got != want {
			t.Errorf("apiURLFor(%q) = %q, want %q", in, got, want)
		}
	}
}
