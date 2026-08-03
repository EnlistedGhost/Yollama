// Package errtypes contains custom error types
package errtypes

import (
	"fmt"
	"strings"
)

const (
	UnknownYollamaKeyErrMsg = "unknown yollama key"
	InvalidModelNameErrMsg = "invalid model name"
)

// TODO: This should have a structured response from the API
type UnknownYollamaKey struct {
	Key string
}

func (e *UnknownYollamaKey) Error() string {
	return fmt.Sprintf("unauthorized: %s %q", UnknownYollamaKeyErrMsg, strings.TrimSpace(e.Key))
}
