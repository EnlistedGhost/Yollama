//go:build !windows && !darwin

package cmd

import (
	"context"
	"errors"

	"github.com/EnlistedGhost/Yollama/api"
)

func startApp(ctx context.Context, client *api.Client) error {
	return errors.New("[YOLLAMA] - Could not connect to a yollama server. Run 'yollama serve' to start a local one.")
}
