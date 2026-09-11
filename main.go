package main

import (
	"context"

	"github.com/spf13/cobra"

	"github.com/EnlistedGhost/Yollama/cmd"
)

func main() {
	cobra.CheckErr(cmd.NewCLI().ExecuteContext(context.Background()))
}
