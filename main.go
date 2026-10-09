package main

import (
	"context"

	"github.com/EnlistedGhost/Yollama/cobra"
	"github.com/EnlistedGhost/Yollama/cmd"
)

func main() {
	cobra.CheckErr(cmd.NewCLI().ExecuteContext(context.Background()))
}
