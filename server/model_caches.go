package server

import "context"

type modelCaches struct {
	show            *modelShowCache
}

func newModelCaches() *modelCaches {
	return &modelCaches{
		show:            newModelShowCache(),
	}
}

func (c *modelCaches) Start(ctx context.Context) {
	if c == nil {
		return
	}
	if c.show != nil {
		c.show.Start(ctx)
	}
}
