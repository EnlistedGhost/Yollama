package gobetween

import (
"log/slog"

"github.com/EnlistedGhost/Yollama/llm"
"github.com/EnlistedGhost/Yollama/server"
)


type StopRequest struct {
	sched         *Scheduler
}

func SetGlobalBatches(n_NumBatch) {
	llmBatch := n_NumBatch
	sched.GlobalCurBatchNum = llmBatch

	slog.Info("[Yollama] | Set Scheduler NumBatch Integer Variable")
}