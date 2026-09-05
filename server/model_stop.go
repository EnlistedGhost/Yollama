package server

import (
	"net/http"
	"time"
	"log/slog"

	"github.com/gin-gonic/gin"
)

type StopRequest struct {
	Model string `json:"model"`
	sched         *Scheduler
}

func (s *Server) StopModelHandler(c *gin.Context) {
	var req StopRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}

	model, err := CheckForModel(req.Model)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Model not found in system manifest"})
		return
	}

	// Convert user string to scheduler model key format
	modelKey := schedulerModelKey(model)

	// Thread-safely
	s.sched.loadedMu.Lock()
	runner, ok := s.sched.loaded[modelKey]
	s.sched.loadedMu.Unlock()

	if !ok || runner == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Model is not currently loaded in VRAM"})
		return
	}

	// Lock runner references, modify lifespan natively
	runner.refMu.Lock()
	
	// Force expiration metadata now
	runner.expiresAt = time.Now()
	
	if runner.expireTimer != nil {
		runner.expireTimer.Stop()
		runner.expireTimer = nil
	}
	runner.sessionDuration = 0

	// Forcefully close llama-server connection
	if runner.llama != nil {
		runner.llama.Close() 
	}

	// Toss into expiration queue
	s.sched.expiredCh <- runner
	runner.refMu.Unlock()

	// Return informative status
	slog.Info("Started VRAM eviction for model", "model", req.Model)
	c.JSON(http.StatusOK, gin.H{
		"status": "success",
		"message": "Model stopped and evicted from VRAM: " + req.Model,
	})
}

// Flush all active runners out of VRAM instantly
func (s *Server) StopAllModelsHandler(c *gin.Context) {
	var req StopRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}
	
	slog.Info("Global VRAM purge requested")


	// Trigger native routine
	s.sched.unloadAllRunners()

	// Don't call srvr.Close() or done() here, 
	// keep  main Go API server alive
	c.JSON(http.StatusOK, gin.H{
		"status": "success",
		"message": "All model runners terminated. VRAM flushed.",
	})
}