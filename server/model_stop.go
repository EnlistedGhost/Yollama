package server

import (
	"net/http"
	"time"
	"log/slog"

	"github.com/gin-gonic/gin"
)

type StopRequest struct {
	Model string `json:"model"`
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

	modelKey := schedulerModelKey(model)

	// Lock the scheduler map and isolate the model
	s.sched.loadedMu.Lock()
	runner, ok := s.sched.loaded[modelKey]
	if ok && runner != nil {
		// Remove the loaded map to snub new runners
		delete(s.sched.loaded, modelKey)
	}
	s.sched.loadedMu.Unlock()

	if !ok || runner == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Model is not currently loaded in VRAM"})
		return
	}

	// Lock runner metadata and force instant expiration
	runner.refMu.Lock()
	runner.expiresAt = time.Now()
	
	if runner.expireTimer != nil {
		runner.expireTimer.Stop()
		runner.expireTimer = nil
	}
	runner.sessionDuration = 0

	// Graceful check: Only kill or push to expire queue if no active clients are using it
	// If refCount > 0, the scheduler's completion handler loop (processCompleted) 
	// will automatically catch it and evict it when inference finishes.
	if runner.refCount <= 0 {
		if runner.llama != nil {
			runner.llama.Close()
		}
		select {
		case s.sched.expiredCh <- runner:
		default:
			slog.Warn("Expired channel full, dropping eviction token safely", "model", req.Model)
		}
	}
	runner.refMu.Unlock()

	slog.Info("Successfully initiated VRAM eviction for model", "model", req.Model)
	c.JSON(http.StatusOK, gin.H{
		"status": "success",
		"message": "Model stopped and scheduled for eviction from VRAM: " + req.Model,
	})
}

// Flush all active runners out of VRAM
func (s *Server) StopAllModelsHandler(c *gin.Context) {
	// Remove rigid body-binding checks so empty POST requests execute smoothly
	slog.Info("Global VRAM purge requested")

	// Trigger native cleanup loop
	s.sched.unloadAllRunners()

	c.JSON(http.StatusOK, gin.H{
		"status": "success",
		"message": "All model runners terminated. VRAM flushed successfully.",
	})
}
