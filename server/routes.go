package server

import (
	"bytes"
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"io"
	"io/fs"
	"log/slog"
	"math"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"os"
	"os/signal"
	"slices"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"golang.org/x/image/webp"
	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/auth"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	internalcloud "github.com/ollama/ollama/internal/cloud"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
)

func writeModelRefParseError(c *gin.Context, err error, fallbackStatus int, fallbackMessage string) {
	switch {
	case errors.Is(err, errConflictingModelSource):
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
	case errors.Is(err, model.ErrUnqualifiedName):
		c.JSON(http.StatusBadRequest, gin.H{"error": errtypes.InvalidModelNameErrMsg})
	default:
		c.JSON(fallbackStatus, gin.H{"error": fallbackMessage})
	}
}

var mode string = gin.DebugMode

type Server struct {
	addr          net.Addr
	sched         *Scheduler
	defaultNumCtx int
	requestLogger *inferenceRequestLogger
	modelCaches   *modelCaches
}

func init() {
	switch mode {
	case gin.DebugMode:
	case gin.ReleaseMode:
	case gin.TestMode:
	default:
		mode = gin.DebugMode
	}

	gin.SetMode(mode)
}

var (
	errRequired    = errors.New("is required")
	errBadTemplate = errors.New("template error")
)

func (s *Server) modelOptions(model *Model, requestOpts map[string]any) (api.Options, error) {
	return s.modelOptionsWithEmbeddingBatchDefault(model, requestOpts, shouldApplyEmbeddingBatchDefault(model, requestOpts))
}

func (s *Server) modelOptionsWithEmbeddingBatchDefault(model *Model, requestOpts map[string]any, applyEmbeddingBatchDefault bool) (api.Options, error) {
	opts := api.DefaultOptions()
	if opts.NumCtx == 0 {
		opts.NumCtx = s.defaultNumCtx
	}

	// api.Options stores defaulted values
	if model != nil {
		if err := opts.FromMap(model.Options); err != nil {
			return api.Options{}, err
		}
	}

	if err := opts.FromMap(requestOpts); err != nil {
		return api.Options{}, err
	}

	if applyEmbeddingBatchDefault {
		opts = llm.WithDefaultEmbeddingNumBatch(opts)
	}

	return opts, nil
}

func shouldApplyEmbeddingBatchDefault(m *Model, requestOpts map[string]any) bool {
	if m == nil || hasOption(m.Options, "num_batch") || hasOption(requestOpts, "num_batch") {
		return false
	}
	if slices.Contains(m.Config.Capabilities, string(model.CapabilityEmbedding)) {
		return true
	}
	return m.ModelPath != "" && m.CheckCapabilities(model.CapabilityEmbedding) == nil
}

func hasOption(opts map[string]any, name string) bool {
	_, ok := opts[name]
	return ok
}

func usesAutomaticNumCtx(model *Model, requestOpts map[string]any) bool {
	if _, ok := requestOpts["num_ctx"]; ok {
		return false
	}
	if model != nil {
		if _, ok := model.Options["num_ctx"]; ok {
			return false
		}
	}
	return envconfig.ContextLength() == 0
}

func usesAutomaticNumBatch(model *Model, requestOpts map[string]any) bool {
	if _, ok := requestOpts["num_batch"]; ok {
		return false
	}
	if model != nil {
		if _, ok := model.Options["num_batch"]; ok {
			return false
		}
	}
	return true
}

// scheduleRunner schedules a runner after validating inputs such as capabilities and model options.
// It returns the allocated runner, model instance, and consolidated options if successful and error otherwise.
func (s *Server) scheduleRunner(ctx context.Context, name string, capable []model.Capability, requestOpts map[string]any, keepAlive *api.Duration, shift *bool) (llm.LlamaServer, *Model, *api.Options, error) {
	if name == "" {
		return nil, nil, nil, fmt.Errorf("model %w", errRequired)
	}

	model, err := GetModel(name)
	if err != nil {
		return nil, nil, nil, err
	}

	if err := model.CheckCapabilities(capable...); err != nil {
		return nil, nil, nil, fmt.Errorf("%s %w", name, err)
	}

	numCtxAuto := usesAutomaticNumCtx(model, requestOpts)
	embeddingBatchDefault := shouldApplyEmbeddingBatchDefault(model, requestOpts)
	numBatchAuto := usesAutomaticNumBatch(model, requestOpts) && !embeddingBatchDefault
	opts, err := s.modelOptionsWithEmbeddingBatchDefault(model, requestOpts, embeddingBatchDefault)
	if err != nil {
		return nil, nil, nil, err
	}

	runnerCh, errCh := s.sched.getRunner(ctx, model, opts, keepAlive, numCtxAuto, numBatchAuto)
	var runner *runnerRef
	select {
	case runner = <-runnerCh:
	case err = <-errCh:
		return nil, nil, nil, err
	}

	return runner.llama, model, &opts, nil
}

func signinURL() (string, error) {
	pubKey, err := auth.GetPublicKey()
	if err != nil {
		return "", err
	}

	encKey := base64.RawURLEncoding.EncodeToString([]byte(pubKey))
	h, _ := os.Hostname()
	return fmt.Sprintf("signinURLStr", url.PathEscape(h), encKey), nil
}

func (s *Server) GenerateHandler(c *gin.Context) {
	checkpointStart := time.Now()
	var req api.GenerateRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.TopLogprobs < 0 || req.TopLogprobs > 20 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "top_logprobs must be between 0 and 20"})
		return
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusNotFound, fmt.Sprintf("model '%s' not found", req.Model))
		return
	}

	//if modelRef.Source == modelSourceCloud {
	//	req.Model = modelRef.Base
	//	proxyCloudJSONRequest(c, req, cloudErrRemoteInferenceUnavailable)
	//	return
	//}

	name := modelRef.Name

	// We cannot currently consolidate this into GetModel because all we'll
	// induce infinite recursion given the current code structure.
	name, err = getExistingName(name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	m, err := GetModel(name.String())
	if err != nil {
		switch {
		case errors.Is(err, fs.ErrNotExist):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if req.TopLogprobs < 0 || req.TopLogprobs > 20 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "top_logprobs must be between 0 and 20"})
		return
	}

	//if modelRef.Source == modelSourceLocal && m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
	//	c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
	//	return
	//}

	// expire the runner if unload is requested (empty prompt, keep alive is 0)
	if req.Prompt == "" && req.KeepAlive != nil && req.KeepAlive.Duration == 0 {
		s.sched.expireRunner(m)

		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Response:   "",
			Done:       true,
			DoneReason: "unload",
		})
		return
	}

	if req.Raw && (req.Template != "" || req.System != "" || len(req.Context) > 0) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "raw mode does not support template, system, or context"})
		return
	}

	capable := []model.Capability{model.CapabilityCompletion}
	if req.Suffix != "" {
		capable = append(capable, model.CapabilityInsert)
	}

	modelCaps := m.Capabilities()
	if slices.Contains(modelCaps, model.CapabilityThinking) {
		capable = append(capable, model.CapabilityThinking)
		if req.Think == nil {
			req.Think = &api.ThinkValue{Value: true}
		}
	} else {
		if req.Think != nil && req.Think.Bool() {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support thinking", req.Model)})
			return
		}
	}

	r, m, opts, err := s.scheduleRunner(c.Request.Context(), name.String(), capable, req.Options, req.KeepAlive, req.Shift)
	if errors.Is(err, errCapabilityCompletion) {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support generate", req.Model)})
		return
	} else if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	checkpointLoaded := time.Now()

	// load the model
	if req.Prompt == "" {
		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Done:       true,
			DoneReason: "load",
		})
		return
	}

	if slices.Contains(m.Config.ModelFamilies, "mllama") && len(req.Images) > 1 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "this model only supports one image while more than one image requested"})
		return
	}

	media := make([]llm.MediaData, len(req.Images))
	for i := range req.Images {
		media[i] = llm.NewMediaData(i, req.Images[i])
	}

	prompt := req.Prompt
	var leadingBOS string
	if !req.Raw {
		tmpl := m.Template
		if req.Template != "" {
			tmpl, err = template.Parse(req.Template)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
		}

		var values template.Values
		if req.Suffix != "" {
			values.Prompt = prompt
			values.Suffix = req.Suffix
		} else {
			var msgs []api.Message
			if req.System != "" {
				msgs = append(msgs, api.Message{Role: "system", Content: req.System})
			} else if m.System != "" {
				msgs = append(msgs, api.Message{Role: "system", Content: m.System})
			}

			if req.Context == nil {
				msgs = append(msgs, m.Messages...)
			}

			userMsg := api.Message{Role: "user", Content: req.Prompt}
			for _, m := range media {
				userMsg.Images = append(userMsg.Images, m.Data)
			}
			values.Messages = append(msgs, userMsg)
		}

		values.Think = req.Think != nil && req.Think.Bool()
		values.ThinkLevel = ""
		if req.Think != nil {
			values.ThinkLevel = req.Think.String()
		}
		values.IsThinkSet = req.Think != nil

		var b bytes.Buffer
		if req.Context != nil {
			slog.Warn("the context field is deprecated and will be removed in a future version of Yollama")
			s, err := r.Detokenize(c.Request.Context(), req.Context)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			b.WriteString(s)
		}

		// check that we're in the `api/chat`-like flow, and if so, generate the
		// prompt the same way
		if values.Messages != nil && values.Suffix == "" && req.Template == "" {
			prompt, media, err = chatPrompt(c.Request.Context(), m, r.Tokenize, optionsForPrompt(opts, r), values.Messages, req.Think)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}
			if req.Context != nil {
				b.WriteString(prompt)
				prompt = b.String()
			}
			leadingBOS = leadingBOSForModel()
		} else {
			// Direct template execution flow.
			if err := tmpl.Execute(&b, values); err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			prompt = b.String()
		}
	}

	// If debug mode is enabled, return the rendered template instead of calling the model
	if req.DebugRenderOnly {
		c.JSON(http.StatusOK, api.GenerateResponse{
			Model:     req.Model,
			CreatedAt: time.Now().UTC(),
			DebugInfo: &api.DebugInfo{
				RenderedTemplate: prompt,
				ImageCount:       len(media),
			},
		})
		//return
	}

	

	ch := make(chan any)
	go func() {
		var sb strings.Builder
		defer close(ch)
		if err := r.Completion(c.Request.Context(), llm.CompletionRequest{
			Prompt:          prompt,
			Media:           media,
			Format:          req.Format,
			Options:         opts,
			Shift:           req.Shift == nil || *req.Shift,
			Logprobs:        req.Logprobs,
			TopLogprobs:     req.TopLogprobs,
			LeadingBOS:      leadingBOS,
		}, func(cr llm.CompletionResponse) {
			res := api.GenerateResponse{
				Model:     req.Model,
				CreatedAt: time.Now().UTC(),
				Response:  cr.Content,
				Done:      cr.Done,
				Metrics: api.Metrics{
					PromptEvalCount:    cr.PromptEvalCount,
					PromptEvalDuration: cr.PromptEvalDuration,
					EvalCount:          cr.EvalCount,
					EvalDuration:       cr.EvalDuration,
				},
				Logprobs: toAPILogprobs(cr.Logprobs),
			}

			if _, err := sb.WriteString(cr.Content); err != nil {
				ch <- gin.H{"error": err.Error()}
			}

			if cr.Done {
				res.DoneReason = cr.DoneReason.String()
				res.TotalDuration = time.Since(checkpointStart)
				res.LoadDuration = checkpointLoaded.Sub(checkpointStart)

				if !req.Raw {
					tokens, err := r.Tokenize(c.Request.Context(), prompt+sb.String())
					if err != nil {
						ch <- gin.H{"error": err.Error()}
						return
					}
					res.Context = tokens
				}
			}

			ch <- res
		}); err != nil {
			s.sched.expireRunnersForRuntimeOOM(m, err)
			var serr api.StatusError
			if errors.As(err, &serr) {
				ch <- gin.H{"error": serr.ErrorMessage, "status": serr.StatusCode}
			} else {
				ch <- gin.H{"error": err.Error()}
			}
		}
	}()

	if req.Stream != nil && !*req.Stream {
		var r api.GenerateResponse
		var allLogprobs []api.Logprob
		var sbThinking strings.Builder
		var sbContent strings.Builder
		for rr := range ch {
			switch t := rr.(type) {
			case api.GenerateResponse:
				sbThinking.WriteString(t.Thinking)
				sbContent.WriteString(t.Response)
				r = t
				// Accumulate logprobs from all chunks for non-streaming response
				if len(t.Logprobs) > 0 {
					allLogprobs = append(allLogprobs, t.Logprobs...)
				}
			case gin.H:
				msg, ok := t["error"].(string)
				if !ok {
					msg = "unexpected error format in response"
				}

				status, ok := t["status"].(int)
				if !ok {
					status = http.StatusInternalServerError
				}

				c.JSON(status, gin.H{"error": msg})
				return
			default:
				c.JSON(http.StatusInternalServerError, gin.H{"error": "unexpected response"})
				return
			}
		}

		r.Thinking = sbThinking.String()
		r.Response = sbContent.String()
		r.Logprobs = allLogprobs

		c.JSON(http.StatusOK, r)
		return
	}

	streamResponse(c, ch)
}

func (s *Server) EmbedHandler(c *gin.Context) {
	checkpointStart := time.Now()
	var req api.EmbedRequest
	err := c.ShouldBindJSON(&req)
	switch {
	case errors.Is(err, io.EOF):
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusNotFound, fmt.Sprintf("model '%s' not found", req.Model))
		return
	}

	//if modelRef.Source == modelSourceCloud {
	//	req.Model = modelRef.Base
	//	proxyCloudJSONRequest(c, req, cloudErrRemoteInferenceUnavailable)
	//	return
	//}

	var input []string

	switch i := req.Input.(type) {
	case string:
		if len(i) > 0 {
			input = append(input, i)
		}
	case []any:
		for _, v := range i {
			if _, ok := v.(string); !ok {
				c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "invalid input type"})
				return
			}
			input = append(input, v.(string))
		}
	default:
		if req.Input != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "invalid input type"})
			return
		}
	}

	name, err := getExistingName(modelRef.Name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	r, m, opts, err := s.scheduleRunner(c.Request.Context(), name.String(), []model.Capability{}, req.Options, req.KeepAlive, nil)
	slog.Warn("EmbedHandler annoyance:", opts)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	checkpointLoaded := time.Now()

	if len(input) == 0 {
		c.JSON(http.StatusOK, api.EmbedResponse{Model: req.Model, Embeddings: [][]float32{}})
		return
	}

	kvData, _, err := getModelData(m.ModelPath, false)
	slog.Warn("EmbedHandler annoyance:", kvData)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	ctx := c.Request.Context()

	embedWithRetry := func(text string) ([]float32, int, error) {
		emb, tokCount, err := r.Embedding(ctx, text)
		if err == nil {
			return emb, tokCount, nil
		}

		var serr api.StatusError
		if !errors.As(err, &serr) || serr.StatusCode != http.StatusBadRequest {
			return nil, 0, err
		}

		return emb, tokCount, err
	}

	var g errgroup.Group
	embeddings := make([][]float32, len(input))
	var totalTokens uint64
	for i, text := range input {
		g.Go(func() error {
			embedding, tokenCount, err := embedWithRetry(text)
			if err != nil {
				return err
			}
			// TODO: this first normalization should be done by the model
			embedding, err = normalize(embedding)
			if err != nil {
				return err
			}
			if req.Dimensions > 0 && req.Dimensions < len(embedding) {
				embedding, err = normalize(embedding[:req.Dimensions])
				if err != nil {
					return err
				}
			}
			embeddings[i] = embedding
			atomic.AddUint64(&totalTokens, uint64(tokenCount))
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		s.sched.expireRunnersForRuntimeOOM(m, err)
		var serr api.StatusError
		if errors.As(err, &serr) {
			c.AbortWithStatusJSON(serr.StatusCode, gin.H{
				"error": strings.TrimSpace(serr.ErrorMessage),
			})
			return
		}

		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{
			"error": strings.TrimSpace(err.Error()),
		})
		return
	}

	resp := api.EmbedResponse{
		Model:           req.Model,
		Embeddings:      embeddings,
		TotalDuration:   time.Since(checkpointStart),
		LoadDuration:    checkpointLoaded.Sub(checkpointStart),
		PromptEvalCount: int(totalTokens),
	}
	c.JSON(http.StatusOK, resp)
}

func normalize(vec []float32) ([]float32, error) {
	var sum float32
	for _, v := range vec {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, errors.New("embedding contains NaN or Inf values")
		}
		sum += v * v
	}

	norm := float32(1.0 / max(math.Sqrt(float64(sum)), 1e-12))
	for i := range vec {
		vec[i] *= norm
	}
	return vec, nil
}

func (s *Server) EmbeddingsHandler(c *gin.Context) {
	var req api.EmbeddingRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, "model is required")
		return
	}

	//if modelRef.Source == modelSourceCloud {
	//	req.Model = modelRef.Base
	//	proxyCloudJSONRequest(c, req, cloudErrRemoteInferenceUnavailable)
	//	return
	//}

	name := modelRef.Name

	r, m, _, err := s.scheduleRunner(c.Request.Context(), name.String(), []model.Capability{}, req.Options, req.KeepAlive, nil)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	// an empty request loads the model
	if req.Prompt == "" {
		c.JSON(http.StatusOK, api.EmbeddingResponse{Embedding: []float64{}})
		return
	}

	embedding, _, err := r.Embedding(c.Request.Context(), req.Prompt)
	if err != nil {
		s.sched.expireRunnersForRuntimeOOM(m, err)
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": strings.TrimSpace(err.Error())})
		return
	}

	var e []float64
	for _, v := range embedding {
		e = append(e, float64(v))
	}

	resp := api.EmbeddingResponse{
		Embedding: e,
	}
	c.JSON(http.StatusOK, resp)
}

func (s *Server) PullHandler(c *gin.Context) {
	var req api.PullRequest
	err := c.ShouldBindJSON(&req)
	switch {
	case errors.Is(err, io.EOF):
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	modelRef, err := parseNormalizePullModelRef(cmp.Or(req.Model, req.Name))
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, errtypes.InvalidModelNameErrMsg)
		return
	}

	name := modelRef.Name

	name, err = getExistingName(name)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ch := make(chan any)
	go func() {
		defer close(ch)
		fn := func(r api.ProgressResponse) {
			ch <- r
		}

		regOpts := &registryOptions{
			Insecure: req.Insecure,
		}

		ctx, cancel := context.WithCancel(c.Request.Context())
		defer cancel()

		if err := PullModel(ctx, name.DisplayShortest(), regOpts, fn); err != nil {
			ch <- gin.H{"error": err.Error()}
			return
		}

		s.refreshModelListCache(name)
	}()

	if req.Stream != nil && !*req.Stream {
		waitForStream(c, ch)
		return
	}

	streamResponse(c, ch)
}

// getExistingName searches the models directory for the longest prefix match of
// the input name and returns the input name with all existing parts replaced
// with each part found. If no parts are found, the input name is returned as
// is.
func getExistingName(n model.Name) (model.Name, error) {
	var zero model.Name
	existing, err := manifest.Manifests(true)
	if err != nil {
		return zero, err
	}
	var set model.Name // tracks parts already canonicalized
	for e := range existing {
		if set.Host == "" && strings.EqualFold(e.Host, n.Host) {
			n.Host = e.Host
		}
		if set.Namespace == "" && strings.EqualFold(e.Namespace, n.Namespace) {
			n.Namespace = e.Namespace
		}
		if set.Model == "" && strings.EqualFold(e.Model, n.Model) {
			n.Model = e.Model
		}
		if set.Tag == "" && strings.EqualFold(e.Tag, n.Tag) {
			n.Tag = e.Tag
		}
	}

	return n, nil
}

func (s *Server) DeleteHandler(c *gin.Context) {
	var r api.DeleteRequest
	if err := c.ShouldBindJSON(&r); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	modelRef, err := parseNormalizePullModelRef(cmp.Or(r.Model, r.Name))
	if err != nil {
		switch {
		case errors.Is(err, errConflictingModelSource):
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		case errors.Is(err, model.ErrUnqualifiedName):
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("name %q is invalid", cmp.Or(r.Model, r.Name))})
		default:
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		}
		return
	}

	n, err := getExistingName(modelRef.Name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", cmp.Or(r.Model, r.Name))})
		return
	}

	m, err := manifest.ParseNamedManifest(n)
	if err != nil {
		switch {
		case os.IsNotExist(err):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", cmp.Or(r.Model, r.Name))})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if err := m.Remove(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	s.deleteModelListCache(n)

	if err := m.RemoveLayers(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
}

func (s *Server) ShowHandler(c *gin.Context) {
	var req api.ShowRequest
	err := c.ShouldBindJSON(&req)
	switch {
	case errors.Is(err, io.EOF):
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.Model != "" {
		// noop
	} else if req.Name != "" {
		req.Model = req.Name
	} else {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}
	//requestedModel := req.Model

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, err.Error())
		return
	}

	name := modelRef.Name
	name, err = getExistingName(name)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	req.Model = name.DisplayShortest()

	var resp *api.ShowResponse
	if modelShowCacheable(req) && s.modelCaches != nil && s.modelCaches.show != nil {
		resp, err = s.modelCaches.show.GetLocal(req)
	} else {
		resp, err = GetModelInfo(req)
	}
	if err != nil {
		var statusErr api.StatusError
		switch {
		case os.IsNotExist(err):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		case errors.As(err, &statusErr):
			c.JSON(statusErr.StatusCode, gin.H{"error": statusErr.ErrorMessage})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if modelRef.Source == modelSourceLocal && resp.RemoteHost != "" {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", modelRef.Original)})
		return
	}

	c.JSON(http.StatusOK, resp)
}

func GetModelInfo(req api.ShowRequest) (*api.ShowResponse, error) {
	name := model.ParseName(req.Model)
	if !name.IsValid() {
		return nil, model.Unqualified(name)
	}
	name, err := getExistingName(name)
	if err != nil {
		return nil, err
	}

	m, err := GetModel(name.String())
	if err != nil {
		return nil, err
	}

	modelDetails := api.ModelDetails{
		ParentModel:       m.ParentModel,
		Format:            m.Config.ModelFormat,
		Family:            m.Config.ModelFamily,
		Families:          m.Config.ModelFamilies,
		ParameterSize:     m.Config.ModelType,
		QuantizationLevel: m.Config.FileType,
	}

	if req.System != "" {
		m.System = req.System
	}

	msgs := make([]api.Message, len(m.Messages))
	for i, msg := range m.Messages {
		msgs[i] = api.Message{Role: msg.Role, Content: msg.Content}
	}

	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		return nil, err
	}

	resp := &api.ShowResponse{
		License:      strings.Join(m.License, "\n"),
		System:       m.System,
		Template:     m.Template.String(),
		Details:      modelDetails,
		Messages:     msgs,
		Capabilities: m.Capabilities(),
		ModifiedAt:   mf.FileInfo().ModTime(),
		Requires:     m.Config.Requires,
		// Several integrations crash on a nil/omitempty+empty ModelInfo, so by
		// default we return an empty map.
		ModelInfo: make(map[string]any),
	}

	var params []string
	cs := 30
	for k, v := range m.Options {
		switch val := v.(type) {
		case []any:
			for _, nv := range val {
				params = append(params, fmt.Sprintf("%-*s %#v", cs, k, nv))
			}
		default:
			params = append(params, fmt.Sprintf("%-*s %#v", cs, k, v))
		}
	}
	resp.Parameters = strings.Join(params, "\n")

	if len(req.Options) > 0 {
		if m.Options == nil {
			m.Options = make(map[string]any)
		}
		for k, v := range req.Options {
			m.Options[k] = v
		}
	}

	var sb strings.Builder
	fmt.Fprintln(&sb, "# Modelfile generated by \"yollama show\"")
	modelfile := m.String()
	fmt.Fprintln(&sb, "# To build a new Modelfile based on this, replace FROM with:")
	fmt.Fprintf(&sb, "# FROM %s\n\n", m.ShortName)
	fmt.Fprint(&sb, modelfile)
	resp.Modelfile = sb.String()

	if slices.Contains(m.Capabilities(), model.CapabilityImage) {
		return resp, nil
	}

	kvData, tensors, err := getModelData(m.ModelPath, req.Verbose)
	if err != nil {
		return nil, err
	}

	resp.Template = selectedModelTemplate(m, kvData)
	if isUnknownQuantization(resp.Details.QuantizationLevel) {
		if fileType := kvData.FileType().String(); !isUnknownQuantization(fileType) {
			resp.Details.QuantizationLevel = fileType
		}
	}

	delete(kvData, "general.name")
	delete(kvData, "tokenizer.chat_template")
	resp.ModelInfo = kvData

	tensorData := make([]api.Tensor, len(tensors.Items()))
	for cnt, t := range tensors.Items() {
		tensorData[cnt] = api.Tensor{Name: t.Name, Type: t.Type(), Shape: t.Shape}
	}
	resp.Tensors = tensorData

	if len(m.ProjectorPaths) > 0 {
		projectorData, _, err := getModelData(m.ProjectorPaths[0], req.Verbose)
		if err != nil {
			return nil, err
		}
		resp.ProjectorInfo = projectorData
	}

	return resp, nil
}

func getModelData(digest string, verbose bool) (ggml.KV, ggml.Tensors, error) {
	maxArraySize := 0
	if verbose {
		maxArraySize = -1
	}
	data, err := llm.LoadModel(digest, maxArraySize)
	if err != nil {
		return nil, ggml.Tensors{}, err
	}

	kv := data.KV()

	if !verbose {
		for k := range kv {
			if t, ok := kv[k].([]any); len(t) > 5 && ok {
				kv[k] = []any{}
			}
		}
	}

	return kv, data.Tensors(), nil
}

func selectedModelTemplate(m *Model, kv ggml.KV) string {
	if m.HasChatTemplate && chatModeForModel(m) == chatExecutionModeNative {
		if chatTemplate := kv.String("tokenizer.chat_template"); chatTemplate != "" {
			return chatTemplate
		}
	}
	return m.Template.String()
}

func (s *Server) ListHandler(c *gin.Context) {
	if s.modelCaches == nil || s.modelCaches.modelList == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "model list cache unavailable"})
		return
	}

	models, err := s.modelCaches.modelList.List(c.Request.Context())
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, api.ListResponse{Models: models})
}

func (s *Server) CopyHandler(c *gin.Context) {
	var r api.CopyRequest
	if err := c.ShouldBindJSON(&r); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	src := model.ParseName(r.Source)
	if !src.IsValid() {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("source %q is invalid", r.Source)})
		return
	}
	src, err := getExistingName(src)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	dst := model.ParseName(r.Destination)
	if !dst.IsValid() {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("destination %q is invalid", r.Destination)})
		return
	}
	dst, err = getExistingName(dst)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := CopyModel(src, dst); errors.Is(err, os.ErrNotExist) {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model %q not found", r.Source)})
	} else if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
	} else {
		s.refreshModelListCache(dst)
	}
}

func (s *Server) HeadBlobHandler(c *gin.Context) {
	path, err := manifest.BlobsPath(c.Param("digest"))
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if _, err := os.Stat(path); err != nil {
		c.AbortWithStatusJSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("blob %q not found", c.Param("digest"))})
		return
	}

	c.Status(http.StatusOK)
}

func (s *Server) CreateBlobHandler(c *gin.Context) {
	if ib, ok := intermediateBlobs[c.Param("digest")]; ok {
		p, err := manifest.BlobsPath(ib)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		if _, err := os.Stat(p); errors.Is(err, os.ErrNotExist) {
			slog.Info("evicting intermediate blob which no longer exists", "digest", ib)
			delete(intermediateBlobs, c.Param("digest"))
		} else if err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		} else {
			c.Status(http.StatusOK)
			return
		}
	}

	path, err := manifest.BlobsPath(c.Param("digest"))
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	_, err = os.Stat(path)
	switch {
	case errors.Is(err, os.ErrNotExist):
		// noop
	case err != nil:
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	default:
		c.Status(http.StatusOK)
		return
	}

	layer, err := manifest.NewLayer(c.Request.Body, "")
	if err != nil {
		c.AbortWithStatusJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if layer.Digest != c.Param("digest") {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("digest mismatch, expected %q, got %q", c.Param("digest"), layer.Digest)})
		return
	}

	c.Status(http.StatusCreated)
}

func isLocalIP(ip netip.Addr) bool {
	if interfaces, err := net.Interfaces(); err == nil {
		for _, iface := range interfaces {
			addrs, err := iface.Addrs()
			if err != nil {
				continue
			}

			for _, a := range addrs {
				if parsed, _, err := net.ParseCIDR(a.String()); err == nil {
					if parsed.String() == ip.String() {
						return true
					}
				}
			}
		}
	}

	return false
}

func allowedHost(host string) bool {
	host = strings.ToLower(host)

	if host == "" || host == "localhost" {
		return true
	}

	if hostname, err := os.Hostname(); err == nil && host == strings.ToLower(hostname) {
		return true
	}

	tlds := []string{
		"localhost",
		"local",
		"internal",
	}

	// check if the host is a local TLD
	for _, tld := range tlds {
		if strings.HasSuffix(host, "."+tld) {
			return true
		}
	}

	return false
}

func (s *Server) GenerateRoutes() (http.Handler, error) {
	corsConfig := cors.DefaultConfig()
	corsConfig.AllowWildcard = true
	corsConfig.AllowBrowserExtensions = true
	corsConfig.AllowHeaders = []string{
		"Authorization",
		"Content-Type",
		"User-Agent",
		"Accept",
		"X-Requested-With",
	}
	corsConfig.AllowOrigins = envconfig.AllowedOrigins()

	r := gin.Default()
	r.HandleMethodNotAllowed = true
	r.Use(
		cors.New(corsConfig),
	)

	// General
	r.HEAD("/", func(c *gin.Context) { c.String(http.StatusOK, "Yollama is running") })
	r.GET("/", func(c *gin.Context) { c.String(http.StatusOK, "Yollama is running") })
	r.HEAD("/api/version", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"version": version.Version}) })
	r.GET("/api/version", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"version": version.Version}) })
	r.GET("/api/status", s.StatusHandler)

	// Local model cache management (new implementation is at end of function)
	r.POST("/api/pull", s.PullHandler)
	r.HEAD("/api/tags", s.ListHandler)
	r.GET("/api/tags", s.ListHandler)
	r.POST("/api/show", s.ShowHandler)
	r.DELETE("/api/delete", s.DeleteHandler)

	// Create
	r.POST("/api/create", s.CreateHandler)
	r.POST("/api/blobs/:digest", s.CreateBlobHandler)
	r.HEAD("/api/blobs/:digest", s.HeadBlobHandler)
	r.POST("/api/copy", s.CopyHandler)

	// Inference
	r.GET("/api/ps", s.PsHandler)
	r.POST("/api/generate", s.withInferenceRequestLogging("/api/generate", s.GenerateHandler)...)
	r.POST("/api/chat", s.withInferenceRequestLogging("/api/chat", s.ChatHandler)...)
	r.POST("/api/embed", s.EmbedHandler)
	r.POST("/api/embeddings", s.EmbeddingsHandler)

	//Stop
	r.POST("/api/stop", s.StopModelHandler)
	r.POST("/api/stop/all", s.StopAllModelsHandler)

	return r, nil
}

func Serve(ln net.Listener) error {
	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))
	slog.Info("server config", "env", envconfig.Values())
	cloudDisabled, _ := internalcloud.Status()
	slog.Info(fmt.Sprintf("Yollama cloud disabled: %t", cloudDisabled))

	blobsDir, err := manifest.BlobsPath("")
	if err != nil {
		return err
	}
	if err := fixBlobs(blobsDir); err != nil {
		return err
	}

	if !envconfig.NoPrune() {
		if _, err := manifest.Manifests(false); err != nil {
			slog.Warn("corrupt manifests detected, skipping prune operation.  Re-pull or delete to clear", "error", err)
		} else {
			// clean up unused layers and manifests
			if err := PruneLayers(); err != nil {
				return err
			}

			manifestsPath, err := manifest.Path()
			if err != nil {
				return err
			}

			if err := manifest.PruneDirectory(manifestsPath); err != nil {
				return err
			}
		}
	}

	s := &Server{
		addr:        ln.Addr(),
		modelCaches: newModelCaches(),
	}
	if err := s.initRequestLogging(); err != nil {
		return err
	}

	h, err := s.GenerateRoutes()
	if err != nil {
		return err
	}

	http.Handle("/", h)

	ctx, done := context.WithCancel(context.Background())
	schedCtx, schedDone := context.WithCancel(ctx)
	sched := InitScheduler(schedCtx)
	s.sched = sched
	s.modelCaches.Start(ctx)

	slog.Info(fmt.Sprintf("Yollama Started with IP-Address: %s | Current Version: %s", ln.Addr(), version.Version))
	srvr := &http.Server{
		// Use http.DefaultServeMux so we get net/http/pprof for
		// free.
		Handler: nil,
	}

	// listen for a ctrl+c and stop any loaded llm
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-signals
		srvr.Close()
		schedDone()
		sched.unloadAllRunners()
		done()
	}()

	s.sched.Run(schedCtx)

	// register the experimental webp decoder
	// so webp images can be used in multimodal inputs
	image.RegisterFormat("webp", "RIFF????WEBP", webp.Decode, webp.DecodeConfig)

	// At startup we retrieve GPU information so we can get log messages before loading a model
	// This will log warnings to the log in case we have problems with detected GPUs
	gpus := discover.GPUDevices(ctx, nil)
	discover.LogDetails(gpus)

	var totalVRAM uint64
	for _, gpu := range gpus {
		totalVRAM += gpu.TotalMemory - envconfig.GpuOverhead()
	}

	// Set default context based on VRAM tier
	// Use slightly lower thresholds (47/23 GiB vs. 48/24 GiB) to account for small differences in the exact value
	switch {
	case totalVRAM >= 47*format.GibiByte:
		s.defaultNumCtx = 262144
	case totalVRAM >= 23*format.GibiByte:
		s.defaultNumCtx = 32768
	default:
		s.defaultNumCtx = 4096
	}
	slog.Info("vram-based default context", "total_vram", format.HumanBytes2(totalVRAM), "default_num_ctx", s.defaultNumCtx)

	err = srvr.Serve(ln)
	// If server is closed from the signal handler, wait for the ctx to be done
	// otherwise error out quickly
	if !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	<-ctx.Done()
	return nil
}

func waitForStream(c *gin.Context, ch chan any) {
	c.Header("Content-Type", "application/json")
	var latest api.ProgressResponse
	for resp := range ch {
		switch r := resp.(type) {
		case api.ProgressResponse:
			latest = r
		case gin.H:
			status, ok := r["status"].(int)
			if !ok {
				status = http.StatusInternalServerError
			}
			errorMsg, ok := r["error"].(string)
			if !ok {
				errorMsg = "unknown error"
			}
			c.JSON(status, gin.H{"error": errorMsg})
			return
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": "unknown message type"})
			return
		}
	}

	c.JSON(http.StatusOK, latest)
}

func streamResponse(c *gin.Context, ch chan any) {
	c.Header("Content-Type", "application/x-ndjson")
	c.Stream(func(w io.Writer) bool {
		val, ok := <-ch
		if !ok {
			return false
		}

		// errors are provided as a gin.H with an "error" field and
		// an optional "status" field.  For errors that are streamed
		// before any content, we need to set the status code and
		// content type for the error.
		if h, ok := val.(gin.H); ok {
			if e, ok := h["error"].(string); ok {
				status, ok := h["status"].(int)
				if !ok {
					status = http.StatusInternalServerError
				}

				if !c.Writer.Written() {
					c.Header("Content-Type", "application/json")
					c.JSON(status, gin.H{"error": e})
				} else {
					if err := json.NewEncoder(c.Writer).Encode(gin.H{"error": e}); err != nil {
						slog.Error("streamResponse failed to encode json error", "error", err)
					}
				}

				return false
			}
		}

		bts, err := json.Marshal(val)
		if err != nil {
			slog.Info(fmt.Sprintf("streamResponse: json.Marshal failed with %s", err))
			return false
		}

		// Delineate chunks with new-line delimiter
		bts = append(bts, '\n')
		if _, err := w.Write(bts); err != nil {
			slog.Info(fmt.Sprintf("streamResponse: w.Write failed with %s", err))
			return false
		}

		return true
	})
}

func (s *Server) StatusHandler(c *gin.Context) {
	disabled, source := internalcloud.Status()
	c.JSON(http.StatusOK, api.StatusResponse{
		Cloud: api.CloudStatus{
			Disabled: disabled,
			Source:   source,
		},
	})
}

func (s *Server) PsHandler(c *gin.Context) {
	models := []api.ProcessModelResponse{}

	for _, v := range s.sched.loaded {
		m := v.model
		displayName := model.ParseName(m.ShortName).DisplayShortest()
		modelDetails := api.ModelDetails{
			Format:            m.Config.ModelFormat,
			Family:            m.Config.ModelFamily,
			Families:          m.Config.ModelFamilies,
			ParameterSize:     m.Config.ModelType,
			QuantizationLevel: m.Config.FileType,
		}

		mr := api.ProcessModelResponse{
			Model:     displayName,
			Name:      displayName,
			Size:      int64(v.totalSize),
			SizeVRAM:  int64(v.vramSize),
			Digest:    m.Digest,
			Details:   modelDetails,
			ExpiresAt: v.expiresAt,
		}
		if v.llama != nil {
			mr.ContextLength = v.llama.ContextLength()
			total, vram := v.llama.MemorySize()
			mr.Size = int64(total)
			mr.SizeVRAM = int64(vram)
		}
		// The scheduler waits to set expiresAt, so if a model is loading it's
		// possible that it will be set to the unix epoch. For those cases, just
		// calculate the time w/ the sessionDuration instead.
		var epoch time.Time
		if v.expiresAt == epoch {
			mr.ExpiresAt = time.Now().Add(v.sessionDuration)
		}

		models = append(models, mr)
	}

	slices.SortStableFunc(models, func(i, j api.ProcessModelResponse) int {
		// longest duration remaining listed first
		return cmp.Compare(j.ExpiresAt.Unix(), i.ExpiresAt.Unix())
	})

	c.JSON(http.StatusOK, api.ProcessResponse{Models: models})
}

func leadingBOSForModel() string {
	return ""
}

func optionsForPrompt(opts *api.Options, runner llm.LlamaServer) *api.Options {
	if opts == nil || runner == nil {
		return opts
	}

	if ctxLen := runner.ContextLength(); ctxLen > 0 && opts.NumCtx > ctxLen {
		copied := *opts
		copied.NumCtx = ctxLen
		return &copied
	}

	return opts
}

type chatExecutionMode int

const (
	chatExecutionModeNative chatExecutionMode = iota
	chatExecutionModeRendered
)

func chatModeForModel(m *Model) chatExecutionMode {
	return chatExecutionModeNative
}

func llamaServerConfigForModel(m *Model) llm.LlamaServerConfig {
	return llm.LlamaServerConfig{
		DisableJinja:   usesYollamaRenderedChat(m),
	}
}

func usesYollamaRenderedChat(m *Model) bool {
	return shouldUseGoTemplate(m)
}

func shouldUseGoTemplate(m *Model) bool {
	return false
}

func writeChatResponse(c *gin.Context, req api.ChatRequest, ch chan any) {
	if req.Stream != nil && !*req.Stream {
		var resp api.ChatResponse
		var allLogprobs []api.Logprob
		var sbThinking strings.Builder
		var sbContent strings.Builder
		for rr := range ch {
			switch t := rr.(type) {
			case api.ChatResponse:
				sbThinking.WriteString(t.Message.Thinking)
				sbContent.WriteString(t.Message.Content)
				resp = t
				if len(t.Logprobs) > 0 {
					allLogprobs = append(allLogprobs, t.Logprobs...)
				}
			case gin.H:
				msg, ok := t["error"].(string)
				if !ok {
					msg = "unexpected error format in response"
				}

				status, ok := t["status"].(int)
				if !ok {
					status = http.StatusInternalServerError
				}

				c.JSON(status, gin.H{"error": msg})
				return
			default:
				c.JSON(http.StatusInternalServerError, gin.H{"error": "unexpected response"})
				return
			}
		}

		resp.Message.Content = sbContent.String()
		resp.Message.Thinking = sbThinking.String()
		resp.Logprobs = allLogprobs

		c.JSON(http.StatusOK, resp)
		return
	}

	streamResponse(c, ch)
}

func (s *Server) ChatHandler(c *gin.Context) {
	checkpointStart := time.Now()

	var req api.ChatRequest
	if err := c.ShouldBindJSON(&req); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.TopLogprobs < 0 || req.TopLogprobs > 20 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "top_logprobs must be between 0 and 20"})
		return
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusBadRequest, "model is required")
		return
	}

	name := modelRef.Name

	name, err = getExistingName(name)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model is required"})
		return
	}

	m, err := GetModel(name.String())
	if err != nil {
		switch {
		case os.IsNotExist(err):
			c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		case err.Error() == errtypes.InvalidModelNameErrMsg:
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		default:
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if req.TopLogprobs < 0 || req.TopLogprobs > 20 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "top_logprobs must be between 0 and 20"})
		return
	}

	if modelRef.Source == modelSourceLocal && m.Config.RemoteHost != "" && m.Config.RemoteModel != "" {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	// expire the runner
	if len(req.Messages) == 0 && req.KeepAlive != nil && req.KeepAlive.Duration == 0 {
		s.sched.expireRunner(m)

		c.JSON(http.StatusOK, api.ChatResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Message:    api.Message{Role: "assistant"},
			Done:       true,
			DoneReason: "unload",
		})
		return
	}

	capable := []model.Capability{model.CapabilityCompletion}
	modelCaps := m.Capabilities()
	if slices.Contains(modelCaps, model.CapabilityThinking) {
		capable = append(capable, model.CapabilityThinking)
		if req.Think == nil {
			req.Think = &api.ThinkValue{Value: true}
		}
	} else {
		if req.Think != nil && req.Think.Bool() {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support thinking", req.Model)})
			return
		}
	}

	r, m, opts, err := s.scheduleRunner(c.Request.Context(), name.String(), capable, req.Options, req.KeepAlive, req.Shift)
	if errors.Is(err, errCapabilityCompletion) {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("%q does not support chat", req.Model)})
		return
	} else if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	checkpointLoaded := time.Now()

	if len(req.Messages) == 0 {
		c.JSON(http.StatusOK, api.ChatResponse{
			Model:      req.Model,
			CreatedAt:  time.Now().UTC(),
			Message:    api.Message{Role: "assistant"},
			Done:       true,
			DoneReason: "load",
		})
		return
	}

	msgs := append(m.Messages, req.Messages...)
	if req.Messages[0].Role != "system" && m.System != "" {
		msgs = append([]api.Message{{Role: "system", Content: m.System}}, msgs...)
	}
	msgs = filterThinkTags(msgs, m)

	if chatModeForModel(m) == chatExecutionModeNative {
		s.handleNativeChat(c, req, m, r, opts, msgs, checkpointStart, checkpointLoaded)
		return
	}

	promptOpts := optionsForPrompt(opts, r)
	prompt, media, err := chatPrompt(c.Request.Context(), m, r.Tokenize, promptOpts, msgs, req.Think)
	if err != nil {
		slog.Error("chat prompt error", "error", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// If debug mode is enabled, return the rendered template instead of calling the model
	if req.DebugRenderOnly {
		c.JSON(http.StatusOK, api.ChatResponse{
			Model:     req.Model,
			CreatedAt: time.Now().UTC(),
			DebugInfo: &api.DebugInfo{
				RenderedTemplate: prompt,
				ImageCount:       len(media),
			},
		})
		//return
	}

	type structuredOutputsState int
	const (
		structuredOutputsState_None structuredOutputsState = iota
		structuredOutputsState_ReadyToApply
		structuredOutputsState_Applying
	)

	ch := make(chan any)
	go func() {
		defer close(ch)

		structuredOutputsState := structuredOutputsState_None

		for {
			var tb strings.Builder

			currentFormat := req.Format

			// sets up new context given parent context per request
			ctx, cancel := context.WithCancel(c.Request.Context())
			slog.Info("chatHandler is annoying:", cancel)
			err := r.Completion(ctx, llm.CompletionRequest{
				Prompt:          prompt,
				Media:           media,
				Format:          currentFormat,
				Options:         opts,
				Shift:           req.Shift == nil || *req.Shift,
				Logprobs:        req.Logprobs,
				TopLogprobs:     req.TopLogprobs,
				LeadingBOS:      leadingBOSForModel(),
			}, func(r llm.CompletionResponse) {
				res := api.ChatResponse{
					Model:     req.Model,
					CreatedAt: time.Now().UTC(),
					Message:   api.Message{Role: "assistant", Content: r.Content},
					Done:      r.Done,
					Metrics: api.Metrics{
						PromptEvalCount:    r.PromptEvalCount,
						PromptEvalDuration: r.PromptEvalDuration,
						EvalCount:          r.EvalCount,
						EvalDuration:       r.EvalDuration,
					},
					Logprobs: toAPILogprobs(r.Logprobs),
				}

				if r.Done {
					res.DoneReason = r.DoneReason.String()
					res.TotalDuration = time.Since(checkpointStart)
					res.LoadDuration = checkpointLoaded.Sub(checkpointStart)
				}

				ch <- res
			})
			if err != nil {
				if structuredOutputsState == structuredOutputsState_ReadyToApply && strings.Contains(err.Error(), "context canceled") && c.Request.Context().Err() == nil {
					// only ignores error if it's a context cancellation due to setting structured outputs
				} else {
					s.sched.expireRunnersForRuntimeOOM(m, err)
					var serr api.StatusError
					if errors.As(err, &serr) {
						ch <- gin.H{"error": serr.ErrorMessage, "status": serr.StatusCode}
					} else {
						ch <- gin.H{"error": err.Error()}
					}
					return
				}
			}

			// ignored structured outputs cancellation falls through to here, start a new request with the structured outputs and updated prompt. use the
			if structuredOutputsState == structuredOutputsState_ReadyToApply {
				structuredOutputsState = structuredOutputsState_Applying
				msg := api.Message{
					Role:     "assistant",
					Thinking: tb.String(),
				}

				msgs = append(msgs, msg)
				prompt, _, err = chatPrompt(c.Request.Context(), m, r.Tokenize, promptOpts, msgs, req.Think)
				if err != nil {
					slog.Error("chat prompt error applying structured outputs", "error", err)
					ch <- gin.H{"error": err.Error()}
					return
				}
				continue
			}

			break
		}
	}()

	writeChatResponse(c, req, ch)
}

func (s *Server) handleNativeChat(c *gin.Context, req api.ChatRequest, m *Model, r llm.LlamaServer, opts *api.Options, msgs []api.Message, checkpointStart, checkpointLoaded time.Time) {
	nativeReq := llm.ChatRequest{
		Messages:    msgs,
		Format:      req.Format,
		Options:     opts,
		Think:       req.Think,
		Shift:       req.Shift == nil || *req.Shift,
		Logprobs:    req.Logprobs,
		TopLogprobs: req.TopLogprobs,
	}
	var err error
	nativeReq = nativeReq
	if err != nil {
		slog.Error("chat template prompt error", "error", err)
		var serr api.StatusError
		if errors.As(err, &serr) {
			c.JSON(serr.StatusCode, gin.H{"error": serr.ErrorMessage})
		} else {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		}
		return
	}

	if req.DebugRenderOnly {
		prompt, err := r.ApplyChatTemplate(c.Request.Context(), nativeReq)
		if err != nil {
			var serr api.StatusError
			if errors.As(err, &serr) {
				c.JSON(serr.StatusCode, gin.H{"error": serr.ErrorMessage})
			} else {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			}
			return
		}

		c.JSON(http.StatusOK, api.ChatResponse{
			Model:     req.Model,
			CreatedAt: time.Now().UTC(),
			DebugInfo: &api.DebugInfo{
				RenderedTemplate: prompt,
				ImageCount:       countChatImages(msgs),
			},
		})
		//return
	}

	ch := make(chan any)
	go func() {
		defer close(ch)

		err := r.Chat(c.Request.Context(), nativeReq, func(r llm.ChatResponse) {
			res := api.ChatResponse{
				Model:     req.Model,
				CreatedAt: time.Now().UTC(),
				Message:   r.Message,
				Done:      r.Done,
				Metrics: api.Metrics{
					PromptEvalCount:    r.PromptEvalCount,
					PromptEvalDuration: r.PromptEvalDuration,
					EvalCount:          r.EvalCount,
					EvalDuration:       r.EvalDuration,
				},
				Logprobs: toAPILogprobs(r.Logprobs),
			}

			if res.Message.Role == "" {
				res.Message.Role = "assistant"
			}

			if r.Done {
				res.DoneReason = r.DoneReason.String()
				res.TotalDuration = time.Since(checkpointStart)
				res.LoadDuration = checkpointLoaded.Sub(checkpointStart)
			}

			ch <- res
		})
		if err != nil {
			s.sched.expireRunnersForRuntimeOOM(m, err)
			var serr api.StatusError
			if errors.As(err, &serr) {
				ch <- gin.H{"error": serr.ErrorMessage, "status": serr.StatusCode}
			} else {
				ch <- gin.H{"error": err.Error()}
			}
		}
	}()

	writeChatResponse(c, req, ch)
}

func countChatImages(msgs []api.Message) int {
	var count int
	for _, msg := range msgs {
		count += len(msg.Images)
	}
	return count
}

func handleScheduleError(c *gin.Context, name string, err error) {
	switch {
	case errors.Is(err, errCapabilities), errors.Is(err, errRequired):
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
	case errors.Is(err, context.Canceled):
		c.JSON(499, gin.H{"error": "request canceled"})
	case errors.Is(err, ErrMaxQueue):
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": err.Error()})
	case errors.Is(err, os.ErrNotExist):
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model %q not found, try pulling it first", name)})
	default:
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
	}
}

func filterThinkTags(msgs []api.Message, m *Model) []api.Message {
	return msgs
}