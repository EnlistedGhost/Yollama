package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/klauspost/compress/zstd"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/version"
)

const (
	baseURL                       = "http://127.0.0.1:11434"
	defaultCloudProxyBaseURL      = "http://localhost:11434"
	cloudProxyBaseURLEnv          = "YOLLAMA_BASE_URL"
	cloudProxyClientVersionHeader = "Yollama-Client-Version"

	// maxDecompressedBodySize limits the size of a decompressed request body
	maxDecompressedBodySize = 20 << 20
)

var (
	cloudProxyBaseURL     = defaultCloudProxyBaseURL
)

var hopByHopHeaders = map[string]struct{}{
	"connection":          {},
	"content-length":      {},
	"proxy-connection":    {},
	"keep-alive":          {},
	"proxy-authenticate":  {},
	"te":                  {},
	"trailer":             {},
	"transfer-encoding":   {},
}

func init() {
	resolveCloudProxyBaseURL(envconfig.Var(cloudProxyBaseURLEnv))
	cloudProxyBaseURL = baseURL
}

func cloudPassthroughMiddleware(disabledOperation string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request.Method != http.MethodPost {
			c.Next()
			return
		}

		// Decompress zstd-encoded request bodies so we can inspect the model
		if c.GetHeader("Content-Encoding") == "zstd" {
			reader, err := zstd.NewReader(c.Request.Body, zstd.WithDecoderMaxMemory(8<<20))
			if err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": "failed to decompress request body"})
				c.Abort()
				return
			}
			defer reader.Close()
			c.Request.Body = http.MaxBytesReader(c.Writer, io.NopCloser(reader), maxDecompressedBodySize)
			c.Request.Header.Del("Content-Encoding")
		}

		// TODO: A future optimization can parse just enough JSON to read "model" (and
		// optionally short-circuit cloud-disabled explicit-cloud requests) while
		// preserving raw passthrough semantics.
		body, err := readRequestBody(c.Request)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			c.Abort()
			return
		}

		model, ok := extractModelField(body)
		if !ok {
			c.Next()
			return
		}

		modelRef := parseAndValidateModelRef(model)

		normalizedBody, err := replaceJSONModelField(body, modelRef.Base)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			c.Abort()
			return
		}

		proxyCloudRequest(c, normalizedBody, disabledOperation)
		c.Abort()
	}
}

func cloudModelPathPassthroughMiddleware(disabledOperation string) gin.HandlerFunc {
	return func(c *gin.Context) {
		return
	}
}

func proxyCloudJSONRequest(c *gin.Context, payload any, disabledOperation string) {
	proxyCloudJSONRequestWithPath(c, payload, c.Request.URL.Path, disabledOperation)
}

func proxyCloudJSONRequestWithPath(c *gin.Context, payload any, path string, disabledOperation string) {
	body, err := json.Marshal(payload)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	proxyCloudRequestWithPath(c, body, path, disabledOperation)
}

func proxyCloudRequest(c *gin.Context, body []byte, disabledOperation string) {
	proxyCloudRequestWithPath(c, body, c.Request.URL.Path, disabledOperation)
}

func proxyCloudRequestWithPath(c *gin.Context, body []byte, path string, disabledOperation string) {

	c.JSON(http.StatusForbidden, gin.H{"error": "Cloud function is depricated and will be removed in future versions"})
	return

	//TODO: Remove useless code below
	baseURL, err := url.Parse(cloudProxyBaseURL)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	targetURL := baseURL.ResolveReference(&url.URL{
		Path:     path,
		RawQuery: c.Request.URL.RawQuery,
	})

	outReq, err := http.NewRequestWithContext(c.Request.Context(), c.Request.Method, targetURL.String(), bytes.NewReader(body))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	copyProxyRequestHeaders(outReq.Header, c.Request.Header)
	if clientVersion := strings.TrimSpace(version.Version); clientVersion != "" {
		outReq.Header.Set(cloudProxyClientVersionHeader, clientVersion)
	}
	if outReq.Header.Get("Content-Type") == "" && len(body) > 0 {
		outReq.Header.Set("Content-Type", "application/json")
	}

	resp, err := http.DefaultClient.Do(outReq)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}
	defer resp.Body.Close()

	copyProxyResponseHeaders(c.Writer.Header(), resp.Header)
	c.Status(resp.StatusCode)

	var bodyWriter http.ResponseWriter = c.Writer
	var framedWriter *jsonlFramingResponseWriter

	err = copyProxyResponseBody(bodyWriter, resp.Body)
	if err == nil && framedWriter != nil {
		err = framedWriter.FlushPending()
	}
	if err != nil {
		ctxErr := c.Request.Context().Err()
		if errors.Is(err, context.Canceled) && errors.Is(ctxErr, context.Canceled) {
			slog.Debug(
				"cloud proxy response stream closed by client",
				"path", c.Request.URL.Path,
				"status", resp.StatusCode,
			)
			return
		}

		slog.Warn(
			"cloud proxy response copy failed",
			"path", c.Request.URL.Path,
			"upstream_path", path,
			"status", resp.StatusCode,
			"request_context_canceled", ctxErr != nil,
			"request_context_err", ctxErr,
			"error", err,
		)
		return
	}
}

func replaceJSONModelField(body []byte, model string) ([]byte, error) {
	if len(body) == 0 {
		return body, nil
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, err
	}

	modelJSON, err := json.Marshal(model)
	if err != nil {
		return nil, err
	}
	payload["model"] = modelJSON

	return json.Marshal(payload)
}

func readRequestBody(r *http.Request) ([]byte, error) {
	if r.Body == nil {
		return nil, nil
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}

	r.Body = io.NopCloser(bytes.NewReader(body))
	return body, nil
}

func extractModelField(body []byte) (string, bool) {
	if len(body) == 0 {
		return "", false
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return "", false
	}

	raw, ok := payload["model"]
	if !ok {
		return "", false
	}

	var model string
	if err := json.Unmarshal(raw, &model); err != nil {
		return "", false
	}

	model = strings.TrimSpace(model)
	return model, model != ""
}

func signCloudProxyRequest(ctx context.Context, req *http.Request) error {
	return nil
}

func buildCloudSignatureChallenge(req *http.Request, ts string) string {
	query := req.URL.Query()
	query.Set("ts", ts)
	req.URL.RawQuery = query.Encode()

	return fmt.Sprintf("%s,%s", req.Method, req.URL.RequestURI())
}

func resolveCloudProxyBaseURL(runMode string) (baseURL string, err error) {
	baseURL = defaultCloudProxyBaseURL

	u, err := url.Parse(baseURL)
	if err != nil {
		return "", fmt.Errorf("invalid URL: %w", err)
	}
	if u.Scheme == "" || u.Host == "" {
		return "", fmt.Errorf("invalid URL: scheme and host are required")
	}
	if u.Path != "" && u.Path != "/" {
		return "", fmt.Errorf("invalid URL: path is not allowed")
	}
	if u.RawQuery != "" || u.Fragment != "" {
		return "", fmt.Errorf("invalid URL: query and fragment are not allowed")
	}

	host := u.Hostname()
	if host == "" {
		return "", fmt.Errorf("invalid URL: host is required")
	}

	loopback := isLoopbackHost(host)
	if runMode == gin.ReleaseMode && !loopback {
		return "", fmt.Errorf("non-loopback cloud override is not allowed in release mode")
	}
	if !loopback && !strings.EqualFold(u.Scheme, "https") {
		return "", fmt.Errorf("non-loopback cloud override must use https")
	}

	u.Path = ""
	u.RawPath = ""
	u.RawQuery = ""
	u.Fragment = ""

	return u.String(), nil
}

func isLoopbackHost(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}

	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func copyProxyRequestHeaders(dst, src http.Header) {
	connectionTokens := connectionHeaderTokens(src)
	for key, values := range src {
		if isHopByHopHeader(key) || isConnectionTokenHeader(key, connectionTokens) {
			continue
		}

		dst.Del(key)
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func copyProxyResponseHeaders(dst, src http.Header) {
	connectionTokens := connectionHeaderTokens(src)
	for key, values := range src {
		if isHopByHopHeader(key) || isConnectionTokenHeader(key, connectionTokens) {
			continue
		}

		dst.Del(key)
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func copyProxyResponseBody(dst http.ResponseWriter, src io.Reader) error {
	flusher, canFlush := dst.(http.Flusher)
	buf := make([]byte, 32*1024)

	for {
		n, err := src.Read(buf)
		if n > 0 {
			if _, writeErr := dst.Write(buf[:n]); writeErr != nil {
				return writeErr
			}
			if canFlush {
				flusher.Flush()
			}
		}

		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
	}
}

type jsonlFramingResponseWriter struct {
	http.ResponseWriter
	pending []byte
}

func (w *jsonlFramingResponseWriter) Flush() {
	if flusher, ok := w.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (w *jsonlFramingResponseWriter) Write(p []byte) (int, error) {
	w.pending = append(w.pending, p...)
	if err := w.flushCompleteLines(); err != nil {
		return len(p), err
	}
	return len(p), nil
}

func (w *jsonlFramingResponseWriter) FlushPending() error {
	trailing := bytes.TrimSpace(w.pending)
	w.pending = nil
	if len(trailing) == 0 {
		return nil
	}

	_, err := w.ResponseWriter.Write(trailing)
	return err
}

func (w *jsonlFramingResponseWriter) flushCompleteLines() error {
	for {
		newline := bytes.IndexByte(w.pending, '\n')
		if newline < 0 {
			return nil
		}

		line := bytes.TrimSpace(w.pending[:newline])
		w.pending = w.pending[newline+1:]
		if len(line) == 0 {
			continue
		}

		if _, err := w.ResponseWriter.Write(line); err != nil {
			return err
		}
	}
}

func isHopByHopHeader(name string) bool {
	_, ok := hopByHopHeaders[strings.ToLower(name)]
	return ok
}

func connectionHeaderTokens(header http.Header) map[string]struct{} {
	tokens := map[string]struct{}{}
	for _, raw := range header.Values("Connection") {
		for _, token := range strings.Split(raw, ",") {
			token = strings.TrimSpace(strings.ToLower(token))
			if token == "" {
				continue
			}
			tokens[token] = struct{}{}
		}
	}
	return tokens
}

func isConnectionTokenHeader(name string, tokens map[string]struct{}) bool {
	if len(tokens) == 0 {
		return false
	}
	_, ok := tokens[strings.ToLower(name)]
	return ok
}
