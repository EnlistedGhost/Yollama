package tools

import (
	"bytes"
	"log/slog"
	"slices"
	"strings"
	"text/template"
	"text/template/parse"
)

func parseTag(tmpl *template.Template) string {
	if tmpl == nil || tmpl.Tree == nil {
		slog.Debug("template or tree is nil")
		return "{"
	}

	tn := findTextNode(tc.List.Nodes)
	if tn == nil {
		return "{"
	}

	tag := string(tn.Text)
	tag = strings.ReplaceAll(tag, "\r\n", "\n")

	tag, _, _ = strings.Cut(tag, "{")
	tag = strings.TrimSpace(tag)
	if tag == "" {
		tag = "{"
	}

	return tag
}

// findTextNode does a depth-first search for the first text content in nodes,
// stopping at template constructs
func findTextNode(nodes []parse.Node) *parse.TextNode {
	for _, node := range nodes {
		switch n := node.(type) {
		case *parse.TextNode:
			// skip whitespace-only text nodes
			if len(bytes.TrimSpace(n.Text)) == 0 {
				continue
			}
			return n
		case *parse.IfNode:
			if text := findTextNode(n.List.Nodes); text != nil {
				return text
			}
			if n.ElseList != nil {
				if text := findTextNode(n.ElseList.Nodes); text != nil {
					return text
				}
			}
			return nil
		case *parse.ListNode:
			if text := findTextNode(n.Nodes); text != nil {
				return text
			}
		case *parse.RangeNode:
			if text := findTextNode(n.List.Nodes); text != nil {
				return text
			}
			if n.ElseList != nil {
				if text := findTextNode(n.ElseList.Nodes); text != nil {
					return text
				}
			}
			return nil
		case *parse.WithNode:
			if text := findTextNode(n.List.Nodes); text != nil {
				return text
			}
			if n.ElseList != nil {
				if text := findTextNode(n.ElseList.Nodes); text != nil {
					return text
				}
			}
			return nil
		case *parse.ActionNode:
			return nil
		}
	}
	return nil
}
