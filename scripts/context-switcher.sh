CLAUDE_MD="Claude.md"

touch "$CLAUDE_MD"

add_context() {
    local context_ref="$1"
    grep -qxf "$context_ref" "$CLAUDE_MD" || echo "$context_ref" >> "$CLAUDE_MD"
}

branch=$(git branch --show-current 2>/dev/null)

case $branch in:
    
esac