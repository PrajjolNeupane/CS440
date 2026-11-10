#!/usr/bin/env bash
# Context switcher: context engineering strategy 1 ("dynamic imports").
#
# Runs as a UserPromptSubmit hook. Claude Code sends the prompt as JSON on stdin, and
# anything this script prints is added to Claude's context for that prompt.
# It picks files from .claude/context/ based on the git branch and the prompt text,
# and loads each file at most once per session so the same text does not pile up.

project_directory="${CLAUDE_PROJECT_DIR:-$(pwd)}"
context_directory="$project_directory/.claude/context"

hook_input=$(cat)
prompt_text=$(printf '%s' "$hook_input" | jq -r '.prompt // ""' 2>/dev/null)
session_id=$(printf '%s' "$hook_input" | jq -r '.session_id // "no-session"' 2>/dev/null)
branch_name=$(git -C "$project_directory" branch --show-current 2>/dev/null)

# One list per session of the context files that were already loaded
loaded_list_directory="${TMPDIR:-/tmp}/cs440-context-switcher"
mkdir -p "$loaded_list_directory"
loaded_list="$loaded_list_directory/${session_id:-no-session}"
touch "$loaded_list"

load_context() {
    local context_name="$1"
    local context_file="$context_directory/$context_name.md"
    [ -f "$context_file" ] || return 0
    # Idempotency check: skip a file this session already loaded
    grep -qxF "$context_name" "$loaded_list" && return 0
    echo "$context_name" >> "$loaded_list"
    printf '\n<!-- context-switcher loaded .claude/context/%s.md -->\n' "$context_name"
    cat "$context_file"
}

prompt_mentions() {
    printf '%s' "$prompt_text" | grep -Eiq "$1"
}

# 1. Context chosen by git branch
case "$branch_name" in
    qlearning*|lab5*) load_context qlearning ;;
esac

# 2. Context chosen by words in the prompt
if prompt_mentions 'q-?learning|q-?table|q-?value|epsilon|learning rate|discount|gamma|living reward|\<policy|episode|hyperparameter'; then
    load_context qlearning
fi
if prompt_mentions '\<tests?\>|unittest|testing'; then
    load_context testing
fi

exit 0
