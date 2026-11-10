#!/usr/bin/env bash
# Test runner hook (Lab 5).
#
# Runs as a PostToolUse hook after Claude uses Write or Edit. Claude Code sends the tool call as
# JSON on stdin. If the edited file is Python code under qlearning/ or tests/, this runs the
# unittest suite:
# - tests pass: a one-line note is shown to the user
# - tests fail: the failures are sent back to Claude ("decision": "block"), so it sees them in the
#   main conversation right away and can fix the code. Nothing is undone; the edit stays.
# Any other file is ignored, so editing notes or settings costs nothing.

project_directory="${CLAUDE_PROJECT_DIR:-$(pwd)}"
failure_line_limit=60

hook_input=$(cat)
edited_file=$(printf '%s' "$hook_input" | jq -r '.tool_input.file_path // .tool_response.filePath // ""' 2>/dev/null)

# Only Python files in qlearning/ or tests/ (including subfolders)
case "$edited_file" in
    "$project_directory"/qlearning/*.py|"$project_directory"/tests/*.py) ;;
    *) exit 0 ;;
esac
[ -d "$project_directory/tests" ] || exit 0

relative_path="${edited_file#"$project_directory"/}"
test_output=$(cd "$project_directory" && python3 -m unittest discover -s tests 2>&1)
test_status=$?

# unittest ends with "Ran N tests in Xs" and then "OK" or "FAILED (failures=1, errors=2)"
tests_ran_line=$(printf '%s\n' "$test_output" | grep -E '^Ran [0-9]+ tests?' | tail -n 1)
result_line=$(printf '%s\n' "$test_output" | grep -E '^(OK|FAILED)' | tail -n 1)

if [ "$test_status" -eq 0 ]; then
    jq -n --arg message "Tests passed after editing $relative_path: $tests_ran_line, $result_line" \
        '{systemMessage: $message}'
    exit 0
fi

failure_details=$(printf '%s\n' "$test_output" | tail -n "$failure_line_limit")
jq -n \
    --arg reason "Unit tests failed after editing $relative_path. Fix the code (or the test, if the test is wrong) before moving on.
Last $failure_line_limit lines of: python3 -m unittest discover -s tests

$failure_details" \
    --arg message "Tests FAILED after editing $relative_path: ${result_line:-see output}" \
    '{decision: "block", reason: $reason, systemMessage: $message}'
exit 0
