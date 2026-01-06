import re


def ansi_clean(cmd: str) -> str:
    """
    Wrap a shell command so its stdout+stderr are filtered through sed to strip ANSI escape codes.

    - cmd: full shell command (may or may not already have redirection like '>> file 2>&1')

    Returns: modified command string that is portable under /bin/sh (no bash features).
    """

    # Portable ESC declaration (POSIX sh): put this in front of the command
    esc_decl = "ESC=$(printf '\\033'); "

    # Portable, locale-stable sed filter (ERE via -E; works on GNU & BSD sed)
    # Use double quotes so ${ESC} expands; escape the '[' inside the Python string.
    sed_pipe = 'LC_ALL=C sed -E "s/${ESC}\\[[0-9;]*[[:alpha:]]//g"'

    # If the command already redirects to a file at the end, restructure to:
    #   <left> 2>&1 | sed ... >> <file>
    m = re.search(r"(.*?)(\s*)(>>?|)(\s*)([^>|]*?)(\s*)(2>&1)?\s*$", cmd)
    # Groups:
    # 1: left side (the command proper)
    # 3: redir operator '' or '>' or '>>'
    # 5: redir target (filename) (best-effort; may be empty)
    # 7: trailing '2>&1' if present

    if m and m.group(3):  # has '>' or '>>'
        left = m.group(1).rstrip()
        redir = m.group(3)
        target = (m.group(5) or "").strip()
        target_part = f" {target}" if target else ""
        # Join stderr before piping so both streams get cleaned
        return f"{esc_decl}{left} 2>&1 | {sed_pipe} {redir}{target_part}".strip()
    else:
        # No terminal redirection: pipe cleaned output to stdout
        return f"{esc_decl}{cmd} 2>&1 | {sed_pipe}"


# def remove_ansi_escape_sequences(command: str) -> str:
#     """uses sed to remove ansi escape sequences from a command"""
#     return os.popen(f"sed 's/\x1b\[[0-9;]*m//g'").read()


# def ansi_clean(cmd: str, use_bash_ansi_c: bool = True) -> str:
#     """
#     Wrap a shell command so its stdout+stderr are filtered through sed to strip ANSI escape codes.

#     - cmd: full shell command (may or may not already have redirection like '>> file 2>&1')
#     - clean: if False, returns cmd unchanged
#     - use_bash_ansi_c: if True, uses Bash ANSI-C quoting ($'...') for portable ESC.
#                        If your environment isn't bash, set this to False.

#     Returns: modified command string.
#     """
#     import re

#     # sed expression: remove ESC [ ... letters (cursor/formatting controls)
#     if use_bash_ansi_c:
#         # Requires bash (ANSI-C quoting for \x1B).
#         sed_expr = r"$'s/\x1B\[[0-9;]*[[:alpha:]]//g'"
#     else:
#         # Works on many GNU/BSD seds that accept \033 inside single quotes.
#         sed_expr = r"'s/\033\[[0-9;]*[[:alpha:]]//g'"

#     sed_pipe = f"sed -E {sed_expr}"

#     # If the command already redirects to a file, we want:
#     #   <left> 2>&1 | sed ... >> <file>
#     # i.e., move the 2>&1 before the pipe and keep the same target file.
#     m = re.search(r"(.*?)(\s*)(>>?|)(\s*)([^>|]*?)(\s*)(2>&1)?\s*$", cmd)
#     # This regex tries to capture the final redirection if present:
#     # 1: left part (command proper)
#     # 3: redir operator '' or '>' or '>>'
#     # 5: redir target (filename) (may contain spaces; best-effort)
#     # 7: '2>&1' if present at the end

#     if m and m.group(3):  # has '>' or '>>'
#         left = m.group(1).rstrip()
#         redir = m.group(3)
#         target = m.group(5).strip()
#         # Ensure we don't duplicate an empty filename
#         target_part = f" {target}" if target else ""
#         return f"{left} 2>&1 | {sed_pipe} {redir}{target_part}".strip()
#     else:
#         # No explicit file redirection at the end; just pipe and keep existing behavior.
#         # If the original cmd already had its own inner pipes, this simply appends our filter.
#         # Ensure stderr joins stdout so both are cleaned.
#         return f"{cmd} 2>&1 | {sed_pipe}"
