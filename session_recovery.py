#!/usr/bin/env python3
"""
Claude Session Recovery Utility
================================
Locates, selects, extracts, and parses Claude Code session JSONL files into Markdown.

Usage:
    python3 session_recovery.py                    # interactive mode
    python3 session_recovery.py --project gpu      # filter by project name
    python3 session_recovery.py --list             # list all sessions
    python3 session_recovery.py --session <id>     # recover specific session
    python3 session_recovery.py --last <n>         # recover last N sessions
    python3 session_recovery.py --all-projects     # show all projects
"""

import os
import sys
import json
import argparse
import re
from datetime import datetime, timezone
from pathlib import Path


# ── Constants ────────────────────────────────────────────────────────────────

PROJECTS_DIR = Path.home() / ".claude" / "projects"
OUTPUT_DIR = Path.home() / ".claude" / "session-exports"
MAX_TOOL_RESULT_CHARS = 800   # truncate tool results to keep MD readable
MAX_ASSISTANT_CHARS = 6000    # truncate very long assistant blocks


# ── ANSI colours (auto-disabled if not a tty) ────────────────────────────────

def _colour(code, text):
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

BOLD  = lambda t: _colour("1", t)
DIM   = lambda t: _colour("2", t)
GREEN = lambda t: _colour("32", t)
CYAN  = lambda t: _colour("36", t)
YELLOW= lambda t: _colour("33", t)
RED   = lambda t: _colour("31", t)
BLUE  = lambda t: _colour("34", t)


# ── Project / session discovery ───────────────────────────────────────────────

def slug_to_path(slug: str) -> str:
    """
    Convert directory slug to a human-readable path.

    Claude encodes workspace paths by replacing '/' with '-', so hyphens in
    directory names are indistinguishable from path separators in the slug.
    We attempt to reconstruct by walking filesystem prefix splits until we
    find a candidate that actually exists on disk.
    """
    parts = slug.lstrip("-").split("-")
    best = "/" + "-".join(parts)   # fallback: raw slug with leading /
    for split_point in range(1, len(parts)):
        candidate = "/" + "/".join(parts[:split_point]) + "/" + "-".join(parts[split_point:])
        if os.path.exists(candidate):
            best = candidate
            break
    return best.replace(str(Path.home()), "~")


def discover_projects() -> list[dict]:
    """Return all project directories with metadata, newest-first."""
    if not PROJECTS_DIR.exists():
        return []
    projects = []
    for d in sorted(PROJECTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        sessions = list(d.glob("*.jsonl"))
        if not sessions:
            continue
        total_size = sum(s.stat().st_size for s in sessions)
        latest = max(s.stat().st_mtime for s in sessions)
        projects.append({
            "slug": d.name,
            "path": d,
            "display": slug_to_path(d.name),
            "session_count": len(sessions),
            "total_size_mb": total_size / 1024 / 1024,
            "latest_mtime": latest,
            "latest_dt": datetime.fromtimestamp(latest).strftime("%Y-%m-%d %H:%M"),
        })
    return projects


def discover_sessions(project_path: Path) -> list[dict]:
    """Return session metadata for a project directory, newest first."""
    sessions = []
    for f in sorted(project_path.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        size_mb = f.stat().st_size / 1024 / 1024
        mtime = f.stat().st_mtime
        topic = _peek_topic(f)
        sessions.append({
            "id": f.stem,
            "path": f,
            "size_mb": size_mb,
            "mtime": mtime,
            "dt": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
            "topic": topic,
        })
    return sessions


def _peek_topic(jsonl_path: Path, max_lines: int = 30) -> str:
    """Read the first user message text as a topic hint."""
    try:
        with open(jsonl_path, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                try:
                    obj = json.loads(line)
                    if obj.get("type") == "user":
                        content = obj.get("message", {}).get("content", "")
                        text = _extract_text(content)
                        if text.strip():
                            text = re.sub(r"<[^>]+>", " ", text)
                            text = " ".join(text.split())
                            return text[:120]
                except Exception:
                    continue
    except Exception:
        pass
    return "(no user messages found)"


# ── JSONL parsing & extraction ────────────────────────────────────────────────

def _extract_text(content) -> str:
    """Recursively extract text from content (str, list of blocks, or dict)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    name = block.get("name", "tool")
                    inp = block.get("input", {})
                    inp_str = json.dumps(inp, ensure_ascii=False)[:400]
                    parts.append(f"[TOOL CALL: {name}({inp_str})]")
                elif btype == "tool_result":
                    result_content = block.get("content", "")
                    result_text = _extract_text(result_content)[:MAX_TOOL_RESULT_CHARS]
                    parts.append(f"[TOOL RESULT: {result_text}]")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p.strip())
    if isinstance(content, dict):
        return _extract_text(content.get("content", ""))
    return ""


def _clean_command_tags(text: str) -> str:
    """Remove XML-style command wrapper tags from skill invocations."""
    text = re.sub(r"<command-message>.*?</command-message>", "", text, flags=re.DOTALL)
    text = re.sub(r"<command-name>(.*?)</command-name>", r"**Skill:** `\1`", text, flags=re.DOTALL)
    text = re.sub(r"<command-args>(.*?)</command-args>", r"**Args:** `\1`", text, flags=re.DOTALL)
    text = re.sub(r"<[a-z_-]+>[^<]{0,200}</[a-z_-]+>", "", text, flags=re.DOTALL)
    return text.strip()


def parse_session(jsonl_path: Path, include_tools: bool = False) -> list[dict]:
    """
    Parse a session JSONL into a list of message dicts:
      { role, text, timestamp, tool_calls, tool_results }
    """
    messages = []
    current_assistant = None

    with open(jsonl_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            record_type = obj.get("type", "")

            if record_type == "user":
                if current_assistant:
                    messages.append(current_assistant)
                    current_assistant = None

                msg = obj.get("message", {})
                content = msg.get("content", "")
                text = _extract_text(content)
                text = _clean_command_tags(text)
                ts = obj.get("timestamp", "")
                if text.strip():
                    messages.append({
                        "role": "user",
                        "text": text,
                        "timestamp": ts,
                    })

            elif record_type == "assistant":
                msg = obj.get("message", {})
                content = msg.get("content", [])
                ts = obj.get("timestamp", "")

                text_parts = []
                tool_calls = []
                for block in (content if isinstance(content, list) else []):
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use" and include_tools:
                            name = block.get("name", "tool")
                            inp = json.dumps(block.get("input", {}), ensure_ascii=False)[:300]
                            tool_calls.append(f"`{name}({inp})`")

                combined = "\n".join(t for t in text_parts if t.strip())
                if combined.strip() or tool_calls:
                    if current_assistant and not tool_calls:
                        current_assistant["text"] += "\n\n" + combined
                    else:
                        if current_assistant:
                            messages.append(current_assistant)
                        current_assistant = {
                            "role": "assistant",
                            "text": combined,
                            "timestamp": ts,
                            "tool_calls": tool_calls,
                        }

            elif record_type == "tool_result" and include_tools:
                content = obj.get("content", "")
                result_text = _extract_text(content)[:MAX_TOOL_RESULT_CHARS]
                tool_name = obj.get("tool_name", "tool")
                if current_assistant is not None:
                    current_assistant.setdefault("tool_results", []).append(
                        f"**Result of `{tool_name}`:** {result_text}"
                    )

    if current_assistant:
        messages.append(current_assistant)

    return messages


# ── Markdown rendering ────────────────────────────────────────────────────────

def render_markdown(
    session_meta: dict,
    project_meta: dict,
    messages: list[dict],
    include_tools: bool = False,
    include_timestamps: bool = True,
) -> str:
    """Render parsed messages to Markdown."""
    lines = []

    lines.append(f"# Session: `{session_meta['id'][:8]}`")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| **Project** | `{project_meta['display']}` |")
    lines.append(f"| **Session ID** | `{session_meta['id']}` |")
    lines.append(f"| **Date** | {session_meta['dt']} |")
    lines.append(f"| **Size** | {session_meta['size_mb']:.2f} MB |")
    lines.append(f"| **Messages** | {len(messages)} |")
    lines.append(f"| **Topic** | {session_meta['topic'][:100]} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    for msg in messages:
        role = msg["role"]
        text = msg.get("text", "").strip()
        ts = msg.get("timestamp", "")
        tool_calls = msg.get("tool_calls", [])
        tool_results = msg.get("tool_results", [])

        header = "## 💬 User" if role == "user" else "## 🤖 Assistant"

        if include_timestamps and ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                dt_local = dt.astimezone()
                header += f" _{dt_local.strftime('%H:%M:%S')}_"
            except Exception:
                pass

        lines.append(header)
        lines.append("")

        if text:
            if role == "assistant" and len(text) > MAX_ASSISTANT_CHARS:
                text = text[:MAX_ASSISTANT_CHARS] + f"\n\n> *(truncated — {len(msg.get('text',''))} chars total)*"
            lines.append(text)
            lines.append("")

        if include_tools and tool_calls:
            lines.append("<details>")
            lines.append("<summary>🔧 Tool Calls</summary>")
            lines.append("")
            for tc in tool_calls:
                lines.append(f"- {tc}")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        if include_tools and tool_results:
            lines.append("<details>")
            lines.append("<summary>📤 Tool Results</summary>")
            lines.append("")
            for tr in tool_results:
                lines.append(tr)
                lines.append("")
            lines.append("</details>")
            lines.append("")

        lines.append("---")
        lines.append("")

    lines.append(f"*Exported by Claude Session Recovery Utility — {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    return "\n".join(lines)


# ── Interactive UI ────────────────────────────────────────────────────────────

def print_projects(projects: list[dict], filter_str: str = ""):
    print(f"\n{BOLD('Claude Code Projects')}\n")
    filtered = [p for p in projects if not filter_str or filter_str.lower() in p["display"].lower()]
    if not filtered:
        print(RED(f"  No projects matching '{filter_str}'"))
        return []
    for i, p in enumerate(filtered):
        size_str = f"{p['total_size_mb']:.1f}MB"
        print(f"  {CYAN(str(i+1)):>4}.  {GREEN(p['display']):<60} "
              f"{DIM(p['latest_dt'])}  {DIM(size_str):>8}  {p['session_count']} sessions")
    return filtered


def print_sessions(sessions: list[dict]):
    print(f"\n{BOLD('Sessions (newest first)')}\n")
    for i, s in enumerate(sessions):
        size_str = f"{s['size_mb']:.2f}MB"
        flag = YELLOW(" ⚠ LARGE") if s["size_mb"] > 5 else ""
        print(f"  {CYAN(str(i+1)):>4}.  {DIM(s['id'][:8])}  {s['dt']}  {size_str:>8}{flag}")
        print(f"        {DIM(s['topic'][:100])}")
    print()


def choose(prompt: str, options: list, allow_multi: bool = False):
    """Simple numbered choice prompt."""
    while True:
        raw = input(f"{BOLD(prompt)} ").strip()
        if not raw:
            return []
        if raw.lower() == "a":
            return list(range(len(options)))
        parts = raw.replace(",", " ").split()
        indices = []
        valid = True
        for p in parts:
            try:
                n = int(p)
                if 1 <= n <= len(options):
                    indices.append(n - 1)
                else:
                    print(RED(f"  '{n}' is out of range (1-{len(options)})"))
                    valid = False
                    break
            except ValueError:
                print(RED(f"  '{p}' is not a number"))
                valid = False
                break
        if valid and indices:
            return indices if allow_multi else [indices[0]]


def interactive_mode(filter_str: str = ""):
    """Full interactive session recovery flow."""
    projects = discover_projects()
    if not projects:
        print(RED("No Claude project directories found."))
        sys.exit(1)

    filtered = print_projects(projects, filter_str)
    if not filtered:
        sys.exit(1)

    indices = choose("Select project (number, or 'a' for all):", filtered)
    if not indices:
        print("No project selected. Exiting.")
        sys.exit(0)

    selected_project = filtered[indices[0]]
    print(f"\n{BOLD('Project:')} {GREEN(selected_project['display'])}\n")

    sessions = discover_sessions(selected_project["path"])
    if not sessions:
        print(RED("No sessions found."))
        sys.exit(1)

    print_sessions(sessions)

    indices = choose(
        f"Select sessions (1-{len(sessions)}, comma-separated, or 'a' for all):",
        sessions,
        allow_multi=True,
    )
    if not indices:
        print("No sessions selected. Exiting.")
        sys.exit(0)

    selected_sessions = [sessions[i] for i in indices]

    print()
    include_tools = input(f"{BOLD('Include tool calls/results? [y/N]:')} ").strip().lower() == "y"
    include_timestamps = input(f"{BOLD('Include timestamps? [Y/n]:')} ").strip().lower() != "n"
    output_dir_str = input(f"{BOLD('Output directory')} [{OUTPUT_DIR}]: ").strip()
    output_dir = Path(output_dir_str) if output_dir_str else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    export_paths = []
    for s in selected_sessions:
        print(f"  {CYAN('↳')} Parsing {s['id'][:8]}  ({s['size_mb']:.2f} MB)...")
        messages = parse_session(s["path"], include_tools=include_tools)
        md = render_markdown(s, selected_project, messages,
                             include_tools=include_tools,
                             include_timestamps=include_timestamps)
        fname = (f"{selected_project['slug'][:40]}_{s['id'][:8]}"
                 f"_{s['dt'].replace(' ', 'T').replace(':', '')}.md")
        out_path = output_dir / fname
        out_path.write_text(md, encoding="utf-8")
        export_paths.append(out_path)
        print(f"     {GREEN('✓')} Exported {len(messages)} messages → {out_path}")

    print(f"\n{BOLD('Done!')} {len(export_paths)} file(s) written to {output_dir}\n")
    return export_paths


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Claude Session Recovery Utility — extract conversations to Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--project", "-p", help="Filter projects by name substring")
    parser.add_argument("--list",    "-l", action="store_true", help="List all projects and sessions")
    parser.add_argument("--all-projects", "-A", action="store_true", help="Show all projects")
    parser.add_argument("--session", "-s", help="Extract a specific session ID (prefix ok)")
    parser.add_argument("--last",    "-n", type=int, help="Extract last N sessions from matched project")
    parser.add_argument("--output",  "-o", help="Output directory", default=str(OUTPUT_DIR))
    parser.add_argument("--include-tools", "-t", action="store_true", help="Include tool calls/results")
    parser.add_argument("--no-timestamps",        action="store_true", help="Omit timestamps")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    include_tools      = args.include_tools
    include_timestamps = not args.no_timestamps

    if args.list or args.all_projects:
        projects = discover_projects()
        filtered = print_projects(projects, args.project or "")
        if args.list:
            for p in (filtered or projects):
                sessions = discover_sessions(p["path"])
                print_sessions(sessions)
        return

    if args.session:
        projects = discover_projects()
        found = False
        for proj in projects:
            for s in discover_sessions(proj["path"]):
                if s["id"].startswith(args.session):
                    print(f"\nFound: {s['path']}")
                    messages = parse_session(s["path"], include_tools)
                    md = render_markdown(s, proj, messages, include_tools, include_timestamps)
                    fname = f"{proj['slug'][:40]}_{s['id'][:8]}.md"
                    out = output_dir / fname
                    out.write_text(md, encoding="utf-8")
                    print(GREEN(f"✓ Exported {len(messages)} messages → {out}"))
                    found = True
        if not found:
            print(RED(f"Session '{args.session}' not found."))
        return

    if args.last:
        projects = discover_projects()
        filtered = [p for p in projects
                    if not args.project or args.project.lower() in p["display"].lower()]
        if not filtered:
            print(RED("No matching projects."))
            return
        proj = filtered[0]
        sessions = discover_sessions(proj["path"])[: args.last]
        for s in sessions:
            messages = parse_session(s["path"], include_tools)
            md = render_markdown(s, proj, messages, include_tools, include_timestamps)
            fname = (f"{proj['slug'][:40]}_{s['id'][:8]}"
                     f"_{s['dt'].replace(' ', 'T').replace(':', '')}.md")
            out = output_dir / fname
            out.write_text(md, encoding="utf-8")
            print(GREEN(f"✓ {s['id'][:8]}  {s['dt']}  ({s['size_mb']:.2f}MB)  → {out}"))
        return

    # Default: interactive
    interactive_mode(filter_str=args.project or "")


if __name__ == "__main__":
    main()
