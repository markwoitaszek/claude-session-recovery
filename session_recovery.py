#!/usr/bin/env python3
"""
Claude Session Recovery Utility
================================
Locates, selects, extracts, and parses Claude Code session JSONL files into Markdown.

Usage:
    python3 session_recovery.py                    # interactive hierarchy + selection
    python3 session_recovery.py --tree             # display full project/session hierarchy
    python3 session_recovery.py --project gpu      # filter by project name
    python3 session_recovery.py --list             # list all sessions (compact)
    python3 session_recovery.py --session <id>     # recover specific session
    python3 session_recovery.py --last <n>         # recover last N sessions
    python3 session_recovery.py --all-projects     # show all projects
"""

import os
import sys
import json
import argparse
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path


# ── Constants ────────────────────────────────────────────────────────────────

PROJECTS_DIR = Path.home() / ".claude" / "projects"
OUTPUT_DIR   = Path.home() / ".claude" / "session-exports"
MAX_TOOL_RESULT_CHARS = 800    # truncate tool results to keep MD readable
MAX_ASSISTANT_CHARS   = 6000   # truncate very long assistant blocks

# A session is flagged as crash-risk when it exceeds this size AND contains skill XML tags
CRASH_SIZE_THRESHOLD_MB = 2.0
LARGE_SIZE_THRESHOLD_MB = 5.0


# ── ANSI colours (auto-disabled when stdout is not a tty) ────────────────────

def _colour(code: str, text: str) -> str:
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

BOLD    = lambda t: _colour("1",  t)
DIM     = lambda t: _colour("2",  t)
GREEN   = lambda t: _colour("32", t)
CYAN    = lambda t: _colour("36", t)
YELLOW  = lambda t: _colour("33", t)
RED     = lambda t: _colour("31", t)
BLUE    = lambda t: _colour("34", t)
MAGENTA = lambda t: _colour("35", t)


# ── Project / session discovery ───────────────────────────────────────────────

def slug_to_path(slug: str) -> str:
    """
    Convert a Claude Code directory slug back to a human-readable workspace path.

    Claude encodes workspace paths by replacing every '/' with '-', which makes
    hyphenated directory names (like 'gpu-photo-pipeline') ambiguous in the slug.
    We walk prefix splits and check os.path.exists() to find the longest real path.
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
    """Return all Claude project directories with metadata, sorted newest-first."""
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
            "slug":          d.name,
            "path":          d,
            "display":       slug_to_path(d.name),
            "session_count": len(sessions),
            "total_size_mb": total_size / 1024 / 1024,
            "latest_mtime":  latest,
            "latest_dt":     datetime.fromtimestamp(latest).strftime("%Y-%m-%d %H:%M"),
        })
    return projects


def discover_sessions(project_path: Path) -> list[dict]:
    """Return session metadata for a project directory, sorted newest-first."""
    sessions = []
    for f in sorted(project_path.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        size_mb = f.stat().st_size / 1024 / 1024
        mtime   = f.stat().st_mtime
        topic   = _peek_topic(f)
        sessions.append({
            "id":      f.stem,
            "path":    f,
            "size_mb": size_mb,
            "mtime":   mtime,
            "dt":      datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
            "topic":   topic,
        })
    return sessions


def _peek_topic(jsonl_path: Path, max_lines: int = 30) -> str:
    """Read the first user message text as a topic hint for display."""
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


def get_project_spec(workspace_display: str) -> str | None:
    """
    Find and return the first meaningful line from CLAUDE.md in the workspace.

    Claude Code stores project memory / specifications in CLAUDE.md files:
      - <workspace>/CLAUDE.md        project-level spec
      - <workspace>/CLAUDE.local.md  local overrides (not committed)

    This function returns the first non-heading, non-empty line as a one-line
    description, or None if no spec file is found.
    """
    full = workspace_display.replace("~", str(Path.home()))
    for candidate in [
        Path(full) / "CLAUDE.md",
        Path(full) / "CLAUDE.local.md",
    ]:
        if candidate.exists():
            result = _read_first_spec_line(candidate)
            if result:
                return result
    return None


def _read_first_spec_line(path: Path) -> str | None:
    """Return the first meaningful content line from a CLAUDE.md file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        # Priority 1: first non-empty, non-heading paragraph line
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and \
               not stripped.startswith("<!--") and not stripped.startswith("!"):
                return stripped[:120]
        # Priority 2: first heading text (strip the leading #s)
        for line in text.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                return stripped[:120]
    except Exception:
        pass
    return None


def session_crash_risk(s: dict) -> str:
    """
    Estimate crash risk for a session.

    The React hydration bug (error #418) is triggered by XML-like skill tags
    in sessions larger than ~2 MB. We scan the first ~60 lines for the known
    tag patterns to confirm the presence of the injection vector.

    Returns: 'CRASH', 'LARGE', or '' (likely safe).
    """
    if s["size_mb"] > LARGE_SIZE_THRESHOLD_MB:
        return "CRASH"
    if s["size_mb"] > CRASH_SIZE_THRESHOLD_MB:
        try:
            xml_count = 0
            with open(s["path"], encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if i > 60:
                        break
                    if re.search(
                        r"<command-message>|<ide_opened_file>|<command-args>|<command-name>",
                        line,
                    ):
                        xml_count += 1
                        if xml_count >= 2:
                            return "CRASH"
        except Exception:
            pass
        return "LARGE" if s["size_mb"] > 3.0 else ""
    return ""


def group_sessions_by_date(sessions: list[dict]) -> list[tuple[str, list[dict]]]:
    """
    Group sessions into human-readable recency buckets.

    Returns an ordered list of (label, sessions) tuples preserving
    newest-first ordering within each group.
    """
    today = datetime.now().date()
    ordered_keys: list[str] = []
    groups: dict[str, list] = {}

    for s in sessions:
        d    = datetime.fromtimestamp(s["mtime"]).date()
        days = (today - d).days
        if days == 0:
            key = "Today"
        elif days == 1:
            key = "Yesterday"
        elif days < 7:
            key = f"This Week  ({d.strftime('%A')})"
        elif days < 30:
            key = d.strftime("%-d %b")
        else:
            key = d.strftime("%B %Y")

        if key not in groups:
            groups[key] = []
            ordered_keys.append(key)
        groups[key].append(s)

    return [(k, groups[k]) for k in ordered_keys]


# ── JSONL parsing & extraction ────────────────────────────────────────────────

def _extract_text(content) -> str:
    """Recursively extract plain text from content (str, list of blocks, or dict)."""
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
                    name    = block.get("name", "tool")
                    inp_str = json.dumps(block.get("input", {}), ensure_ascii=False)[:400]
                    parts.append(f"[TOOL CALL: {name}({inp_str})]")
                elif btype == "tool_result":
                    result_text = _extract_text(block.get("content", ""))[:MAX_TOOL_RESULT_CHARS]
                    parts.append(f"[TOOL RESULT: {result_text}]")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p.strip())
    if isinstance(content, dict):
        return _extract_text(content.get("content", ""))
    return ""


def _clean_command_tags(text: str) -> str:
    """Remove XML-style skill invocation wrapper tags from user message text."""
    text = re.sub(r"<command-message>.*?</command-message>", "", text, flags=re.DOTALL)
    text = re.sub(r"<command-name>(.*?)</command-name>",     r"**Skill:** `\1`", text, flags=re.DOTALL)
    text = re.sub(r"<command-args>(.*?)</command-args>",     r"**Args:** `\1`",  text, flags=re.DOTALL)
    text = re.sub(r"<[a-z_-]+>[^<]{0,200}</[a-z_-]+>", "", text, flags=re.DOTALL)
    return text.strip()


def parse_session(jsonl_path: Path, include_tools: bool = False) -> list[dict]:
    """
    Parse a session JSONL into a list of message dicts:
      { role, text, timestamp, tool_calls?, tool_results? }
    """
    messages: list[dict]  = []
    current_assistant: dict | None = None

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

                msg     = obj.get("message", {})
                content = msg.get("content", "")
                text    = _clean_command_tags(_extract_text(content))
                ts      = obj.get("timestamp", "")
                if text.strip():
                    messages.append({"role": "user", "text": text, "timestamp": ts})

            elif record_type == "assistant":
                msg     = obj.get("message", {})
                content = msg.get("content", [])
                ts      = obj.get("timestamp", "")

                text_parts: list[str] = []
                tool_calls: list[str] = []
                for block in (content if isinstance(content, list) else []):
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use" and include_tools:
                            name    = block.get("name", "tool")
                            inp_str = json.dumps(block.get("input", {}), ensure_ascii=False)[:300]
                            tool_calls.append(f"`{name}({inp_str})`")

                combined = "\n".join(t for t in text_parts if t.strip())
                if combined.strip() or tool_calls:
                    if current_assistant and not tool_calls:
                        current_assistant["text"] += "\n\n" + combined
                    else:
                        if current_assistant:
                            messages.append(current_assistant)
                        current_assistant = {
                            "role":       "assistant",
                            "text":       combined,
                            "timestamp":  ts,
                            "tool_calls": tool_calls,
                        }

            elif record_type == "tool_result" and include_tools:
                content     = obj.get("content", "")
                result_text = _extract_text(content)[:MAX_TOOL_RESULT_CHARS]
                tool_name   = obj.get("tool_name", "tool")
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
    messages:     list[dict],
    include_tools:      bool = False,
    include_timestamps: bool = True,
) -> str:
    """Render parsed messages as a Markdown document."""
    lines: list[str] = []

    lines += [
        f"# Session: `{session_meta['id'][:8]}`", "",
        "| Field | Value |",
        "|-------|-------|",
        f"| **Project** | `{project_meta['display']}` |",
        f"| **Session ID** | `{session_meta['id']}` |",
        f"| **Date** | {session_meta['dt']} |",
        f"| **Size** | {session_meta['size_mb']:.2f} MB |",
        f"| **Messages** | {len(messages)} |",
        f"| **Topic** | {session_meta['topic'][:100]} |",
        "", "---", "",
    ]

    for msg in messages:
        role        = msg["role"]
        text        = msg.get("text", "").strip()
        ts          = msg.get("timestamp", "")
        tool_calls  = msg.get("tool_calls",  [])
        tool_results= msg.get("tool_results", [])

        header = "## 💬 User" if role == "user" else "## 🤖 Assistant"
        if include_timestamps and ts:
            try:
                dt_local = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone()
                header  += f" _{dt_local.strftime('%H:%M:%S')}_"
            except Exception:
                pass

        lines += [header, ""]

        if text:
            if role == "assistant" and len(text) > MAX_ASSISTANT_CHARS:
                text = (text[:MAX_ASSISTANT_CHARS] +
                        f"\n\n> *(truncated — {len(msg.get('text',''))} chars total)*")
            lines += [text, ""]

        if include_tools and tool_calls:
            lines += ["<details>", "<summary>🔧 Tool Calls</summary>", ""]
            lines += [f"- {tc}" for tc in tool_calls]
            lines += ["", "</details>", ""]

        if include_tools and tool_results:
            lines += ["<details>", "<summary>📤 Tool Results</summary>", ""]
            for tr in tool_results:
                lines += [tr, ""]
            lines += ["</details>", ""]

        lines += ["---", ""]

    lines.append(
        f"*Exported by Claude Session Recovery Utility — {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
    )
    return "\n".join(lines)


# ── Hierarchy display ─────────────────────────────────────────────────────────

def show_hierarchy(
    projects:   list[dict],
    filter_str: str = "",
) -> list[tuple[dict, dict]]:
    """
    Display the full Claude Code project/specification/session hierarchy as a tree.

    For each project the display shows:
      📁 <workspace path>
         │  spec: <first line of CLAUDE.md, if present>
         │  N session(s)  ·  X MB total  ·  last active …
         │
         ├─ Today
         │    ├─ [1] <id>  <date>  <size>  ⚠ CRASH RISK
         │    │       <topic hint>
         │    └─ [2] <id>  <date>  <size>
         │           <topic hint>
         └─ Yesterday
              └─ [3] <id>  <date>  <size>

    Returns a flat list of (project_meta, session_meta) tuples in display order,
    indexed by the [N] numbers shown — so the caller can accept a number like "3"
    and directly recover that session without needing a two-step project→session flow.
    """
    filtered = [
        p for p in projects
        if not filter_str or filter_str.lower() in p["display"].lower()
    ]
    if not filtered:
        print(RED(f"  No projects matching '{filter_str}'"))
        return []

    print(f"\n{BOLD('Claude Code  ·  Project & Session Hierarchy')}")
    print(DIM("  " + "━" * 72))

    all_sessions: list[tuple[dict, dict]] = []
    counter = 0

    for p in filtered:
        sessions    = discover_sessions(p["path"])
        spec        = get_project_spec(p["display"])
        size_str    = f"{p['total_size_mb']:.1f} MB total"

        print()
        print(f"  {BLUE('📁')} {BOLD(GREEN(p['display']))}")
        if spec:
            print(f"     {DIM('│')}  {CYAN('spec:')} {DIM(spec)}")
        print(
            f"     {DIM('│')}  {p['session_count']} session(s)  ·  {size_str}"
            f"  ·  last active {p['latest_dt']}"
        )

        if sessions:
            date_groups = group_sessions_by_date(sessions)
            for gi, (date_label, group_sessions) in enumerate(date_groups):
                is_last_group = gi == len(date_groups) - 1
                group_branch  = "└─" if is_last_group else "├─"

                print(f"     {DIM('│')}")
                print(f"     {DIM('├─')} {MAGENTA(date_label)}")

                for si, s in enumerate(group_sessions):
                    counter += 1
                    all_sessions.append((p, s))

                    is_last_sess = si == len(group_sessions) - 1
                    branch       = "└─" if is_last_sess else "├─"

                    risk = session_crash_risk(s)
                    if risk == "CRASH":
                        risk_tag = f"  {RED('⚠ CRASH RISK')}"
                    elif risk == "LARGE":
                        risk_tag = f"  {YELLOW('◈ LARGE')}"
                    else:
                        risk_tag = ""

                    num_str   = CYAN(f"[{counter:2}]")
                    id_str    = DIM(s["id"][:8])
                    size_s    = f"{s['size_mb']:5.1f} MB"
                    topic_raw = s["topic"]
                    topic_s   = (
                        DIM("(no content)")
                        if topic_raw == "(no user messages found)"
                        else DIM(topic_raw[:70])
                    )

                    print(f"     {DIM('│')}    {DIM(branch)} {num_str} {id_str}  {s['dt']}  {size_s}{risk_tag}")
                    print(f"     {DIM('│')}          {topic_s}")

        print(f"     {DIM('─' * 70)}")

    print()
    return all_sessions


# ── Compact list views (used by --all-projects / --list) ─────────────────────

def print_projects(projects: list[dict], filter_str: str = "") -> list[dict]:
    """Compact one-line-per-project listing."""
    print(f"\n{BOLD('Claude Code Projects')}\n")
    filtered = [p for p in projects if not filter_str or filter_str.lower() in p["display"].lower()]
    if not filtered:
        print(RED(f"  No projects matching '{filter_str}'"))
        return []
    for i, p in enumerate(filtered):
        size_str = f"{p['total_size_mb']:.1f}MB"
        print(
            f"  {CYAN(str(i + 1)):>4}.  {GREEN(p['display']):<60} "
            f"{DIM(p['latest_dt'])}  {DIM(size_str):>8}  {p['session_count']} sessions"
        )
    print()
    return filtered


def print_sessions(sessions: list[dict]):
    """Compact session list with date grouping."""
    print(f"\n{BOLD('Sessions (newest first)')}\n")
    date_groups = group_sessions_by_date(sessions)
    counter = 0
    for date_label, group in date_groups:
        print(f"  {MAGENTA(date_label)}")
        for s in group:
            counter += 1
            size_str = f"{s['size_mb']:.2f}MB"
            risk = session_crash_risk(s)
            flag = (RED("  ⚠ CRASH RISK") if risk == "CRASH"
                    else YELLOW("  ◈ LARGE") if risk == "LARGE"
                    else "")
            print(f"    {CYAN(f'[{counter}]'):>6}  {DIM(s['id'][:8])}  {s['dt']}  {size_str:>8}{flag}")
            print(f"           {DIM(s['topic'][:100])}")
        print()


# ── Selection parsing ─────────────────────────────────────────────────────────

def parse_selection(raw: str, total: int) -> list[int]:
    """
    Parse a user selection string into 0-based indices.

    Supported formats:
      3         single item
      1,3,5     comma-separated
      1-4       inclusive range
      a         all items
    """
    raw = raw.strip()
    if raw.lower() == "a":
        return list(range(total))

    indices: list[int] = []
    for token in re.split(r"[,\s]+", raw):
        token = token.strip()
        if not token:
            continue
        range_match = re.fullmatch(r"(\d+)-(\d+)", token)
        if range_match:
            lo, hi = int(range_match.group(1)), int(range_match.group(2))
            for n in range(lo, hi + 1):
                if 1 <= n <= total:
                    indices.append(n - 1)
                else:
                    raise ValueError(f"'{n}' is out of range (1–{total})")
        else:
            n = int(token)
            if 1 <= n <= total:
                indices.append(n - 1)
            else:
                raise ValueError(f"'{n}' is out of range (1–{total})")

    # Deduplicate while preserving order
    seen: set[int] = set()
    return [i for i in indices if not (i in seen or seen.add(i))]  # type: ignore[func-returns-value]


# ── Interactive mode ──────────────────────────────────────────────────────────

def interactive_mode(filter_str: str = ""):
    """
    Full interactive session recovery flow.

    1. Shows the complete project/spec/session hierarchy tree.
    2. Accepts direct global session numbers (e.g. "3", "1,4", "2-5", "a").
    3. Asks for export options.
    4. Writes Markdown files.
    """
    projects = discover_projects()
    if not projects:
        print(RED("No Claude project directories found."))
        sys.exit(1)

    all_sessions = show_hierarchy(projects, filter_str)
    if not all_sessions:
        sys.exit(1)

    total = len(all_sessions)
    print(BOLD("Select sessions to recover:"))
    print(DIM(f"  Enter numbers 1–{total}, comma-separated (1,3), ranges (2-5), or 'a' for all\n"))

    while True:
        try:
            raw = input(f"  {BOLD('→')} ").strip()
            if not raw:
                print("No selection. Exiting.")
                sys.exit(0)
            indices = parse_selection(raw, total)
            if indices:
                break
            print(RED("  No valid selections. Try again."))
        except ValueError as e:
            print(RED(f"  {e}. Try again."))
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

    selected = [all_sessions[i] for i in indices]

    print()
    include_tools      = input(f"{BOLD('Include tool calls/results? [y/N]:')} ").strip().lower() == "y"
    include_timestamps = input(f"{BOLD('Include timestamps? [Y/n]:')} ").strip().lower() != "n"
    output_dir_str     = input(f"{BOLD('Output directory')} [{OUTPUT_DIR}]: ").strip()
    output_dir         = Path(output_dir_str) if output_dir_str else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    export_paths: list[Path] = []
    for proj, s in selected:
        print(f"  {CYAN('↳')} Parsing {s['id'][:8]}  ({s['size_mb']:.2f} MB)...")
        messages = parse_session(s["path"], include_tools=include_tools)
        md = render_markdown(
            s, proj, messages,
            include_tools=include_tools,
            include_timestamps=include_timestamps,
        )
        fname    = (f"{proj['slug'][:40]}_{s['id'][:8]}"
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
    parser.add_argument("--project",       "-p", help="Filter projects by name substring")
    parser.add_argument("--tree",          "-T", action="store_true",
                        help="Display full project/specification/session hierarchy and exit")
    parser.add_argument("--list",          "-l", action="store_true",
                        help="List all projects and their sessions (compact)")
    parser.add_argument("--all-projects",  "-A", action="store_true",
                        help="Show all projects (compact, no session detail)")
    parser.add_argument("--session",       "-s", help="Extract a specific session by ID prefix")
    parser.add_argument("--last",          "-n", type=int,
                        help="Extract last N sessions from matched project")
    parser.add_argument("--output",        "-o", help="Output directory",
                        default=str(OUTPUT_DIR))
    parser.add_argument("--include-tools", "-t", action="store_true",
                        help="Expand tool calls and results in output")
    parser.add_argument("--no-timestamps",        action="store_true",
                        help="Omit per-message timestamps")
    args = parser.parse_args()

    output_dir         = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    include_tools      = args.include_tools
    include_timestamps = not args.no_timestamps

    # ── --tree ─────────────────────────────────────────────────────────────
    if args.tree:
        projects = discover_projects()
        show_hierarchy(projects, args.project or "")
        return

    # ── --all-projects / --list ────────────────────────────────────────────
    if args.list or args.all_projects:
        projects = discover_projects()
        filtered = print_projects(projects, args.project or "")
        if args.list:
            for p in (filtered or projects):
                sessions = discover_sessions(p["path"])
                print_sessions(sessions)
        return

    # ── --session <id> ─────────────────────────────────────────────────────
    if args.session:
        projects = discover_projects()
        found    = False
        for proj in projects:
            for s in discover_sessions(proj["path"]):
                if s["id"].startswith(args.session):
                    print(f"\nFound: {s['path']}")
                    messages = parse_session(s["path"], include_tools)
                    md       = render_markdown(s, proj, messages, include_tools, include_timestamps)
                    fname    = f"{proj['slug'][:40]}_{s['id'][:8]}.md"
                    out      = output_dir / fname
                    out.write_text(md, encoding="utf-8")
                    print(GREEN(f"✓ Exported {len(messages)} messages → {out}"))
                    found = True
        if not found:
            print(RED(f"Session '{args.session}' not found."))
        return

    # ── --last N ───────────────────────────────────────────────────────────
    if args.last:
        projects = discover_projects()
        filtered = [
            p for p in projects
            if not args.project or args.project.lower() in p["display"].lower()
        ]
        if not filtered:
            print(RED("No matching projects."))
            return
        proj     = filtered[0]
        sessions = discover_sessions(proj["path"])[: args.last]
        for s in sessions:
            messages = parse_session(s["path"], include_tools)
            md       = render_markdown(s, proj, messages, include_tools, include_timestamps)
            fname    = (f"{proj['slug'][:40]}_{s['id'][:8]}"
                        f"_{s['dt'].replace(' ', 'T').replace(':', '')}.md")
            out = output_dir / fname
            out.write_text(md, encoding="utf-8")
            print(GREEN(f"✓ {s['id'][:8]}  {s['dt']}  ({s['size_mb']:.2f}MB)  → {out}"))
        return

    # ── default: interactive hierarchy mode ────────────────────────────────
    interactive_mode(filter_str=args.project or "")


if __name__ == "__main__":
    main()
