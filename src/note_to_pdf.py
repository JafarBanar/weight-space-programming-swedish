"""Build RESEARCH_NOTE.pdf from RESEARCH_NOTE.md, so that the PDF is exactly the note (needs XeLaTeX).
  python src/note_to_pdf.py
Handles the Markdown used in the note: headings, bold, italics, code, links, one table, one image, numbered lists."""
import os, re, subprocess, tempfile, shutil

REPO = "https://github.com/JafarBanar/weight-space-programming-swedish/blob/main/"
HEAD = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=1.8cm]{geometry}
\usepackage{fontspec}
\usepackage{graphicx,booktabs,array,xcolor,enumitem,titlesec,tabularx,microtype}
\usepackage[colorlinks=true,urlcolor=blue!55!black,linkcolor=black]{hyperref}
\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}
\titleformat{\section}{\large\bfseries}{}{0em}{}
\titlespacing*{\section}{0pt}{8pt}{2pt}
\setlist{itemsep=1.5pt,topsep=2pt,leftmargin=1.5em}
\pagestyle{plain}
\begin{document}
"""


def inline(text):
    keep = []

    def stash(tex):
        keep.append(tex); return f"\x00{len(keep) - 1}\x00"

    def esc(t):
        t = t.replace("\\", r"\textbackslash{}")
        for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
            t = t.replace(a, b)
        return t

    def link(m):
        label, url = m.group(1), m.group(2)
        if not url.startswith("http"):
            url = REPO + url
        return stash(r"\href{" + url.replace("%", r"\%").replace("#", r"\#") + "}{" + inline(label) + "}")
    text = re.sub(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", link, text)
    text = re.sub(r"`([^`]+)`", lambda m: stash(r"{\small\texttt{" + esc(m.group(1)) + "}}"), text)
    text = esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"\*(.+?)\*", r"\\emph{\1}", text)
    text = re.sub(r'"([^"]*)"', r"``\1''", text)
    text = text.replace("→", r"$\rightarrow$").replace("−", r"$-$")
    return re.sub("\x00(\\d+)\x00", lambda m: keep[int(m.group(1))], text)


def convert(md):
    out, lines, i = [HEAD], md.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("# "):
            out.append(r"{\Large\bfseries\raggedright " + inline(ln[2:]) + r"\par}\vspace{3pt}")
        elif ln.startswith("## "):
            out.append(r"\section{" + inline(ln[3:]) + "}")
        elif ln.startswith("!["):
            m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", ln)
            out.append(r"\begin{center}\includegraphics[width=0.56\textwidth]{" + os.path.abspath(m.group(1)) + r"}\end{center}")
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
            i -= 1
            body = [r for r in rows[2:]]
            out.append(r"{\small\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{3.3cm}X@{}}\toprule")
            out.append(" & ".join(inline(c) for c in rows[0]) + r"\\\midrule")
            out += [" & ".join(inline(c) for c in r) + r"\\" for r in body]
            out.append(r"\bottomrule\end{tabularx}}")
        elif re.match(r"\d+\. ", ln):
            out.append(r"\begin{enumerate}")
            while i < len(lines) and re.match(r"\d+\. ", lines[i]):
                out.append(r"\item " + inline(re.sub(r"^\d+\. ", "", lines[i]))); i += 1
            i -= 1
            out.append(r"\end{enumerate}")
        elif ln.strip():
            out.append(inline(ln) + r"\par")
        i += 1
    out.append(r"\end{document}")
    return "\n".join(out) + "\n"


def main():
    tex = convert(open("RESEARCH_NOTE.md", encoding="utf-8").read())
    tmp = tempfile.mkdtemp()
    open(os.path.join(tmp, "note.tex"), "w", encoding="utf-8").write(tex)
    for _ in range(2):
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "note.tex"], cwd=tmp, capture_output=True, text=True)
    log = open(os.path.join(tmp, "note.log"), errors="ignore").read()
    problems = [l for l in log.splitlines() if l.startswith("!") or "Missing character" in l or "Overfull" in l]
    shutil.copy(os.path.join(tmp, "note.pdf"), "RESEARCH_NOTE.pdf")
    pages = re.search(r"Output written on note.pdf \((\d+) page", log)
    print("RESEARCH_NOTE.pdf:", pages.group(1) if pages else "?", "pages;", len(problems), "LaTeX problems")
    for p in problems[:8]:
        print("  ", p[:160])


if __name__ == "__main__":
    main()
