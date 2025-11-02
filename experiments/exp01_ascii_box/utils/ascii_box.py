def box_word(w: str) -> str:
    n = len(w)
    border = "+" + ("-" * n) + "+"
    return "\n".join([border, f"|{w}|", border])


def format_line(line: str) -> str:
    words = [w for w in line.strip().split() if w]
    return "\n".join(box_word(w) for w in words)
