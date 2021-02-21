def progress_bar(length, current_percent, text="Progress"):
    progress = f"{text}: " + "[" + "#" * current_percent + " " * (length - current_percent) + "]"
    print(progress, end="\r")