# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Client Utilities for Terminal Output and Logging
=================================================

This module provides utilities for formatted terminal output during inference,
including colorized logging, token printing with word-wrapping, and visual
feedback for real-time audio processing status.

The module implements two printer classes:
- RawPrinter: Simple output without formatting (for non-TTY environments)
- Printer: Rich formatted output with borders, word-wrapping, and status indicators

These utilities are used by the local inference scripts to provide visual feedback
during real-time audio conversations with the Moshi model.
"""

from dataclasses import dataclass
import sys


def colorize(text, color):
    """
    Apply ANSI color codes to text for terminal output.
    
    Uses ANSI escape sequences to colorize text in terminals that support it.
    The color is applied at the start and reset at the end of the text.
    
    Args:
        text: The string to colorize
        color: ANSI color code string (e.g., "1;31" for bold red, "1;34" for bold blue)
    
    Returns:
        The text wrapped with ANSI escape sequences for the specified color
    
    Example:
        >>> colorize("Warning", "1;31")  # Bold red warning text
        '\033[1;31mWarning\033[0m'
    """
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def make_log(level: str, msg: str) -> str:
    """
    Create a formatted log message with a colorized prefix based on severity level.
    
    Generates a log message string with an appropriate colored prefix:
    - "warning": Bold red "Warning:" prefix
    - "info": Bold blue "Info:" prefix  
    - "error": Bold red "Error:" prefix
    
    Args:
        level: The log severity level ("warning", "info", or "error")
        msg: The message content to log
    
    Returns:
        A formatted string with colorized prefix followed by the message
    
    Raises:
        ValueError: If an unknown log level is provided
    
    Example:
        >>> make_log("info", "Model loaded successfully")
        '\033[1;34mInfo:\033[0m Model loaded successfully'
    """
    if level == "warning":
        prefix = colorize("Warning:", "1;31")
    elif level == "info":
        prefix = colorize("Info:", "1;34")
    elif level == "error":
        prefix = colorize("Error:", "1;31")
    else:
        raise ValueError(f"Unknown level {level}")
    return prefix + " " + msg


class RawPrinter:
    """
    Simple printer for non-TTY environments without rich formatting.
    
    This printer outputs tokens directly to stdout without any special formatting,
    word-wrapping, or visual decorations. It's suitable for piping output to files
    or when running in environments that don't support ANSI escape codes.
    
    The RawPrinter provides the same interface as the Printer class but with
    minimal formatting, making it interchangeable in code that needs to support
    both TTY and non-TTY output modes.
    
    Attributes:
        stream: Output stream for tokens (default: stdout)
        err_stream: Output stream for logs and errors (default: stderr)
    """
    
    def __init__(self, stream=sys.stdout, err_stream=sys.stderr):
        """
        Initialize the RawPrinter with output streams.
        
        Args:
            stream: The stream for token output (default: sys.stdout)
            err_stream: The stream for log messages (default: sys.stderr)
        """
        self.stream = stream
        self.err_stream = err_stream

    def print_header(self):
        """Print a header line. No-op for RawPrinter."""
        pass

    def print_token(self, token: str):
        """
        Output a token directly to the stream without formatting.
        
        Args:
            token: The text token to print
        """
        self.stream.write(token)
        self.stream.flush()

    def log(self, level: str, msg: str):
        """
        Print a log message to the error stream.
        
        Args:
            level: The log level (info, warning, error)
            msg: The message to log
        """
        print(f"{level.capitalize()}: {msg}", file=self.err_stream)

    def print_lag(self):
        """
        Print a lag indicator when audio processing falls behind real-time.
        
        This is displayed in red to alert the user that the system is not
        keeping up with the audio stream.
        """
        self.err_stream.write(colorize(" [LAG]", "31"))
        self.err_stream.flush()

    def print_pending(self):
        """Print a pending indicator. No-op for RawPrinter."""
        pass


@dataclass
class LineEntry:
    """
    A single entry in a formatted output line with optional color.
    
    LineEntry represents a segment of text that can be rendered with or without
    ANSI color codes. Multiple LineEntry objects are combined to form a complete
    output line in the Printer class.
    
    Attributes:
        msg: The text content of this entry
        color: Optional ANSI color code string (e.g., "31" for red)
    """
    msg: str
    color: str | None = None

    def render(self):
        """
        Render the entry as a string, applying color if specified.
        
        Returns:
            The message text, optionally wrapped with ANSI color codes
        """
        if self.color is None:
            return self.msg
        else:
            return colorize(self.msg, self.color)

    def __len__(self):
        """Return the display length of the entry (excluding color codes)."""
        return len(self.msg)


class Line:
    """
    Manages a single line of formatted terminal output with editing capabilities.
    
    The Line class handles the complexity of updating terminal output in-place,
    supporting operations like adding text, erasing entries, and managing padding
    to ensure clean visual output. It tracks the maximum line length to properly
    clear previous content when updating.
    
    This is used by the Printer class to implement word-wrapping and in-place
    updates for the real-time token display.
    
    Attributes:
        stream: The output stream to write to
        _line: List of LineEntry objects comprising the current line
        _has_padding: Whether trailing padding spaces have been added
        _max_line_length: Maximum length this line has reached (for clearing)
    """
    
    def __init__(self, stream):
        """
        Initialize a Line with an output stream.
        
        Args:
            stream: The output stream (e.g., sys.stdout) to write to
        """
        self.stream = stream
        self._line: list[LineEntry] = []
        self._has_padding: bool = False
        self._max_line_length = 0

    def __bool__(self):
        """Return True if the line has any entries."""
        return bool(self._line)

    def __len__(self):
        """Return the total display length of all entries in the line."""
        return sum(len(entry) for entry in self._line)

    def add(self, msg: str, color: str | None = None) -> int:
        """
        Add a new text entry to the line.
        
        Args:
            msg: The text to add
            color: Optional ANSI color code
        
        Returns:
            The length of the added entry
        """
        entry = LineEntry(msg, color)
        return self._add(entry)

    def _add(self, entry: LineEntry) -> int:
        """
        Internal method to add a LineEntry and update the display.
        
        Handles clearing padding if present and updates the maximum line length.
        
        Args:
            entry: The LineEntry to add
        
        Returns:
            The length of the added entry
        """
        if self._has_padding:
            self.erase(count=0)
        self._line.append(entry)
        self.stream.write(entry.render())
        self._max_line_length = max(self._max_line_length, len(self))
        return len(entry)

    def erase(self, count: int = 1):
        """
        Erase the last N entries from the line and redraw.
        
        Uses carriage return to move to the start of the line, then redraws
        all entries except the last 'count' entries.
        
        Args:
            count: Number of entries to erase from the end (0 = redraw all)
        """
        if count:
            entries = list(self._line[:-count])
        else:
            entries = list(self._line)
        self._line.clear()
        self.stream.write("\r")
        for entry in entries:
            self._line.append(entry)
            self.stream.write(entry.render())

        self._has_padding = False

    def newline(self):
        """
        Complete the current line and move to a new line.
        
        Pads the line to the maximum length to clear any previous content,
        then outputs a newline and resets the line state.
        """
        missing = self._max_line_length - len(self)
        if missing > 0:
            self.stream.write(" " * missing)
        self.stream.write("\n")
        self._line.clear()
        self._max_line_length = 0
        self._has_padding = False

    def flush(self):
        """
        Flush the output stream, adding padding if needed.
        
        Ensures any buffered output is written and pads to the maximum
        line length to maintain consistent visual appearance.
        """
        missing = self._max_line_length - len(self)
        if missing > 0:
            self.stream.write(" " * missing)
            self._has_padding = True
        self.stream.flush()


class Printer:
    """
    Rich formatted printer for TTY environments with word-wrapping and visual feedback.
    
    The Printer class provides a visually appealing output format for real-time
    token display during audio conversations. It features:
    
    - Bordered output area with configurable width
    - Automatic word-wrapping that respects word boundaries
    - Colorized status indicators (LAG, pending spinner)
    - In-place updates for smooth visual feedback
    
    The output format looks like:
    ```
     -------------------------------------------------------------------------------- 
    | Generated text appears here with automatic word wrapping when lines get too   |
    | long. The system handles word boundaries intelligently.                        |
    ```
    
    Attributes:
        max_cols: Maximum column width for the output area
        line: The Line object managing the current output line
        stream: Output stream for tokens
        err_stream: Output stream for log messages
        _pending_count: Counter for the spinning pending indicator
        _pending_printed: Whether a pending indicator is currently displayed
    """
    
    def __init__(self, max_cols: int = 80, stream=sys.stdout, err_stream=sys.stderr):
        """
        Initialize the Printer with output configuration.
        
        Args:
            max_cols: Maximum width of the output area (default: 80)
            stream: Output stream for tokens (default: stdout)
            err_stream: Output stream for logs (default: stderr)
        """
        self.max_cols = max_cols
        self.line = Line(stream)
        self.stream = stream
        self.err_stream = err_stream
        self._pending_count = 0
        self._pending_printed = False

    def print_header(self):
        """
        Print the header border and start the first content line.
        
        Creates a visual border at the top of the output area:
        ```
         -------------------------------------------------------------------------------- 
        | 
        ```
        """
        self.line.add(" " + "-" * (self.max_cols) + " ")
        self.line.newline()
        self.line.flush()
        self.line.add("| ")

    def _remove_pending(self) -> bool:
        """
        Remove the pending spinner indicator if one is displayed.
        
        Returns:
            True if a pending indicator was removed, False otherwise
        """
        if self._pending_printed:
            self._pending_printed = False
            self.line.erase(1)
            return True
        return False

    def print_token(self, token: str, color: str | None = None):
        """
        Print a token with intelligent word-wrapping.
        
        This method handles the complexity of fitting tokens within the bordered
        output area while respecting word boundaries. When a token would exceed
        the line width, it either:
        
        1. If the token starts with a space: wrap to the next line
        2. If the token is part of a word: move the entire word to the next line
        3. If no word boundary is found: split the token at the line boundary
        
        This ensures readable output where words aren't split mid-word when possible.
        
        Args:
            token: The text token to print
            color: Optional ANSI color code for the token
        """
        self._remove_pending()
        remaining = self.max_cols - len(self.line)
        if len(token) <= remaining:
            # Token fits on current line
            self.line.add(token, color)
        else:
            # Token doesn't fit - need to wrap
            end = " " * remaining + " |"
            if token.startswith(" "):
                # Token starts with space - clean break point
                token = token.lstrip()
                self.line.add(end)
                self.line.newline()
                self.line.add("| ")
                self.line.add(token, color)
            else:
                # Token is continuation of a word - try to move whole word to next line
                assert color is None
                erase_count = None
                cumulated = ""
                # Search backwards for a word boundary (space) or colored entry (like LAG)
                for idx, entry in enumerate(self.line._line[::-1]):
                    if entry.color:
                        # Colored entry (probably a LAG message) - break here
                        erase_count = idx
                        break
                    if entry.msg.startswith(" "):
                        # Found a space - this is our word boundary
                        erase_count = idx + 1
                        cumulated = entry.msg + cumulated
                        break
                if erase_count is not None:
                    # Move the partial word to the next line
                    if erase_count > 0:
                        self.line.erase(erase_count)
                    remaining = self.max_cols - len(self.line)
                    end = " " * remaining + " |"
                    self.line.add(end)
                    self.line.newline()
                    self.line.add("| ")
                    token = cumulated.lstrip() + token
                    self.line.add(token)
                else:
                    # No word boundary found - hard split at line boundary
                    self.line.add(token[:remaining])
                    self.line.add(" |")
                    self.line.newline()
                    self.line.add("| ")
                    self.line.add(token[remaining:])
        self.line.flush()

    def log(self, level: str, msg: str):
        """
        Print a log message, properly handling the current line state.
        
        Ensures the current line is completed before printing the log message
        to stderr, maintaining clean visual output.
        
        Args:
            level: The log level (info, warning, error)
            msg: The message to log
        """
        msg = make_log(level, msg)
        self._remove_pending()
        if self.line:
            self.line.newline()
        self.line.flush()
        print(msg, file=self.err_stream)
        self.err_stream.flush()

    def print_lag(self):
        """
        Print a red [LAG] indicator when audio processing falls behind.
        
        This visual indicator alerts the user that the system is not keeping
        up with real-time audio processing, which may result in choppy output.
        """
        self.print_token(" [LAG]", "31")

    def print_pending(self):
        """
        Print an animated spinner to indicate the model is processing.
        
        Displays a rotating character (|, /, -, \) that cycles through
        green, yellow, and red colors to provide visual feedback that
        the system is working on generating the next token.
        """
        chars = ["|", "/", "-", "\\"]
        count = int(self._pending_count / 5)
        char = chars[count % len(chars)]
        colors = ["32", "33", "31"]  # green, yellow, red
        self._remove_pending()
        self.line.add(char, colors[count % len(colors)])
        self._pending_printed = True
        self._pending_count += 1


# Type alias for either printer type, enabling polymorphic usage
AnyPrinter = Printer | RawPrinter
