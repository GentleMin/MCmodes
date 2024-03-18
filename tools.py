# -*- coding: utf-8 -*-
"""Practical tools for inputing and outputing
"""


from typing import Optional, TextIO, Literal
import functools


"""
=======================================
I/O tools
=======================================
"""

def str_to_print(func):
    """Decorator function for adding additional printing option
    """
    @functools.wraps(func)
    def print_str(*args, file: Optional[TextIO] = None, **kwargs):
        str_form = func(*args, **kwargs)
        print(str_form, file=file)
    return print_str


def str_repeat(pattern: str, N: int) -> str:
    """Form a string of repeating patterns"""
    return pattern*N

DEFAULT_LEN_HLINE = 64

def str_hline(char: str = '=', N: int = DEFAULT_LEN_HLINE) -> str:
    """Form horizontal line"""
    return str_repeat(char, N)

def str_tab_indent(N: int) -> str:
    """Form indentation block (tab)"""
    return str_repeat('\t', N)

def str_heading(heading: str, prefix: str = '', suffix: str = '', 
    lines: Literal["none", "over", "under", "both"] = "both", **linestyle) -> str:
    """Form headings"""
    prev = prefix + str_hline(**linestyle) + '\n' if lines in ("over", "both") else prefix
    post = '\n' + str_hline(**linestyle) + suffix if lines in ("under", "both") else suffix
    return f"{prev}{heading}{post}"
    
func_list_str = (
    str_repeat, str_hline, str_heading
)

for func_str in func_list_str:
    assert func_str.__name__[:3] == "str"
    func_print_name = list(func_str.__name__.split('_'))
    func_print_name[0] = "print"
    func_print_name = '_'.join(func_print_name)
    globals()[func_print_name] = str_to_print(func_str)
