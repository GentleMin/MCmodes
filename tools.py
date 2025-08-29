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



"""
=======================================
Timing
=======================================
"""

import time

class ProcTimer:
    
    def __init__(self, start: bool = False) -> None:
        self._starttime = None
        self._logtimes = None
        self._loginfos = None
        if start:
            self.start()
        
    def start(self, num: bool = True) -> None:
        self._starttime = time.perf_counter()
        self._logtimes = list((self._starttime,))
        if num:
            self._loginfos = list((0,))
        else:
            self._loginfos = list(('start',))
            
    def clear(self) -> None:
        self._starttime = None
        self._logtimes = None
        self._loginfos = None
    
    def flag(self, loginfo=None, print_str: bool = False, **kwargs) -> None:
        self._logtimes.append(time.perf_counter())
        self._loginfos.append(loginfo)
        if print_str:
            self.print_elapse(**kwargs)
        
    def elapse_time(self, increment: bool = True) -> float:
        if increment:
            return self._logtimes[-1] - self._logtimes[-2]
        else:
            return self._logtimes[-1] - self._logtimes[0]
    
    def print_elapse(self, mode: Literal['0', '+', '0+'] = '0', **kwargs) -> None:
        t_inc = self.elapse_time(increment=True)
        t_tot = self.elapse_time(increment=False)
        t_info = self._loginfos[-1]
        if mode == '+':
            print("Elapse time (+) = {:8.2f} | Info: {}".format(t_inc, t_info), **kwargs)
        elif mode == '0':
            print("Elapse time (0) = {:8.2f} | Info: {}".format(t_tot, t_info), **kwargs)
        elif mode == '0+':
            print("Elapse time = {:8.2f} ({:+8.2f}) | Info: {}".format(t_tot, t_inc, t_info), **kwargs)
