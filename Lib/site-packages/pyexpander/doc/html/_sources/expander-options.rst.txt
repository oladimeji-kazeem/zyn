expander.py command line options
================================

``-h``
++++++

    Show help for command line options.

``--summary``
+++++++++++++

    Print a summary of the function of the program.

``-f FILE --file FILE``
+++++++++++++++++++++++

    Specify a FILE to process. This option may be used more than once to
    process more than one file but note than this option is not really needed.
    Files can also be specified directly after the other command line options.
    If not given, the program gets it's input from stdin.

``--encoding``
++++++++++++++
    Specify the encoding of the input to the program. This encoding is also the
    default for all files that are read by ``$include()`` or ``$subst()``. Note
    that both mentioned commands allow to specify a different encoding at each
    call. Known encodings can be found at 
    `python encodings <https://docs.python.org/3/library/codecs.html#standard-encodings>`_.

``--output-encoding``
+++++++++++++++++++++
    Specify the encoding of the output of the program. Known encodings can be
    found at 
    `python encodings <https://docs.python.org/3/library/codecs.html#standard-encodings>`_.

``--eval``
++++++++++

    Evaluate PYTHONEXPRESSION in global context.

``-I PATH --include PATH``
++++++++++++++++++++++++++

    Add PATH to the list of include paths.

``-s --simple-vars``
++++++++++++++++++++

    Allow variables without brackets.

``-a --auto-continuation``
++++++++++++++++++++++++++

    Assume '\\' at the end of lines with commands.

``-i --auto-indent``
++++++++++++++++++++
    
    Automatically indent macros.

``--no-stdin-msg``
++++++++++++++++++

    Do not print a message on stderr when the program is reading it's input
    from stdin.

``--deps``
++++++++++

    Print dependencies due to $include commands in a makefile compatible
    format.

``-d --dump``
+++++++++++++

    Dump list of strings instead of printing to the console, this is for
    debugging only.

