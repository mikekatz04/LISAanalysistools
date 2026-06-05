"""Command-line entry points for lisatools.

Each submodule provides a ``main()`` callable wired into ``pyproject.toml``'s
``[project.scripts]`` table -- e.g. ``lisatools_citations`` ->
:func:`lisatools.cmd.citations.main`. Pattern lifted from FastEMRIWaveforms'
``few.cmd`` package.
"""
