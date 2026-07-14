"""Optional detection backends.

These wrap external tools (Omnipose for cells, ilastik for pili) behind the
callable/stack interfaces that ``pipeline.analyze_movie`` accepts. Neither is a
hard dependency of pilitrack -- imports are guarded so the core installs and
runs without them.
"""
