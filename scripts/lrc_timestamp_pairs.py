"""Whole-file paired timestamp detection for bilingual LRC input."""
from __future__ import annotations


def merge_timestamped_pairs(entries):
    """Only merge same-time display rows when the file is predominantly paired."""
    def is_pair(index):
        return (
            index + 1 < len(entries)
            and entries[index].source_time_cs is not None
            and entries[index].source_time_cs == entries[index + 1].source_time_cs
            and len(entries[index].lines) == len(entries[index + 1].lines) == 1
        )

    pair_count = 0
    index = 0
    while index < len(entries):
        if is_pair(index):
            pair_count += 1
            index += 2
        else:
            index += 1
    if pair_count < 4 or pair_count * 2 <= len(entries) * 0.60:
        return entries

    result = []
    index = 0
    while index < len(entries):
        if is_pair(index):
            result.append(type(entries[index])(
                [*entries[index].lines, *entries[index + 1].lines],
                entries[index].source_time_cs,
            ))
            index += 2
        else:
            result.append(entries[index])
            index += 1
    return result
