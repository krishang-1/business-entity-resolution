import os
import pyarrow as pa
import pyarrow.csv as pacsv


def read_tsv(path, columns=None):
    """Read a challenge TSV as all-string pandas DataFrame (no quoting, no NA inference)."""
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
    tbl = pacsv.read_csv(
        path,
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False, newlines_in_values=False),
        convert_options=pacsv.ConvertOptions(
            column_types={c: pa.string() for c in header},
            strings_can_be_null=False,
            quoted_strings_can_be_null=False,
            null_values=[],
            include_columns=columns,
        ),
        read_options=pacsv.ReadOptions(block_size=1 << 24),
    )
    df = tbl.to_pandas(types_mapper=lambda t: None)
    return df


def write_id_lists(path, s1_ids, lists, col):
    """Write source1_entity_id<TAB><col> with comma-joined ids; lists aligned to s1_ids."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s1, ids in zip(s1_ids, lists):
            if ids:
                seen = dict.fromkeys(ids)
                f.write(f"{s1}\t{','.join(seen)}\n")
            else:
                f.write(f"{s1}\t\n")
