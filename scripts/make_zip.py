import sys, zipfile, os
name = sys.argv[1] if len(sys.argv) > 1 else "team"
zp = f"{name}_submission.zip"
files = [("output/matching_results.tsv", "output/matching_results.tsv"), ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
         ("code/business_entity_resolution/README.md", None), ("code/business_entity_resolution/requirements.txt", None),
         ("Documentation_template.md", None)]
files += [(f"code/business_entity_resolution/src/{f}", None) for f in sorted(os.listdir("code/business_entity_resolution/src"))
          if f.endswith(".py") and f != "eda.py"] + [("code/business_entity_resolution/src/eda.py", None)]
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for src, arc in files:
        z.write(src, arc or src)
print(zp, os.path.getsize(zp))
