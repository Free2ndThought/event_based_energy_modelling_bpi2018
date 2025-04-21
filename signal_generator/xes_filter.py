from os import write

import pm4py

#log = pm4py.read_xes("BPI_Challenge_2019.xes")
#pm4py.write_xes(pm4py.filtering.filter_time_range(log,"2018-01-01 00:00:00", "2018-12-31 23:59:59"), "filtered_log.xes")
log = pm4py.read_xes("filtered_log.xes")
all_variants = pm4py.get_variants(log)
i=0
name = f"variant_{i}.xes"
all_main_variants_log = pm4py.filtering.filter_variants_top_k(log,4)
main_variants = pm4py.get_variants(all_main_variants_log)
for variant in main_variants:
    i += 1
    name = f"variant_{i}.xes"
    if variant in all_variants.keys():
        print(f"variant {variant} found with {all_variants.get(variant)} occurences")
        variant_log = pm4py.filtering.filter_variants(log, [variant])
        pm4py.write_xes(variant_log, name)
    else:
        print(f"Variant {variant} not found in the log.")
        continue

print("Variants filtered and saved as xes files")