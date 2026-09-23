# Phase30b: execute Phase30 source with corrected walk-forward fold tuples
src=open('sar_phase30_entry_filters.py').read()
src=src.replace("for fi,(te,a,b) in enumerate(folds,1):", "for fi,(a,b) in enumerate(folds,1):\n   te=a")
exec(compile(src,'sar_phase30_entry_filters.py','exec'))
