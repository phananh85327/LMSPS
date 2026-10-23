archs = {
    "dblp": [
        ['AP', 'APT', 'APVP', 'APAPA', 'APTPA', 'APTPT', 'APTPV', 'APVPA', 'APVPV', 'APAPAP', 'APAPTP', 'APAPVP',
         'APTPTP', 'APTPVP', 'APVPTP', 'APAPAPV', 'APAPTPA', 'APAPTPV', 'APAPVPA', 'APAPVPT', 'APAPVPV', 'APTPAPA',
         'APTPAPT', 'APTPTPT', 'APTPVPV', 'APVPAPT', 'APVPTPA', 'APVPVPA', 'APVPVPT', 'APVPVPV'], []],
    "imdb": [
        ['M', 'MA', 'MK', 'MAM', 'MDM', 'MKM', 'MAMK', 'MDMK', 'MKMD', 'MDMKM', 'MKMAM', 'MKMDM', 'MKMKM', 'MAMAMK',
         'MAMDMA', 'MAMDMK', 'MAMKMD', 'MAMKMK', 'MDMAMA', 'MDMAMD', 'MDMDMA', 'MDMDMK', 'MDMKMA', 'MKMAMA', 'MKMAMD',
         'MKMAMK', 'MKMDMA', 'MKMDMK', 'MKMKMD', 'MKMKMK'], []],
    "acm": [
        ['PPP', 'PAPP', 'PCPA', 'PCPP', 'PPPC', 'PPPP', 'PAPAP', 'PAPPP', 'PCPAP', 'PCPPA', 'PPAPA', 'PPAPC', 'PPAPP',
         'PAPAPA', 'PAPCPA', 'PAPPAP', 'PAPPCP', 'PAPPPP', 'PCPAPP', 'PCPCPP', 'PCPPAP', 'PCPPPP', 'PPAPAP', 'PPAPCP',
         'PPAPPA', 'PPAPPP', 'PPCPAP', 'PPPAPA', 'PPPCPA', 'PPPPPP'], []],
    # Discovered sets. Record provenance so retrain/downstream can use the SAME
    # --split-seed as the search (the final set is chosen by validation BPR on
    # that split's val edges). "?" = not recorded at search time -- fill in.
    # provenance: split-seed ?, stage ?, ns 5, num_final ?, dedupe-reversed ?
    "amazon_lp_han": [
        ['PCPPUP', 'PPPPUP', 'PPUPUP', 'PUPCCP', 'PUPCPP'], []],
    # provenance: split-seed ?, stage ?, ns 5, num_final ?, dedupe-reversed ?
    "amazon_lp_han2": [
        ['PCPPUP', 'PPPPUP', 'PPUPPP', 'PPUPUP', 'PUPPCP'], []],
    # provenance: split-seed ?, stage ?, ns 5, num_final ?, dedupe-reversed ?
    "amazon_lp_han3": [
        ['PUPP', 'PCPCP', 'PPCCCP', 'PPCPUP', 'PPUPUP'], []],
    # hand-engineered baseline (what an expert would define with NO discovery):
    # co-purchase, same-category, co-reviewed, 2-hop co-purchase, sibling-category.
    # 5 paths to match the discovered amazon_lp_han count, so the ablation
    # "does AUTOMATIC path discovery beat manual paths?" compares path QUALITY at
    # an equal path budget -- see random_paths.py for the matching random row.
    "amazon_lp_manual": [
        ['PP', 'PCP', 'PUP', 'PPP', 'PCCP'], []],
    # ---- Fleet v2 final schedule (2026-10-22): 30 seeds per ns, split-seed 1,
    # stage 150, 36 reverse-deduplicated candidates, warm-up 15, exhaustive
    # noise-free selection; 1-SE re-selection (delta 0.010, leave-one-out vote;
    # source: logs/Fleet v2/final_ns4|final_ns6/reselect_d010.json "shortlist").
    # "xN" = number of the 30 runs whose 1-SE pick is this set. "_c" = the
    # assembled path-frequency consensus (the ns most frequent paths), which is
    # not necessarily any run's pick. Core relations {PPUPUP, PUPPUP, PUPUP}
    # appear in nearly every pick; the remaining slots are fillers with ~0
    # marginal effect. Downstream MUST use --split-seed 1. Chat log #18, Add. 9-10.

    "amazon_lp_unique_strings": [
        ['PCCPUP', 'PCPPUP', 'PCPUP', 'PCPUPP', 'PPCPUP', 'PPPPUP', 'PPPUP', 'PPPUPP', 'PPUP', 'PPUPP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    # ns 4 shortlist (6 sets); _c is also the most frequent pick
    "amazon_lp_f4_s1": [          # x5, consensus
        ['PPPUPP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    "amazon_lp_f4_s2": [         # x4
        ['PPUPP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    "amazon_lp_f4_s3": [         # x4
        ['PCCPUP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    "amazon_lp_f4_s4": [         # x3
        ['PCPUPP', 'PPPUPP', 'PPUPUP', 'PUPPUP'], []],
    "amazon_lp_f4_s5": [         # x3
        ['PCCPUP', 'PPPUPP', 'PPUPUP', 'PUPPUP'], []],
    "amazon_lp_f4_s6": [         # x2
        ['PCPUP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    # ns 6 shortlist (7 sets); here the consensus (_c) was itself picked only x2,
    # so _s1 (x5) is the better single representative
    "amazon_lp_f6_s1": [         # x5, most frequent pick
        ['PCPPUP', 'PPPPUP', 'PPPUPP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    "amazon_lp_f6_s2": [         # x3
        ['PPCPUP', 'PPPPUP', 'PPPUPP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    "amazon_lp_f6_s3": [         # x2
        ['PPPPUP', 'PPPUPP', 'PPUP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    "amazon_lp_f6_s4": [          # x2, consensus
        ['PPPPUP', 'PPPUPP', 'PPUPP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    "amazon_lp_f6_s5": [         # x2
        ['PPCPUP', 'PPPUPP', 'PPUPP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    "amazon_lp_f6_s6": [         # x2
        ['PPPPUP', 'PPUP', 'PPUPP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
    "amazon_lp_f6_s7": [         # x2
        ['PPPPUP', 'PPPUP', 'PPPUPP', 'PPUPUP', 'PUPPUP', 'PUPUP'], []],
}
